import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from src.config import FIGURE_DIR, ID_TO_LABEL, LABEL_NAMES, METRICS_DIR, MODEL_DIR
from src.data_preprocessing import create_dataloader, prepare_dataframes, tokenize_dataframes
from src.model import load_checkpoint


def collect_predictions(model, dataloader, device):
    """Return labels, predictions, and class probabilities for a dataloader."""
    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(batch["labels"].tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def calculate_test_metrics(labels, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro")),
        "macro_recall": float(recall_score(labels, predictions, average="macro")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def save_confusion_matrix(labels, predictions) -> None:
    matrix = confusion_matrix(labels, predictions)
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=axis,
    )
    axis.set_title("DistilBERT Test Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "transformer_confusion_matrix.png", dpi=160)
    plt.close(figure)


def build_error_analysis(test_frame, labels, predictions, probabilities) -> pd.DataFrame:
    """Create an evidence table for every incorrect test prediction."""
    errors = test_frame.copy()
    errors["true_label"] = [ID_TO_LABEL[int(value)] for value in labels]
    errors["predicted_label"] = [ID_TO_LABEL[int(value)] for value in predictions]
    errors["confidence"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    errors["confidence_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    errors["word_count"] = errors["text"].str.split().str.len()
    for label_id, label_name in enumerate(LABEL_NAMES):
        column_name = f"probability_{label_name.lower().replace('/', '_')}"
        errors[column_name] = probabilities[:, label_id]

    errors = errors[labels != predictions].copy()
    errors["confusion_pair"] = errors["true_label"] + " -> " + errors["predicted_label"]
    return errors.sort_values("confidence", ascending=False).reset_index(drop=True)


def _spread_confidence_examples(group: pd.DataFrame, count: int) -> pd.DataFrame:
    """Choose examples across the confidence range instead of only the highest."""
    ordered = group.sort_values("confidence").reset_index(drop=True)
    if count >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=count)
    positions = np.rint(positions).astype(int)
    return ordered.iloc[positions]


def select_twenty_errors(errors: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
    """Select errors in proportion to the observed confusion-pair frequencies."""
    pair_counts = errors["confusion_pair"].value_counts()
    exact_quotas = pair_counts / len(errors) * sample_size
    quotas = np.floor(exact_quotas).astype(int)

    remaining = sample_size - int(quotas.sum())
    remainders = (exact_quotas - quotas).sort_values(ascending=False)
    for pair_name in remainders.index[:remaining]:
        quotas[pair_name] += 1

    selected_groups = []
    for pair_name, count in quotas.items():
        if count == 0:
            continue
        pair_errors = errors[errors["confusion_pair"] == pair_name]
        selected_groups.append(_spread_confidence_examples(pair_errors, int(count)))

    selected = pd.concat(selected_groups, ignore_index=True)
    selected["selection_note"] = (
        "Representative confusion-pair sample with varied confidence"
    )
    return selected.sort_values(
        ["confusion_pair", "confidence"], ascending=[True, False]
    ).reset_index(drop=True)


def summarize_error_patterns(errors: pd.DataFrame, all_test_lengths: pd.Series) -> dict:
    pair_counts = Counter(errors["confusion_pair"])
    business_sci_tech_errors = pair_counts["Business -> Sci/Tech"] + pair_counts[
        "Sci/Tech -> Business"
    ]
    high_confidence_errors = int((errors["confidence"] >= 0.80).sum())
    return {
        "total_incorrect_predictions": int(len(errors)),
        "most_common_confusions": dict(pair_counts.most_common(8)),
        "average_error_confidence": round(float(errors["confidence"].mean()), 4),
        "high_confidence_errors_at_least_0_80": high_confidence_errors,
        "high_confidence_error_percentage": round(
            high_confidence_errors / len(errors) * 100, 2
        ),
        "business_sci_tech_errors": int(business_sci_tech_errors),
        "business_sci_tech_error_percentage": round(
            business_sci_tech_errors / len(errors) * 100, 2
        ),
        "average_words_in_errors": round(float(errors["word_count"].mean()), 2),
        "average_words_in_all_test_examples": round(float(all_test_lengths.mean()), 2),
        "selected_error_method": (
            "Proportional confusion-pair sample with confidence-range coverage"
        ),
    }


def save_model_comparison(transformer_metrics: dict) -> None:
    baseline_path = METRICS_DIR / "baseline_metrics.json"
    if not baseline_path.exists():
        return
    with open(baseline_path, encoding="utf-8") as file:
        baseline_metrics = json.load(file)["test"]
    comparison = pd.DataFrame(
        [
            {"model": "TF-IDF + Logistic Regression", **baseline_metrics},
            {"model": "DistilBERT", **transformer_metrics},
        ]
    )
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)


def evaluate_model(batch_size: int = 32) -> dict:
    """Evaluate the saved best model on the untouched official test set."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, settings = load_checkpoint(MODEL_DIR, device)

    frames = prepare_dataframes()
    tokenized = tokenize_dataframes(
        {"test": frames["test"]},
        tokenizer,
        max_length=settings["max_length"],
    )
    test_loader = create_dataloader(tokenized["test"], tokenizer, batch_size, shuffle=False)
    labels, predictions, probabilities = collect_predictions(model, test_loader, device)

    metrics = calculate_test_metrics(labels, predictions)
    with open(METRICS_DIR / "transformer_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        labels,
        predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(METRICS_DIR / "transformer_report.csv")
    save_confusion_matrix(labels, predictions)

    errors = build_error_analysis(frames["test"], labels, predictions, probabilities)
    errors.to_csv(METRICS_DIR / "all_incorrect_predictions.csv", index=False)
    selected_errors = select_twenty_errors(errors)
    selected_errors.to_csv(METRICS_DIR / "error_analysis_20.csv", index=False)

    test_lengths = frames["test"]["text"].str.split().str.len()
    error_summary = summarize_error_patterns(errors, test_lengths)
    with open(METRICS_DIR / "error_pattern_summary.json", "w", encoding="utf-8") as file:
        json.dump(error_summary, file, indent=2)

    save_model_comparison(metrics)
    return {"test_metrics": metrics, "error_summary": error_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved DistilBERT model.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.batch_size), indent=2))


if __name__ == "__main__":
    main()
