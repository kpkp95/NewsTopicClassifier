import argparse
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIGURE_DIR, LABEL_NAMES, METRICS_DIR, PROJECT_ROOT
from src.data_preprocessing import prepare_dataframes


def calculate_metrics(labels, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def run_baseline(max_features: int = 30_000) -> dict:
    """Train and evaluate a TF-IDF Logistic Regression baseline."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_dir = PROJECT_ROOT / "models" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    frames = prepare_dataframes()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )

    train_features = vectorizer.fit_transform(frames["train"]["text"])
    validation_features = vectorizer.transform(frames["validation"]["text"])
    test_features = vectorizer.transform(frames["test"]["text"])

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_features, frames["train"]["label"])

    validation_predictions = classifier.predict(validation_features)
    test_predictions = classifier.predict(test_features)
    results = {
        "model": "TF-IDF + Logistic Regression",
        "max_features": max_features,
        "validation": calculate_metrics(frames["validation"]["label"], validation_predictions),
        "test": calculate_metrics(frames["test"]["label"], test_predictions),
    }

    with open(METRICS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    report = classification_report(
        frames["test"]["label"],
        test_predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(METRICS_DIR / "baseline_report.csv")

    matrix = confusion_matrix(frames["test"]["label"], test_predictions)
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=axis,
    )
    axis.set_title("Baseline Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "baseline_confusion_matrix.png", dpi=160)
    plt.close(figure)

    joblib.dump(vectorizer, baseline_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, baseline_dir / "logistic_regression.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the traditional baseline model.")
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.max_features), indent=2))


if __name__ == "__main__":
    main()
