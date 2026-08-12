import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import (
    FIGURE_DIR,
    ID_TO_LABEL,
    LABEL_NAMES,
    MAX_LENGTH,
    METRICS_DIR,
    MODEL_NAME,
)
from src.data_preprocessing import get_tokenizer, load_raw_data
from src.text_cleaning import BROKEN_HTML_ENTITY, clean_text


WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]+")


def add_length_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add character and word counts used in the exploration."""
    result = frame.copy()
    result["missing_text"] = result["text"].isna()
    result["text"] = result["text"].fillna("").map(clean_text)
    result["character_count"] = result["text"].str.len()
    result["word_count"] = result["text"].str.split().str.len()
    result["label_name"] = result["label"].map(ID_TO_LABEL)
    return result


def calculate_token_lengths(texts, tokenizer, batch_size: int = 1000) -> list[int]:
    """Measure unpadded Transformer token lengths in manageable batches."""
    lengths = []
    text_list = list(texts)
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def vocabulary_statistics(texts, top_n: int = 20) -> dict:
    """Return vocabulary size and frequent content words."""
    all_words = []
    for text in texts:
        all_words.extend(word.lower() for word in WORD_PATTERN.findall(str(text)))

    counts = Counter(all_words)
    content_counts = Counter(
        {word: count for word, count in counts.items() if word not in ENGLISH_STOP_WORDS}
    )
    return {
        "total_word_occurrences": len(all_words),
        "unique_words": len(counts),
        "top_content_words": content_counts.most_common(top_n),
    }


def build_summary(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict:
    """Calculate the main dataset statistics required by the rubric."""
    vocabulary = vocabulary_statistics(train_frame["text"])
    train_texts = set(train_frame["text"].dropna())
    test_texts = set(test_frame["text"].dropna())

    class_counts = (
        train_frame["label_name"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    )
    token_lengths = train_frame["token_count"]

    return {
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "number_of_classes": len(LABEL_NAMES),
        "class_distribution": {name: int(count) for name, count in class_counts.items()},
        "missing_train_texts": int(train_frame["missing_text"].sum()),
        "missing_test_texts": int(test_frame["missing_text"].sum()),
        "duplicate_train_texts": int(train_frame["text"].duplicated().sum()),
        "duplicate_test_texts": int(test_frame["text"].duplicated().sum()),
        "exact_train_test_overlap": int(len(train_texts.intersection(test_texts))),
        "average_words": round(float(train_frame["word_count"].mean()), 2),
        "median_words": round(float(train_frame["word_count"].median()), 2),
        "average_characters": round(float(train_frame["character_count"].mean()), 2),
        "average_tokens": round(float(token_lengths.mean()), 2),
        "median_tokens": round(float(token_lengths.median()), 2),
        "token_length_percentiles": {
            "90th": int(np.percentile(token_lengths, 90)),
            "95th": int(np.percentile(token_lengths, 95)),
            "99th": int(np.percentile(token_lengths, 99)),
            "maximum": int(token_lengths.max()),
        },
        f"token_coverage_at_{MAX_LENGTH}_percent": round(
            float((token_lengths <= MAX_LENGTH).mean() * 100), 3
        ),
        f"examples_truncated_at_{MAX_LENGTH}": int((token_lengths > MAX_LENGTH).sum()),
        **vocabulary,
    }


def plot_class_distribution(train_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR):
    counts = train_frame["label_name"].value_counts().reindex(LABEL_NAMES)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=axis)
    axis.set_title("AG News Training Class Distribution")
    axis.set_xlabel("News category")
    axis.set_ylabel("Number of articles")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=160)
    return figure


def plot_length_distributions(train_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(train_frame["word_count"], bins=50, ax=axes[0], color="#3b82f6")
    axes[0].set_title("Article Length in Words")
    axes[0].set_xlabel("Words")

    sns.histplot(train_frame["token_count"], bins=50, ax=axes[1], color="#f97316")
    axes[1].axvline(
        MAX_LENGTH,
        color="black",
        linestyle="--",
        label=f"max_length = {MAX_LENGTH}",
    )
    axes[1].set_title("DistilBERT Token Length")
    axes[1].set_xlabel("Tokens")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "length_distributions.png", dpi=160)
    return figure


def plot_length_by_class(train_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR):
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=train_frame,
        x="label_name",
        y="word_count",
        order=LABEL_NAMES,
        showfliers=False,
        ax=axis,
    )
    axis.set_title("Article Length by News Category")
    axis.set_xlabel("News category")
    axis.set_ylabel("Words")
    figure.tight_layout()
    figure.savefig(output_dir / "length_by_class.png", dpi=160)
    return figure


def plot_top_words(train_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR):
    statistics = vocabulary_statistics(train_frame["text"], top_n=15)
    words, counts = zip(*statistics["top_content_words"])
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), legend=False, ax=axis)
    axis.set_title("Most Frequent Content Words")
    axis.set_xlabel("Frequency")
    axis.set_ylabel("Word")
    figure.tight_layout()
    figure.savefig(output_dir / "top_words.png", dpi=160)
    return figure


def plot_top_words_by_class(train_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, label_name in zip(axes.flat, LABEL_NAMES):
        class_texts = train_frame.loc[train_frame["label_name"] == label_name, "text"]
        statistics = vocabulary_statistics(class_texts, top_n=10)
        words, counts = zip(*statistics["top_content_words"])
        sns.barplot(x=list(counts), y=list(words), hue=list(words), legend=False, ax=axis)
        axis.set_title(label_name)
        axis.set_xlabel("Frequency")
        axis.set_ylabel("")
    figure.suptitle("Frequent Content Words by Category", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "top_words_by_class.png", dpi=160)
    return figure


def run_eda(model_name: str = MODEL_NAME) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run the complete EDA and save figures and summary files."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data()
    raw_train_frame = raw_data["train"].to_pandas()
    raw_test_frame = raw_data["test"].to_pandas()
    raw_artifact_count = int(
        raw_train_frame["text"].fillna("").str.count(BROKEN_HTML_ENTITY).sum()
        + raw_test_frame["text"].fillna("").str.count(BROKEN_HTML_ENTITY).sum()
    )
    train_frame = add_length_columns(raw_train_frame)
    test_frame = add_length_columns(raw_test_frame)

    tokenizer = get_tokenizer(model_name)
    train_frame["token_count"] = calculate_token_lengths(
        train_frame["text"].fillna(""), tokenizer
    )
    test_frame["token_count"] = calculate_token_lengths(
        test_frame["text"].fillna(""), tokenizer
    )

    summary = build_summary(train_frame, test_frame)
    summary["raw_html_artifacts_repaired"] = raw_artifact_count
    with open(METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    samples = (
        train_frame.groupby("label_name", group_keys=False)
        .sample(n=3, random_state=42)[["label_name", "text", "word_count", "token_count"]]
    )
    samples.to_csv(METRICS_DIR / "eda_class_samples.csv", index=False)

    plot_class_distribution(train_frame)
    plot_length_distributions(train_frame)
    plot_length_by_class(train_frame)
    plot_top_words(train_frame)
    plot_top_words_by_class(train_frame)
    plt.close("all")

    return summary, train_frame, test_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the AG News dataset.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    summary, _, _ = run_eda(args.model_name)
    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
