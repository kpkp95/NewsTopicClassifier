import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.config import (
    DATASET_NAME,
    MAX_LENGTH,
    MODEL_NAME,
    RANDOM_SEED,
    VALIDATION_SIZE,
)
from src.text_cleaning import clean_text


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Make data splitting and model training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_data(dataset_name: str = DATASET_NAME) -> DatasetDict:
    """Download and return the official AG News train and test splits."""
    return load_dataset(dataset_name)


def _clean_dataframe(frame: pd.DataFrame, remove_duplicates: bool) -> pd.DataFrame:
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].fillna("").map(clean_text)
    frame = frame[frame["text"].str.len() > 0]
    if remove_duplicates:
        frame = frame.drop_duplicates(subset="text", keep="first")
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def prepare_dataframes(
    raw_data: DatasetDict | None = None,
    validation_size: float = VALIDATION_SIZE,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """Clean AG News and create stratified train, validation, and test frames."""
    if raw_data is None:
        raw_data = load_raw_data()

    full_train = _clean_dataframe(raw_data["train"].to_pandas(), remove_duplicates=True)
    test_frame = _clean_dataframe(raw_data["test"].to_pandas(), remove_duplicates=False)

    train_frame, validation_frame = train_test_split(
        full_train,
        test_size=validation_size,
        random_state=seed,
        stratify=full_train["label"],
    )

    return {
        "train": train_frame.reset_index(drop=True),
        "validation": validation_frame.reset_index(drop=True),
        "test": test_frame,
    }


def get_tokenizer(model_name: str = MODEL_NAME):
    """Load the tokenizer used by DistilBERT."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataframes(
    frames: Dict[str, pd.DataFrame],
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> DatasetDict:
    """Tokenize all splits and retain labels for PyTorch."""

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = DatasetDict()
    for split_name, frame in frames.items():
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        tokenized[split_name] = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split_name}",
        )
    return tokenized


def create_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create batches with dynamic padding to reduce unnecessary computation."""
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
