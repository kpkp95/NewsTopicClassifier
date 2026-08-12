import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.config import DROPOUT, LABEL_NAMES, MAX_LENGTH, MODEL_NAME


class DistilBertNewsClassifier(nn.Module):
    """DistilBERT encoder followed by dropout and a four-class linear layer."""

    def __init__(
        self,
        encoder_name_or_path: str | Path = MODEL_NAME,
        number_of_labels: int = len(LABEL_NAMES),
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(encoder_name_or_path))
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, number_of_labels)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        first_token_output = encoder_output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(first_token_output))


def save_checkpoint(
    model: DistilBertNewsClassifier,
    tokenizer,
    checkpoint_dir: str | Path,
    training_information: dict,
) -> None:
    """Save the fine-tuned encoder, classification layer, tokenizer, and settings."""
    checkpoint_dir = Path(checkpoint_dir)
    encoder_dir = checkpoint_dir / "encoder"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save(model.classifier.state_dict(), checkpoint_dir / "classifier.pt")

    project_settings = {
        "label_names": LABEL_NAMES,
        "number_of_labels": len(LABEL_NAMES),
        "dropout": DROPOUT,
        "max_length": MAX_LENGTH,
        "base_model": MODEL_NAME,
        **training_information,
    }
    with open(checkpoint_dir / "project_config.json", "w", encoding="utf-8") as file:
        json.dump(project_settings, file, indent=2)


def load_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    """Load a saved model without retraining it."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "project_config.json", encoding="utf-8") as file:
        settings = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    model = DistilBertNewsClassifier(
        encoder_name_or_path=checkpoint_dir / "encoder",
        number_of_labels=settings["number_of_labels"],
        dropout=settings["dropout"],
    )
    classifier_weights = torch.load(
        checkpoint_dir / "classifier.pt",
        map_location=device,
        weights_only=True,
    )
    model.classifier.load_state_dict(classifier_weights)
    model.to(device)
    model.eval()
    return model, tokenizer, settings
