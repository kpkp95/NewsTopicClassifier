from pathlib import Path

import torch

from src.config import MODEL_DIR
from src.model import load_checkpoint
from src.text_cleaning import clean_text


class NewsPredictor:
    """Load the saved model once and classify new text without retraining."""

    def __init__(self, checkpoint_dir: str | Path = MODEL_DIR) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.settings = load_checkpoint(
            checkpoint_dir,
            self.device,
        )
        self.labels = self.settings["label_names"]

    def predict(self, text: str) -> dict:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            raise ValueError("Please enter some news text before predicting.")

        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.settings["max_length"],
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

        probability_map = {
            label: float(probability) for label, probability in zip(self.labels, probabilities)
        }
        predicted_label = max(probability_map, key=probability_map.get)
        return {
            "label": predicted_label,
            "confidence": probability_map[predicted_label],
            "probabilities": probability_map,
        }
