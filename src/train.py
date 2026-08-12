import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FIGURE_DIR,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_LENGTH,
    METRICS_DIR,
    MODEL_DIR,
    MODEL_NAME,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    create_output_directories,
)
from src.data_preprocessing import (
    create_dataloader,
    get_tokenizer,
    prepare_dataframes,
    set_seed,
    tokenize_dataframes,
)
from src.model import DistilBertNewsClassifier, save_checkpoint


def run_epoch(
    model,
    dataloader,
    loss_function,
    device,
    optimizer=None,
    scheduler=None,
) -> dict:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    progress = tqdm(dataloader, leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(logits, labels)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro"),
    }


def plot_training_history(history_frame: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> None:
    """Save loss and accuracy curves for the report and video."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        history_frame["epoch"], history_frame["train_loss"], marker="o", label="Train"
    )
    axes[0].plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(
        history_frame["epoch"],
        history_frame["train_accuracy"],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        history_frame["epoch"],
        history_frame["validation_accuracy"],
        marker="o",
        label="Validation",
    )
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = MAX_LENGTH,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Fine-tune DistilBERT using a manually written PyTorch loop."""
    create_output_directories()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frames = prepare_dataframes()
    tokenizer = get_tokenizer(model_name)
    tokenized = tokenize_dataframes(frames, tokenizer, max_length=max_length)
    train_loader = create_dataloader(tokenized["train"], tokenizer, batch_size, shuffle=True)
    validation_loader = create_dataloader(
        tokenized["validation"], tokenizer, batch_size, shuffle=False
    )

    model = DistilBertNewsClassifier(encoder_name_or_path=model_name).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_results = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        validation_results = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        epoch_results = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "train_macro_f1": train_results["macro_f1"],
            "validation_loss": validation_results["loss"],
            "validation_accuracy": validation_results["accuracy"],
            "validation_macro_f1": validation_results["macro_f1"],
        }
        history.append(epoch_results)
        print(json.dumps(epoch_results, indent=2))

        history_frame = pd.DataFrame(history)
        history_frame.to_csv(METRICS_DIR / "training_history.csv", index=False)
        plot_training_history(history_frame)

        if validation_results["loss"] < best_validation_loss:
            best_validation_loss = validation_results["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                tokenizer,
                MODEL_DIR,
                {
                    "best_epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                    "best_validation_accuracy": validation_results["accuracy"],
                    "best_validation_macro_f1": validation_results["macro_f1"],
                    "epochs_requested": epochs,
                    "max_length": max_length,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "base_model": model_name,
                    "checkpoint_selection": "lowest validation loss",
                },
            )
            print("Saved a new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print(f"No validation-loss improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on AG News.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
