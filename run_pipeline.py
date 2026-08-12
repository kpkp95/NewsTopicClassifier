import argparse

from src.baseline import run_baseline
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_LENGTH
from src.eda import run_eda
from src.evaluate import evaluate_model
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete AG News project pipeline.")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    if not args.skip_eda:
        print("\n1/4 Running exploratory data analysis")
        run_eda()
    if not args.skip_baseline:
        print("\n2/4 Training the TF-IDF baseline")
        run_baseline()
    if not args.skip_training:
        print("\n3/4 Fine-tuning DistilBERT")
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )
    if not args.skip_evaluation:
        print("\n4/4 Evaluating the best checkpoint")
        evaluate_model()

    print("\nPipeline finished. Review the outputs folder and presentation notebooks.")


if __name__ == "__main__":
    main()
