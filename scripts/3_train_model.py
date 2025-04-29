"""
Train DistilBERT Model for Bot Detection

This script fine-tunes a pre-trained DistilBERT model on the Twibot-20 dataset
for the task of bot detection (sequence classification).
The script supports loading data from both Hugging Face format and Apache Parquet format.
"""

import os
import sys
import numpy as np
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import parquet utilities
from utilities.parquet_utils import load_parquet_as_dataset

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_pred: Tuple containing predictions and labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a DistilBERT model for bot detection')
    parser.add_argument('--use-parquet', action='store_true', help='Use Parquet format instead of Hugging Face format')
    args = parser.parse_args()

    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "models", "distilbert-bot-detector")

    # Set dataset path based on format
    if args.use_parquet:
        tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_tokenized_parquet")
        print("[3/4] Training model (using Parquet format)...")
    else:
        tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
        print("[3/4] Training model (using Hugging Face format)...")

    # Check if fixed tokenized dataset exists, if not, fall back to original
    if not os.path.exists(tokenized_dataset_path):
        print(f"Warning: Dataset not found at {tokenized_dataset_path}")
        if args.use_parquet:
            print("Falling back to Hugging Face format...")
            tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
            args.use_parquet = False
        else:
            print("Falling back to original tokenized dataset...")
            tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_tokenized")

    # Check if tokenized dataset exists
    if not os.path.exists(tokenized_dataset_path):
        print(f"Error: Tokenized dataset not found at {tokenized_dataset_path}")
        print("Please run scripts/2_tokenize_dataset.py first to create the tokenized dataset.")
        return

    # 1. Load tokenized dataset
    print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
    try:
        if args.use_parquet:
            tokenized_dataset = load_parquet_as_dataset(tokenized_dataset_path)
        else:
            tokenized_dataset = load_from_disk(tokenized_dataset_path)
        print("Tokenized dataset loaded successfully")
        print(tokenized_dataset)
    except Exception as e:
        print(f"Error loading tokenized dataset: {e}")
        return

    # 2. Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Define label mappings
    id2label = {0: "human", 1: "bot"}
    label2id = {"human": 0, "bot": 1}

    # 5. Load model
    print("\nLoading model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Check for MPS device (for Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("MPS device found. Model will be trained on MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found. Model will be trained on CUDA.")
    else:
        device = torch.device("cpu")
        print("MPS or CUDA not available. Model will be trained on CPU.")

    # 7. Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # Increased from 3 to 5 epochs
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard, etc.
    )

    # 8. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 9. Train the model
    print("\nStarting training...")
    try:
        trainer.train()
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 10. Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    try:
        test_results = trainer.evaluate(tokenized_dataset["test"])
        print("Test results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # 11. Save the model
    print(f"\nSaving model to {output_dir}...")
    try:
        trainer.save_model(output_dir)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Plot training curves
    print("\nPlotting training curves...")
    try:
        train_loss = []
        eval_loss = []
        eval_accuracy = []
        eval_f1 = []

        for log in trainer.state.log_history:
            if 'loss' in log and 'epoch' in log and 'step' in log: # Filter for training steps
                # Check if it's a training log entry based on keys presence
                is_training_log = all(k in log for k in ['loss', 'learning_rate', 'epoch', 'step'])
                is_eval_log = 'eval_loss' in log

                if is_training_log and not is_eval_log:
                     # Estimate epoch fraction for smoother plotting if needed, or just use epoch
                     epoch_progress = log.get('epoch', 0)
                     train_loss.append((epoch_progress, log['loss']))

            elif 'eval_loss' in log and 'epoch' in log: # Filter for evaluation steps
                eval_loss.append((log['epoch'], log['eval_loss']))
                eval_accuracy.append((log['epoch'], log['eval_accuracy']))
                eval_f1.append((log['epoch'], log['eval_f1']))

        # Sort by epoch
        train_loss.sort(key=lambda x: x[0])
        eval_loss.sort(key=lambda x: x[0])
        eval_accuracy.sort(key=lambda x: x[0])
        eval_f1.sort(key=lambda x: x[0])

        # Create figure
        # Ensure matplotlib is imported: import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        if train_loss:
             ax1.plot([x[0] for x in train_loss], [x[1] for x in train_loss], 'b-', label='Training Loss', alpha=0.6)
        if eval_loss:
             ax1.plot([x[0] for x in eval_loss], [x[1] for x in eval_loss], 'r-o', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot metrics
        if eval_accuracy:
             ax2.plot([x[0] for x in eval_accuracy], [x[1] for x in eval_accuracy], 'g-o', label='Validation Accuracy')
        if eval_f1:
             ax2.plot([x[0] for x in eval_f1], [x[1] for x in eval_f1], 'm-o', label='Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True)

        # Ensure output directory exists (should already exist from TrainingArguments)
        plot_save_path = os.path.join(output_dir, "training_curves.png") # output_dir should be defined earlier in main()
        plt.tight_layout()
        plt.savefig(plot_save_path)
        print(f"Training curves saved to {plot_save_path}")
        plt.close(fig) # Close the plot to free memory
    except Exception as plot_err:
        print(f"Warning: Could not generate training plots. Error: {plot_err}")

    print("\nTraining and evaluation completed!")
    print(f"The fine-tuned model is saved at: {output_dir}")
    print("You can now use this model for inference on new data.")
    print("Run 'python scripts/4_predict.py' to make predictions with the trained model.")

if __name__ == "__main__":
    main()
