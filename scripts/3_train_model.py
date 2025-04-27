"""
Train DistilBERT Model for Bot Detection

This script fine-tunes a pre-trained DistilBERT model on the Twibot-20 dataset
for the task of bot detection (sequence classification).
"""

import os
import numpy as np
import torch
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
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
    output_dir = os.path.join(project_root, "models", "distilbert-bot-detector")

    print("[3/4] Training model...")

    # Check if fixed tokenized dataset exists, if not, fall back to original
    if not os.path.exists(tokenized_dataset_path):
        print(f"Warning: Fixed tokenized dataset not found at {tokenized_dataset_path}")
        print("Falling back to original tokenized dataset...")
        tokenized_dataset_path = os.path.join(project_root, "data", "twibot20_tokenized")

    # Check if tokenized dataset exists
    if not os.path.exists(tokenized_dataset_path):
        print(f"Error: Tokenized dataset not found at {tokenized_dataset_path}")
        print("Please run tokenize_dataset.py first to create the tokenized dataset.")
        return

    # 1. Load tokenized dataset
    print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
    try:
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

    print("\nTraining and evaluation completed!")
    print(f"The fine-tuned model is saved at: {output_dir}")
    print("You can now use this model for inference on new data.")
    print("Run 'python scripts/4_predict.py' to make predictions with the trained model.")

if __name__ == "__main__":
    main()
