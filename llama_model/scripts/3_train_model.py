"""
Train Llama Model for Twitter Bot Detection

This script fine-tunes a Llama model for Twitter bot detection using the tokenized Twibot-20 dataset.
It uses the Hugging Face Transformers library for training and evaluation.
"""

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
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

    # Handle T5 model output format which can be different from BERT-based models
    if isinstance(predictions, tuple):
        # Some models return a tuple of (logits, past_key_values, ...)
        predictions = predictions[0]

    # Handle potential shape issues
    if len(predictions.shape) > 2:
        # For T5, the output might be (batch_size, seq_len, num_labels)
        # We need to reshape it to (batch_size, num_labels)
        predictions = predictions[:, 0, :]

    # Get the predicted class
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
        'f1': f1
    }

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    tokenized_dataset_path = os.path.join(project_root, "..", "data", "twibot20_llama_tokenized")
    output_dir = os.path.join(project_root, "models", "llama-bot-detector")

    print("[3/4] Training Llama model...")

    # 1. Check if tokenized dataset exists
    if not os.path.exists(tokenized_dataset_path):
        print(f"Error: Tokenized dataset not found at {tokenized_dataset_path}")
        print("Please run llama_model/scripts/2_tokenize_dataset.py first to create the tokenized dataset.")
        # Fall back to the original tokenized dataset if available
        tokenized_dataset_path = os.path.join(project_root, "..", "data", "twibot20_fixed_tokenized")
        if os.path.exists(tokenized_dataset_path):
            print(f"Found original tokenized dataset at {tokenized_dataset_path}")
            print("Will use this dataset instead, but note that it was tokenized with DistilBERT, not Llama.")
        else:
            return

    # 2. Load tokenized dataset
    print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
    try:
        tokenized_dataset = load_from_disk(tokenized_dataset_path)
        print("Tokenized dataset loaded:")
        print(tokenized_dataset)
    except Exception as e:
        print(f"Error loading tokenized dataset: {e}")
        return

    # 3. Load tokenizer
    print("\nLoading T5 tokenizer (as a substitute for Llama)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        print("\nNote: Using Flan-T5 as a substitute for Llama due to access restrictions.")
        print("To use the actual Llama model, you need to request access on Hugging Face.")
        print("Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 4. Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Load model
    print("\nLoading T5 model for sequence classification (as a substitute for Llama)...")
    try:
        # Set up label mapping
        id2label = {0: "human", 1: "bot"}
        label2id = {"human": 0, "bot": 1}

        # Load model with classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/flan-t5-base",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        print("Model loaded successfully")
        print("\nNote: Using Flan-T5 as a substitute for Llama due to access restrictions.")
        print("To use the actual Llama model, you need to request access on Hugging Face.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Check for available devices and set appropriate flags for training
    use_fp16 = False
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS device found. Model will be trained on MPS.")
        # MPS doesn't support fp16 training
    elif torch.cuda.is_available():
        print("CUDA device found. Model will be trained on CUDA.")
        # Enable fp16 for CUDA
        use_fp16 = True
    else:
        print("MPS or CUDA not available. Model will be trained on CPU.")

    # 7. Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,  # Standard learning rate for T5
        per_device_train_batch_size=16,  # T5 can handle larger batch sizes
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard, etc.
        gradient_accumulation_steps=1,  # No need for gradient accumulation with T5
        fp16=use_fp16  # Enable fp16 only for CUDA
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
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 10. Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    try:
        test_results = trainer.evaluate(tokenized_dataset["test"])
        print("Test results:")
        for metric_name, metric_value in test_results.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # 11. Save the model
    print(f"\nSaving model to {output_dir}...")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nYou can now use this model for inference on new data.")
    print("Run 'python llama_model/scripts/4_predict.py' to make predictions with the trained model.")

if __name__ == "__main__":
    main()
