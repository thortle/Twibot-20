"""
Train Bot Detection Model on Balanced Twibot-22 Dataset

This script trains a DistilBERT model on the tokenized balanced Twibot-22 dataset.
It loads the tokenized dataset, configures the model, trains it, and evaluates it.
The trained model is saved for later use.

Optimized for memory efficiency on Apple Silicon with 32GB RAM.
"""

import os
import json
import argparse
import numpy as np
from datasets import load_from_disk
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import sys
import gc
import psutil
import time

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
twibot22_dir = os.path.dirname(script_dir)  # One level up
sys.path.append(twibot22_dir)

# Import parquet utilities
from utilities.parquet_utils import load_parquet_as_dataset, print_memory_usage

def compute_metrics(pred):
    """
    Compute evaluation metrics for the model.

    Args:
        pred: Prediction output from the model

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def memory_monitor(threshold_percent=80, force=False):
    """
    Monitor memory usage and trigger garbage collection if it exceeds threshold

    Args:
        threshold_percent: Percentage threshold of system memory to trigger GC
        force: Whether to force garbage collection regardless of usage

    Returns:
        tuple: (memory_mb, memory_percent)
    """
    mem_info = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)

    if mem_info.percent > threshold_percent or force:
        print(f"Memory usage: {memory_mb:.2f} MB, System: {mem_info.percent}%, triggering garbage collection...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Get updated memory info after GC
        mem_info = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        print(f"After GC - Memory usage: {memory_mb:.2f} MB, System: {mem_info.percent}%")

    return memory_mb, mem_info.percent

class MemoryEfficientTrainer(Trainer):
    """
    A trainer that monitors and manages memory during training and evaluation
    """
    def __init__(self, memory_threshold=80, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_threshold = memory_threshold

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to monitor memory"""
        memory_monitor(self.memory_threshold)
        return super().training_step(model, inputs, num_items_in_batch)

    def evaluate(self, *args, **kwargs):
        """Override evaluate to manage memory before evaluation"""
        memory_monitor(self.memory_threshold, force=True)
        return super().evaluate(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        """Override save_model to manage memory before saving"""
        memory_monitor(self.memory_threshold, force=True)
        return super().save_model(*args, **kwargs)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a bot detection model on the tokenized balanced Twibot-22 dataset')
    parser.add_argument('--use-parquet', action='store_true', help='Use Parquet format for input')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for training')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Number of steps to accumulate gradients (effective batch size = batch_size * gradient_accumulation)')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training (fp16) for memory efficiency')
    parser.add_argument('--memory-threshold', type=int, default=80, help='Memory usage percentage threshold for garbage collection')
    args = parser.parse_args()

    start_time = time.time()

    # Get the Twibot-22 directory
    twibot22_dir = os.path.dirname(script_dir)

    # Set input directory based on format
    if args.use_parquet:
        input_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_tokenized_parquet")
        print(f"Using Parquet format. Loading from {input_dir}")
    else:
        input_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_tokenized")
        print(f"Using Hugging Face format. Loading from {input_dir}")

    # Set output directory for the model
    output_dir = os.path.join(twibot22_dir, "models", "bot_detection_model")
    os.makedirs(output_dir, exist_ok=True)

    print_memory_usage()

    # Load the tokenized dataset
    print("\n[1/5] Loading tokenized dataset...")
    if args.use_parquet:
        tokenized_dataset = load_parquet_as_dataset(input_dir)
    else:
        tokenized_dataset = load_from_disk(input_dir)

    print("Dataset loaded successfully!")
    print(f"Splits: {', '.join(tokenized_dataset.keys())}")
    for split, ds in tokenized_dataset.items():
        print(f"  {split}: {len(ds)} samples")

    # Check for memory usage after loading dataset
    memory_monitor(args.memory_threshold)

    # Load the tokenizer
    print("\n[2/5] Loading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Check for GPU availability and set appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple MPS (Metal Performance Shaders) for acceleration")
    else:
        device = torch.device("cpu")
        print(f"Using CPU for computation (no GPU available)")

    # Load the model
    print("\n[3/5] Loading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "human", 1: "bot"},
        label2id={"human": 0, "bot": 1}
    )

    # Move model to the appropriate device
    model = model.to(device)
    print(f"Model loaded and moved to {device}")
    memory_monitor(args.memory_threshold)

    # Set up training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation,  # Accumulate gradients to reduce memory usage
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard, etc.
        fp16=args.fp16,  # Use mixed precision if requested
        dataloader_num_workers=1,  # Reduce number of workers to save memory
        dataloader_pin_memory=False  # Disable pin_memory to reduce memory usage
    )

    # Set up early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.01
    )

    # Create Memory-Efficient Trainer
    trainer = MemoryEfficientTrainer(
        memory_threshold=args.memory_threshold,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],  # Use validation split during training
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Train the model
    print("\n[4/5] Training the model...")
    train_result = trainer.train()
    memory_monitor(args.memory_threshold, force=True)

    # Save the model
    print("\nSaving the model...")
    trainer.save_model(os.path.join(output_dir, "best_model"))

    # Evaluate the model on the test set
    print("\n[5/5] Evaluating the model on the test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])

    # Print test results
    print("\nTest results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # Save test results
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)

    # Plot training curves
    print("\nPlotting training curves...")

    # Extract metrics from training logs
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []

    for log in trainer.state.log_history:
        if 'loss' in log and 'epoch' in log:
            train_loss.append((log['epoch'], log['loss']))
        elif 'eval_loss' in log:
            eval_loss.append((log['epoch'], log['eval_loss']))
            eval_accuracy.append((log['epoch'], log['eval_accuracy']))
            eval_f1.append((log['epoch'], log['eval_f1']))

    # Sort by epoch
    train_loss.sort(key=lambda x: x[0])
    eval_loss.sort(key=lambda x: x[0])
    eval_accuracy.sort(key=lambda x: x[0])
    eval_f1.sort(key=lambda x: x[0])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot([x[0] for x in train_loss], [x[1] for x in train_loss], 'b-', label='Training Loss')
    ax1.plot([x[0] for x in eval_loss], [x[1] for x in eval_loss], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot metrics
    ax2.plot([x[0] for x in eval_accuracy], [x[1] for x in eval_accuracy], 'g-', label='Accuracy')
    ax2.plot([x[0] for x in eval_f1], [x[1] for x in eval_f1], 'm-', label='F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))

    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")

    # Calculate and print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nDone! Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"The trained model has been saved to: {os.path.join(output_dir, 'best_model')}")
    print(f"Test results: Accuracy: {test_results['eval_accuracy']:.4f}, F1: {test_results['eval_f1']:.4f}")

    # Final memory cleanup
    memory_monitor(args.memory_threshold, force=True)

    print("\nNext Steps:")
    print("Run 'python Twibot-22/scripts/4_predict.py' to make predictions with the trained model")

if __name__ == "__main__":
    main()
