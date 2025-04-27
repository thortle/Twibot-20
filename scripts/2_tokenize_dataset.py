"""
Tokenize Fixed Twibot-20 Dataset

This script tokenizes the fixed Twibot-20 dataset for sequence classification.
It uses the DistilBERT tokenizer to prepare the data for training a text classification model.
The script supports loading data from both Hugging Face format and Apache Parquet format.
"""

from datasets import load_from_disk
from transformers import AutoTokenizer
import os
import sys
import argparse

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import parquet utilities
from utilities.parquet_utils import load_parquet_as_dataset, save_dataset_to_parquet

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tokenize the fixed Twibot-20 dataset')
    parser.add_argument('--use-parquet', action='store_true', help='Use Parquet format instead of Hugging Face format')
    args = parser.parse_args()

    # Define the path to the saved dataset directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Set paths based on format
    if args.use_parquet:
        dataset_path = os.path.join(project_root, "data", "twibot20_fixed_parquet")
        output_dir = os.path.join(project_root, "data", "twibot20_tokenized_parquet")
        print("[2/4] Tokenizing dataset (using Parquet format)...")
    else:
        dataset_path = os.path.join(project_root, "data", "twibot20_fixed_dataset")
        output_dir = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
        print("[2/4] Tokenizing dataset (using Hugging Face format)...")

    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Fixed dataset not found at {dataset_path}")
        print("Please run scripts/1_fix_dataset.py first to create the fixed dataset.")
        return

    # 1. Load dataset
    print(f"Loading fixed dataset from {dataset_path}...")
    try:
        if args.use_parquet:
            dataset = load_parquet_as_dataset(dataset_path)
        else:
            dataset = load_from_disk(dataset_path)

        print("Fixed dataset loaded:")
        print(dataset)

        # Print some statistics
        print("\nDataset statistics:")
        for split_name, split_dataset in dataset.items():
            print(f"  {split_name}: {len(split_dataset)} samples")

            # Calculate text statistics
            if 'text' in split_dataset.column_names:
                text_lengths = [len(text) for text in split_dataset['text']]
                avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
                empty_count = sum(1 for length in text_lengths if length == 0)
                print(f"    Average text length: {avg_length:.2f} characters")
                print(f"    Empty text count: {empty_count} ({empty_count/len(text_lengths)*100:.2f}% of samples)")

            # Count labels
            if 'label' in split_dataset.features:
                label_counts = {}
                for label in split_dataset['label']:
                    label_name = split_dataset.features['label'].int2str(label)
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1

                for label_name, count in label_counts.items():
                    print(f"    {label_name}: {count} ({count/len(split_dataset)*100:.1f}%)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Model max length: {tokenizer.model_max_length}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Define preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Padding will be handled later by DataCollator
            max_length=tokenizer.model_max_length
        )

    # 4. Apply tokenization using .map()
    print("\nTokenizing dataset...")
    try:
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Tokenizing dataset"
        )

        # 5. Print tokenized dataset info
        print("\nTokenized dataset:")
        print(tokenized_dataset)

        # Optionally, inspect the features of one split
        print("\nFeatures after tokenization (train split):")
        print(tokenized_dataset['train'].features)

        # Show an example of tokenized output
        print("\nExample of tokenized output (first sample from train split):")
        example = tokenized_dataset['train'][0]
        print("  Original text:", example['text'][:100] + "..." if len(example['text']) > 100 else example['text'])
        print("  Label:", tokenized_dataset['train'].features['label'].int2str(example['label']))
        print("  Input IDs (first 10):", example['input_ids'][:10])
        print("  Attention Mask (first 10):", example['attention_mask'][:10])

        # Count tokens
        token_lengths = [len(x) for x in tokenized_dataset['train']['input_ids']]
        avg_tokens = sum(token_lengths) / len(token_lengths)
        max_tokens = max(token_lengths)
        print(f"\nToken statistics (train split):")
        print(f"  Average tokens per sample: {avg_tokens:.1f}")
        print(f"  Maximum tokens in a sample: {max_tokens}")
        print(f"  Samples exceeding model max length ({tokenizer.model_max_length}): {sum(1 for x in token_lengths if x > tokenizer.model_max_length)}")
        print(f"  Samples with only special tokens (≤2): {sum(1 for x in token_lengths if x <= 2)}")

        # Save tokenized dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving tokenized dataset to {output_dir}...")

        if args.use_parquet:
            # Save in Parquet format
            save_dataset_to_parquet(tokenized_dataset, output_dir)
            print("Tokenized dataset saved in Parquet format!")
            print("You can now load it using:")
            print("from utilities.parquet_utils import load_parquet_as_dataset")
            print(f"tokenized_dataset = load_parquet_as_dataset('{output_dir}')")
        else:
            # Save in Hugging Face format
            tokenized_dataset.save_to_disk(output_dir)
            print("Tokenized dataset saved in Hugging Face format!")
            print("You can now load it using:")
            print("from datasets import load_from_disk")
            print(f"tokenized_dataset = load_from_disk('{output_dir}')")

        print("\nNext steps:")
        print("1. Update scripts/3_train_model.py to use the tokenized dataset")
        if args.use_parquet:
            print("   Change: tokenized_dataset_path = os.path.join(project_root, \"data\", \"twibot20_tokenized_parquet\")")
            print("   Add: use_parquet = True")
        else:
            print("   Change: tokenized_dataset_path = os.path.join(project_root, \"data\", \"twibot20_fixed_tokenized\")")
        print("2. Run 'python scripts/3_train_model.py' to train the model on the dataset")

    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return

if __name__ == "__main__":
    main()
