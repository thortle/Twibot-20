#!/usr/bin/env python3
"""
Prepare Dataset for Tokenization

This script converts the extracted tweets into the format expected by the tokenization script.
It creates both Hugging Face and Parquet formats for the dataset.
"""

import os
import pandas as pd
import random
import argparse
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

# Add project root to path to import utilities
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
twibot22_dir = os.path.dirname(script_dir)
sys.path.append(twibot22_dir)

# Import parquet utilities
from utilities.parquet_utils import save_dataset_to_parquet

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for tokenization')
    parser.add_argument('--input-dir', type=str, default='./extracted_1000_tweets',
                        help='Directory containing the extracted tweets')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the prepared dataset (defaults to Twibot-22/data/twibot22_balanced_dataset)')
    parser.add_argument('--parquet-dir', type=str, default=None,
                        help='Directory to save the Parquet dataset (defaults to Twibot-22/data/twibot22_balanced_parquet)')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of data to use for testing (default: 0.1)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Set default output directories if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_dataset")

    if args.parquet_dir is None:
        args.parquet_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_parquet")

    # Set random seed for reproducibility
    random.seed(args.seed)

    print(f"Preparing dataset from {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parquet directory: {args.parquet_dir}")
    print(f"Test split: {args.test_split}")
    print(f"Validation split: {args.validation_split}")

    # Load the tweets and labels
    tweets_file = os.path.join(args.input_dir, 'tweets.txt')
    labels_file = os.path.join(args.input_dir, 'labels.txt')

    with open(tweets_file, 'r', encoding='utf-8') as f:
        tweets = [line.strip() for line in f]

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f]

    # Create a DataFrame
    df = pd.DataFrame({
        'text': tweets,
        'label': labels,
        'user_id': [f'user_{i}' for i in range(len(tweets))],  # Generate dummy user IDs
        'tweet_count': [1 for _ in range(len(tweets))]  # Each row is one tweet
    })

    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Bot samples: {sum(df['label'] == 1)}")
    print(f"Human samples: {sum(df['label'] == 0)}")

    # Split into train, validation, and test sets
    indices = list(range(len(df)))
    random.shuffle(indices)

    # Calculate split sizes
    test_size = int(len(df) * args.test_split)
    validation_size = int(len(df) * args.validation_split)

    # Split indices
    test_indices = indices[:test_size]
    validation_indices = indices[test_size:test_size + validation_size]
    train_indices = indices[test_size + validation_size:]

    # Create dataframes for each split
    train_df = df.iloc[train_indices].reset_index(drop=True)
    validation_df = df.iloc[validation_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(validation_df)} samples")
    print(f"Test: {len(test_df)} samples")

    # Check class distribution in each split
    print("\nClass distribution:")
    print(f"Train: {sum(train_df['label'] == 1)} bot, {sum(train_df['label'] == 0)} human")
    print(f"Validation: {sum(validation_df['label'] == 1)} bot, {sum(validation_df['label'] == 0)} human")
    print(f"Test: {sum(test_df['label'] == 1)} bot, {sum(test_df['label'] == 0)} human")

    # Define features
    features = Features({
        'user_id': Value('string'),
        'text': Value('string'),
        'tweet_count': Value('int64'),
        'label': ClassLabel(num_classes=2, names=['human', 'bot'])
    })

    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df, features=features)
    validation_dataset = Dataset.from_pandas(validation_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    # Save in Hugging Face format
    print(f"\nSaving dataset in Hugging Face format to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    # Save in Parquet format
    print(f"\nSaving dataset in Parquet format to {args.parquet_dir}...")
    os.makedirs(args.parquet_dir, exist_ok=True)
    save_dataset_to_parquet(dataset_dict, args.parquet_dir)

    print("\nDataset preparation complete!")
    print("\nNext steps:")
    print("1. Run 'python Twibot-22/scripts/2_tokenize_balanced_dataset.py' to tokenize using Hugging Face format")
    print("2. Or run 'python Twibot-22/scripts/2_tokenize_balanced_dataset.py --use-parquet' to tokenize using Parquet format")

if __name__ == "__main__":
    main()
