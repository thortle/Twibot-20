"""
Tokenize Balanced Twibot-22 Dataset

This script tokenizes the balanced Twibot-22 dataset using the DistilBERT tokenizer.
It loads the dataset, applies tokenization, and saves the tokenized dataset in both
Hugging Face format and Apache Parquet format.
"""

import os
import json
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
import sys

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Two levels up to reach the main project root
sys.path.append(project_root)

# Import parquet utilities
from utilities.parquet_utils import save_dataset_to_parquet, load_parquet_as_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tokenize the balanced Twibot-22 dataset')
    parser.add_argument('--use-parquet', action='store_true', help='Use Parquet format for input and output')
    args = parser.parse_args()
    
    # Get the Twibot-22 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    twibot22_dir = os.path.dirname(script_dir)
    
    # Set input and output directories based on format
    if args.use_parquet:
        input_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_parquet")
        output_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_tokenized_parquet")
        print(f"Using Parquet format. Loading from {input_dir}")
    else:
        input_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_dataset")
        output_dir = os.path.join(twibot22_dir, "data", "twibot22_balanced_tokenized")
        print(f"Using Hugging Face format. Loading from {input_dir}")
    
    # Load the dataset
    print("Loading dataset...")
    if args.use_parquet:
        dataset = load_parquet_as_dataset(input_dir)
    else:
        dataset = load_from_disk(input_dir)
    
    print("Dataset loaded successfully!")
    print(f"Splits: {', '.join(dataset.keys())}")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} samples")
    
    # Load the tokenizer
    print("\nLoading DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Define preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Padding will be handled later by DataCollator
            max_length=tokenizer.model_max_length
        )
    
    # Apply tokenization
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing dataset"
    )
    
    # Analyze tokenization results
    print("\nTokenization statistics:")
    for split, ds in tokenized_dataset.items():
        # Calculate average tokens per sample
        avg_tokens = sum(len(ids) for ids in ds['input_ids']) / len(ds)
        
        # Calculate maximum tokens in a sample
        max_tokens = max(len(ids) for ids in ds['input_ids'])
        
        # Count samples that exceed max length
        exceeded_max_length = sum(1 for ids in ds['input_ids'] if len(ids) == tokenizer.model_max_length)
        
        # Count samples with only special tokens (essentially empty)
        only_special_tokens = sum(1 for ids in ds['input_ids'] if len(ids) <= 2)
        
        print(f"  {split}:")
        print(f"    Average tokens per sample: {avg_tokens:.1f}")
        print(f"    Maximum tokens in a sample: {max_tokens}")
        print(f"    Samples exceeding max length: {exceeded_max_length} ({exceeded_max_length/len(ds)*100:.1f}%)")
        print(f"    Samples with only special tokens: {only_special_tokens} ({only_special_tokens/len(ds)*100:.1f}%)")
    
    # Show examples of tokenized output
    print("\nExample of tokenized output:")
    for split in tokenized_dataset:
        example = tokenized_dataset[split][0]
        print(f"  {split} example:")
        print(f"    Text: {example['text'][:100]}...")
        print(f"    Input IDs: {example['input_ids'][:10]}... (length: {len(example['input_ids'])})")
        print(f"    Attention Mask: {example['attention_mask'][:10]}... (length: {len(example['attention_mask'])})")
        break  # Just show one example
    
    # Save the tokenized dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving tokenized dataset to {output_dir}...")
    if args.use_parquet:
        save_dataset_to_parquet(tokenized_dataset, output_dir)
    else:
        tokenized_dataset.save_to_disk(output_dir)
    
    # Save tokenization info
    info = {
        'tokenizer': "distilbert-base-uncased",
        'max_length': tokenizer.model_max_length,
        'splits': {split: len(ds) for split, ds in tokenized_dataset.items()},
        'statistics': {
            split: {
                'avg_tokens': sum(len(ids) for ids in ds['input_ids']) / len(ds),
                'max_tokens': max(len(ids) for ids in ds['input_ids']),
                'exceeded_max_length': sum(1 for ids in ds['input_ids'] if len(ids) == tokenizer.model_max_length),
                'only_special_tokens': sum(1 for ids in ds['input_ids'] if len(ids) <= 2)
            } for split, ds in tokenized_dataset.items()
        }
    }
    
    with open(os.path.join(output_dir, 'tokenization_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Done!")
    print(f"The tokenized dataset has been saved to: {output_dir}")
    print("\nNext steps:")
    if args.use_parquet:
        print("Run 'python Twibot-22/scripts/3_train_model.py --use-parquet' to train the model on the tokenized dataset")
    else:
        print("Run 'python Twibot-22/scripts/3_train_model.py' to train the model on the tokenized dataset")

if __name__ == "__main__":
    main()
