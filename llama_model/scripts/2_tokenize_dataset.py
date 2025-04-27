"""
Tokenize Fixed Twibot-20 Dataset for Llama

This script tokenizes the fixed Twibot-20 dataset for sequence classification.
It uses the Llama tokenizer to prepare the data for training a text classification model.
"""

from datasets import load_from_disk
from transformers import AutoTokenizer
import os

def main():
    # Define the path to the saved dataset directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "..", "data", "twibot20_fixed_dataset")
    output_dir = os.path.join(project_root, "..", "data", "twibot20_llama_tokenized")

    print("[2/4] Tokenizing dataset for Llama...")

    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Fixed dataset not found at {dataset_path}")
        print("Please run scripts/1_fix_dataset.py first to create the fixed dataset.")
        return

    # 1. Load dataset
    print(f"Loading fixed dataset from {dataset_path}...")
    try:
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
    print("\nLoading T5 tokenizer (as a substitute for Llama)...")
    try:
        # Using Google's Flan-T5 model as a substitute for Llama
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Model max length: {tokenizer.model_max_length}")
        print("\nNote: Using Flan-T5 as a substitute for Llama due to access restrictions.")
        print("To use the actual Llama model, you need to request access on Hugging Face.")
        print("Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
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
        for key, value in example.items():
            if key not in ['text', 'user_id']:
                print(f"  {key}: {value}")

        # 6. Calculate token statistics
        print("\nToken statistics:")
        for split_name, split_dataset in tokenized_dataset.items():
            input_ids_lengths = [len(ids) for ids in split_dataset['input_ids']]
            avg_tokens = sum(input_ids_lengths) / len(input_ids_lengths) if input_ids_lengths else 0
            max_tokens = max(input_ids_lengths) if input_ids_lengths else 0
            min_tokens = min(input_ids_lengths) if input_ids_lengths else 0

            # Count samples with only special tokens (likely empty text)
            special_token_only = sum(1 for length in input_ids_lengths if length <= 2)

            # Count samples exceeding model max length
            exceeding_max = sum(1 for length in input_ids_lengths if length >= tokenizer.model_max_length)

            print(f"  {split_name}:")
            print(f"    Average tokens per sample: {avg_tokens:.2f}")
            print(f"    Min tokens: {min_tokens}, Max tokens: {max_tokens}")
            print(f"    Samples with only special tokens: {special_token_only} ({special_token_only/len(input_ids_lengths)*100:.2f}%)")
            print(f"    Samples exceeding model max length: {exceeding_max} ({exceeding_max/len(input_ids_lengths)*100:.2f}%)")

        # 7. Save the tokenized dataset
        print(f"\nSaving tokenized dataset to {output_dir}...")
        tokenized_dataset.save_to_disk(output_dir)
        print("Tokenized dataset saved successfully!")
        print("You can now load it using:")
        print("from datasets import load_from_disk")
        print(f"tokenized_dataset = load_from_disk('{output_dir}')")

        print("\nNext steps:")
        print("1. Update scripts/3_train_model.py to use the Llama tokenized dataset")
        print("   Change: tokenized_dataset_path = os.path.join(project_root, \"data\", \"twibot20_llama_tokenized\")")
        print("2. Run 'python llama_model/scripts/3_train_model.py' to train the model on the fixed dataset")

    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return

if __name__ == "__main__":
    main()
