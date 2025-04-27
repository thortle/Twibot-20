"""
Fix Twibot-20 Dataset for Llama

This script fixes the Twibot-20 dataset by extracting meaningful text from the user profiles
and tweets. It creates a new dataset with better text content for training the Llama bot detection model.
"""

import os
import json
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import re

def load_twibot20_data(data_dir):
    """
    Load the Twibot-20 dataset from the data files.

    Args:
        data_dir (str): Path to the directory containing the data files

    Returns:
        tuple: (nodes, labels, splits) - The loaded data
    """
    print(f"Loading data from {data_dir}...")

    # Load the data files
    with open(os.path.join(data_dir, "node_new.json"), 'r', encoding='utf-8') as f:
        nodes = json.load(f)

    with open(os.path.join(data_dir, "label_new.json"), 'r', encoding='utf-8') as f:
        labels = json.load(f)
        # Remove header if exists
        if "id" in labels:
            del labels["id"]

    with open(os.path.join(data_dir, "split_new.json"), 'r', encoding='utf-8') as f:
        splits = json.load(f)

    print(f"Loaded {len(nodes)} nodes, {len(labels)} labels, and {len(splits)} splits")
    return nodes, labels, splits

def extract_text_from_user(user_data):
    """
    Extract meaningful text from a user's data.

    Args:
        user_data (dict): The user data from the nodes dictionary

    Returns:
        str: Concatenated text from the user's profile and tweets
    """
    # Initialize text parts
    text_parts = []

    # Extract username
    if "username" in user_data and user_data["username"]:
        text_parts.append(f"Username: {user_data['username']}")

    # Extract name
    if "name" in user_data and user_data["name"]:
        text_parts.append(f"Name: {user_data['name']}")

    # Extract description
    if "description" in user_data and user_data["description"]:
        text_parts.append(f"Description: {user_data['description']}")

    # Extract location
    if "location" in user_data and user_data["location"]:
        text_parts.append(f"Location: {user_data['location']}")

    # Extract tweets (up to 5)
    if "tweet" in user_data and isinstance(user_data["tweet"], list):
        tweets = user_data["tweet"][:5]  # Limit to 5 tweets
        for i, tweet in enumerate(tweets):
            if isinstance(tweet, dict) and "text" in tweet and tweet["text"]:
                text_parts.append(f"Tweet {i+1}: {tweet['text']}")

    # Combine all text parts
    combined_text = " ".join(text_parts)

    # Clean the text
    combined_text = clean_text(combined_text)

    return combined_text

def clean_text(text):
    """
    Clean the text by removing URLs, extra whitespace, etc.

    Args:
        text (str): The text to clean

    Returns:
        str: The cleaned text
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def convert_to_hf_dataset(nodes, labels, splits):
    """
    Convert the Twibot-20 data to a Hugging Face DatasetDict with improved text extraction.

    Args:
        nodes (dict): The node data
        labels (dict): The label data
        splits (dict): The split data

    Returns:
        DatasetDict: The converted dataset
    """
    print("Converting to Hugging Face DatasetDict with improved text extraction...")

    # Create datasets for each split
    datasets = {}

    for split_name in ['train', 'test']:
        if split_name not in splits:
            print(f"Warning: Split '{split_name}' not found in the data")
            continue

        user_ids = []
        texts = []
        label_list = []

        # Process each user in the split
        for user_id in splits[split_name]:
            if user_id in nodes and user_id in labels:
                user_ids.append(user_id)

                # Extract and clean text from the user's data
                text = extract_text_from_user(nodes[user_id])
                texts.append(text)

                # Get label (0 = human, 1 = bot)
                label = 1 if labels[user_id] == "bot" else 0
                label_list.append(label)

        # Create dataset for this split
        print(f"  {split_name}: {len(user_ids)} users")
        datasets[split_name] = Dataset.from_dict({
            "user_id": user_ids,
            "text": texts,
            "label": label_list
        }, features=Features({
            "user_id": Value("string"),
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["human", "bot"])
        }))

    # Return the datasets
    if not datasets:
        print("Warning: No datasets were created. Check if the data files are correct.")

    return DatasetDict(datasets)

def main():
    # Get the project root directory (one level up from scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "..", "data", "Twibot-20")
    output_dir = os.path.join(project_root, "..", "data", "twibot20_fixed_dataset")

    print("[1/4] Processing data for Llama model...")

    # Load the data
    nodes, labels, splits = load_twibot20_data(data_dir)

    # Convert to Hugging Face DatasetDict with improved text extraction
    combined_dataset = convert_to_hf_dataset(nodes, labels, splits)

    # Print dataset statistics
    print("\nDataset statistics:")
    for split_name, dataset in combined_dataset.items():
        print(f"  {split_name}: {len(dataset)} samples")

    # Split the train set into train and validation sets
    # Import the dataset_splitter from utilities
    import sys
    sys.path.append(os.path.dirname(project_root))  # Add project root to path
    from utilities.dataset_splitter import split_dataset
    print("\nSplitting train set into train and validation sets...")
    final_dataset = split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label')

    # Save the dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving fixed dataset to {output_dir}...")
    final_dataset.save_to_disk(output_dir)

    # Save dataset info
    info = {
        'splits': {split: len(final_dataset[split]) for split in final_dataset},
        'label_distribution': {
            split: {
                'human': sum(1 for label in final_dataset[split]['label'] if label == 0),
                'bot': sum(1 for label in final_dataset[split]['label'] if label == 1)
            } for split in final_dataset
        }
    }

    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print("\nDataset saved successfully!")
    print(f"You can now load it using:")
    print("from datasets import load_from_disk")
    print(f"dataset = load_from_disk('{output_dir}')")

    print("\nNext steps:")
    print("1. Run 'python llama_model/scripts/2_tokenize_dataset.py' to tokenize the fixed dataset")
    print("2. Run 'python llama_model/scripts/3_train_model.py' to train the model on the tokenized fixed dataset")

if __name__ == "__main__":
    main()
