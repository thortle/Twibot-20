"""
Fix Twibot-20 Dataset

This script fixes the Twibot-20 dataset by extracting more meaningful text from the user profiles
and tweets. It creates a new dataset with better text content for training the bot detection model.
The dataset is saved in both Hugging Face format and Apache Parquet format for efficient storage and access.
"""

import os
import json
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import re
import sys

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import parquet utilities
from utilities.parquet_utils import save_dataset_to_parquet

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
    text_parts = []

    # Extract profile information
    if isinstance(user_data, dict):
        # Add username if available
        if 'username' in user_data and user_data['username']:
            username = str(user_data['username']).strip()
            if username:
                text_parts.append(f"Username: {username}")

        # Add name if available
        if 'name' in user_data and user_data['name']:
            name = str(user_data['name']).strip()
            if name:
                text_parts.append(f"Name: {name}")

        # Add description if available
        if 'description' in user_data and user_data['description']:
            description = str(user_data['description']).strip()
            if description:
                text_parts.append(f"Description: {description}")

        # Add location if available
        if 'location' in user_data and user_data['location']:
            location = str(user_data['location']).strip()
            if location:
                text_parts.append(f"Location: {location}")

        # Extract tweets if available
        if 'tweet' in user_data and isinstance(user_data['tweet'], list):
            tweet_texts = []
            for tweet in user_data['tweet']:
                if isinstance(tweet, dict) and 'text' in tweet and tweet['text']:
                    tweet_text = str(tweet['text']).strip()
                    if tweet_text:
                        tweet_texts.append(tweet_text)

            # Add up to 5 tweets to avoid making the text too long
            if tweet_texts:
                text_parts.append("Tweets:")
                for i, tweet_text in enumerate(tweet_texts[:5]):
                    text_parts.append(f"  Tweet {i+1}: {tweet_text}")

    # Join all text parts with newlines
    return "\n".join(text_parts)

def clean_text(text):
    """
    Clean the text by removing URLs, extra whitespace, etc.

    Args:
        text (str): The text to clean

    Returns:
        str: The cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

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
        features_list = []
        label_list = []

        # Process each user in the split
        for user_id in splits[split_name]:
            if user_id in nodes and user_id in labels:
                user_ids.append(user_id)

                # Extract and clean text from the user's data
                user_text = extract_text_from_user(nodes[user_id])
                cleaned_text = clean_text(user_text)
                texts.append(cleaned_text)

                # Store the full features as JSON string
                features_list.append(json.dumps(nodes[user_id]))

                # Convert label to integer (0 for human, 1 for bot)
                label_value = 1 if labels[user_id] == 'bot' else 0
                label_list.append(label_value)

        # Define features with proper types
        features = Features({
            'user_id': Value('string'),
            'text': Value('string'),
            'features': Value('string'),
            'label': ClassLabel(num_classes=2, names=['human', 'bot'])
        })

        # Create a Dataset with defined features
        datasets[split_name] = Dataset.from_dict({
            'user_id': user_ids,
            'text': texts,
            'features': features_list,
            'label': label_list
        }, features=features)

        # Print statistics
        print(f"Created {split_name} dataset with {len(user_ids)} samples")
        print(f"  Empty text count: {sum(1 for t in texts if not t)}")
        print(f"  Average text length: {sum(len(t) for t in texts) / len(texts) if texts else 0:.2f} characters")

        # Print label distribution
        bot_count = sum(1 for label in label_list if label == 1)
        human_count = sum(1 for label in label_list if label == 0)
        print(f"  Bots: {bot_count} ({bot_count/len(label_list)*100:.1f}%)")
        print(f"  Humans: {human_count} ({human_count/len(label_list)*100:.1f}%)")

    # Create a DatasetDict
    return DatasetDict(datasets)

def main():
    # Get the project root directory (one level up from scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "Twibot-20")
    output_dir = os.path.join(project_root, "data", "twibot20_fixed_dataset")

    print("[1/4] Processing data...")

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
    sys.path.append(os.path.dirname(script_dir))  # Add project root to path
    from utilities.dataset_splitter import split_dataset
    print("\nSplitting train set into train and validation sets...")
    final_dataset = split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label')

    # Save the dataset in Hugging Face format
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

    # Save the dataset in Parquet format
    parquet_output_dir = os.path.join(project_root, "data", "twibot20_fixed_parquet")
    if not os.path.exists(parquet_output_dir):
        os.makedirs(parquet_output_dir, exist_ok=True)

    print(f"\nSaving fixed dataset in Parquet format to {parquet_output_dir}...")
    save_dataset_to_parquet(final_dataset, parquet_output_dir)

    print("Done!")
    print(f"The fixed dataset has been saved to:")
    print(f"  - Hugging Face format: {output_dir}")
    print(f"  - Parquet format: {parquet_output_dir}")
    print("\nNext steps:")
    print("1. Run 'python scripts/2_tokenize_dataset.py' to tokenize the fixed dataset")
    print("2. Run 'python scripts/3_train_model.py' to train the model on the tokenized fixed dataset")

if __name__ == "__main__":
    main()
