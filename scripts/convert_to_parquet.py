"""
Convert Dataset to Parquet Format

This script converts the Twibot-20 dataset from JSON to Apache Parquet format.
Parquet is a columnar storage format that offers better compression and faster query performance.
"""

import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk, Dataset, DatasetDict

def convert_json_to_parquet(input_path, output_path):
    """
    Convert JSON data files to Parquet format.
    
    Args:
        input_path (str): Path to the directory containing JSON data files
        output_path (str): Path to save the Parquet files
    """
    print(f"Converting JSON files from {input_path} to Parquet format...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load the data files
    try:
        with open(os.path.join(input_path, "node_new.json"), 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        
        with open(os.path.join(input_path, "label_new.json"), 'r', encoding='utf-8') as f:
            labels = json.load(f)
            # Remove header if exists
            if "id" in labels:
                del labels["id"]
        
        with open(os.path.join(input_path, "split_new.json"), 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        print(f"Loaded {len(nodes)} nodes, {len(labels)} labels, and {len(splits)} splits")
        
        # Convert nodes to DataFrame
        print("Converting nodes to DataFrame...")
        nodes_data = []
        for user_id, user_data in nodes.items():
            # Extract basic user data
            user_row = {
                'user_id': user_id,
                'username': user_data.get('username', ''),
                'name': user_data.get('name', ''),
                'description': user_data.get('description', ''),
                'location': user_data.get('location', ''),
                'created_at': user_data.get('created_at', ''),
                'followers_count': user_data.get('followers_count', 0),
                'friends_count': user_data.get('friends_count', 0),
                'statuses_count': user_data.get('statuses_count', 0),
                'favourites_count': user_data.get('favourites_count', 0),
                'listed_count': user_data.get('listed_count', 0),
                'verified': user_data.get('verified', False),
                'protected': user_data.get('protected', False),
                'default_profile': user_data.get('default_profile', False),
                'default_profile_image': user_data.get('default_profile_image', False),
                'has_tweets': 'tweet' in user_data and isinstance(user_data['tweet'], list) and len(user_data['tweet']) > 0
            }
            nodes_data.append(user_row)
        
        # Create DataFrame
        nodes_df = pd.DataFrame(nodes_data)
        
        # Convert labels to DataFrame
        print("Converting labels to DataFrame...")
        labels_data = []
        for user_id, label in labels.items():
            labels_data.append({
                'user_id': user_id,
                'label': 1 if label == 'bot' else 0,
                'label_str': label
            })
        
        labels_df = pd.DataFrame(labels_data)
        
        # Convert splits to DataFrame
        print("Converting splits to DataFrame...")
        splits_data = []
        for split_name, user_ids in splits.items():
            for user_id in user_ids:
                splits_data.append({
                    'user_id': user_id,
                    'split': split_name
                })
        
        splits_df = pd.DataFrame(splits_data)
        
        # Convert tweets to DataFrame
        print("Converting tweets to DataFrame...")
        tweets_data = []
        for user_id, user_data in nodes.items():
            if 'tweet' in user_data and isinstance(user_data['tweet'], list):
                for i, tweet in enumerate(user_data['tweet']):
                    if isinstance(tweet, dict):
                        tweet_row = {
                            'user_id': user_id,
                            'tweet_id': tweet.get('id', f"{user_id}_tweet_{i}"),
                            'text': tweet.get('text', ''),
                            'created_at': tweet.get('created_at', ''),
                            'retweet_count': tweet.get('retweet_count', 0),
                            'favorite_count': tweet.get('favorite_count', 0),
                            'tweet_index': i
                        }
                        tweets_data.append(tweet_row)
        
        tweets_df = pd.DataFrame(tweets_data)
        
        # Save DataFrames as Parquet files
        print("Saving DataFrames as Parquet files...")
        
        # Convert to PyArrow Tables for better control over Parquet writing
        nodes_table = pa.Table.from_pandas(nodes_df)
        labels_table = pa.Table.from_pandas(labels_df)
        splits_table = pa.Table.from_pandas(splits_df)
        tweets_table = pa.Table.from_pandas(tweets_df)
        
        # Write Parquet files with compression
        pq.write_table(nodes_table, os.path.join(output_path, "nodes.parquet"), compression='snappy')
        pq.write_table(labels_table, os.path.join(output_path, "labels.parquet"), compression='snappy')
        pq.write_table(splits_table, os.path.join(output_path, "splits.parquet"), compression='snappy')
        pq.write_table(tweets_table, os.path.join(output_path, "tweets.parquet"), compression='snappy')
        
        print("Parquet files saved successfully!")
        
        # Print file sizes
        print("\nFile size comparison:")
        json_size = os.path.getsize(os.path.join(input_path, "node_new.json")) / (1024 * 1024)
        parquet_size = os.path.getsize(os.path.join(output_path, "nodes.parquet")) / (1024 * 1024)
        print(f"  nodes.json: {json_size:.2f} MB")
        print(f"  nodes.parquet: {parquet_size:.2f} MB")
        print(f"  Compression ratio: {json_size / parquet_size:.2f}x")
        
        return True
    
    except Exception as e:
        print(f"Error converting JSON to Parquet: {e}")
        return False

def convert_hf_dataset_to_parquet(input_path, output_path):
    """
    Convert a Hugging Face dataset to Parquet format.
    
    Args:
        input_path (str): Path to the Hugging Face dataset
        output_path (str): Path to save the Parquet files
    """
    print(f"Converting Hugging Face dataset from {input_path} to Parquet format...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load the dataset
        dataset = load_from_disk(input_path)
        print(f"Loaded dataset with splits: {list(dataset.keys())}")
        
        # Convert each split to Parquet
        for split_name, split_dataset in dataset.items():
            # Convert to pandas DataFrame
            df = split_dataset.to_pandas()
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # Save as Parquet
            output_file = os.path.join(output_path, f"{split_name}.parquet")
            pq.write_table(table, output_file, compression='snappy')
            
            # Print statistics
            print(f"  Saved {split_name} split with {len(df)} rows to {output_file}")
            print(f"  File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        
        print("Conversion completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error converting dataset to Parquet: {e}")
        return False

def load_parquet_as_hf_dataset(parquet_path):
    """
    Load Parquet files as a Hugging Face dataset.
    
    Args:
        parquet_path (str): Path to the directory containing Parquet files
    
    Returns:
        DatasetDict: A Hugging Face DatasetDict
    """
    print(f"Loading Parquet files from {parquet_path} as Hugging Face dataset...")
    
    try:
        # Get all Parquet files in the directory
        parquet_files = [f for f in os.listdir(parquet_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            print(f"No Parquet files found in {parquet_path}")
            return None
        
        # Create a DatasetDict
        datasets_dict = {}
        
        for parquet_file in parquet_files:
            # Extract split name from filename (remove .parquet extension)
            split_name = os.path.splitext(parquet_file)[0]
            
            # Skip if not a split file
            if split_name in ['nodes', 'labels', 'splits', 'tweets']:
                continue
            
            # Load Parquet file
            file_path = os.path.join(parquet_path, parquet_file)
            df = pd.read_parquet(file_path)
            
            # Convert to Hugging Face Dataset
            datasets_dict[split_name] = Dataset.from_pandas(df)
            
            print(f"  Loaded {split_name} split with {len(df)} rows")
        
        if not datasets_dict:
            print("No valid split files found")
            return None
        
        # Create DatasetDict
        dataset_dict = DatasetDict(datasets_dict)
        print(f"Created DatasetDict with splits: {list(dataset_dict.keys())}")
        
        return dataset_dict
    
    except Exception as e:
        print(f"Error loading Parquet files: {e}")
        return None

def main():
    # Get the project root directory (one level up from scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths
    raw_data_dir = os.path.join(project_root, "data", "Twibot-20")
    raw_parquet_dir = os.path.join(project_root, "data", "twibot20_parquet")
    
    fixed_dataset_dir = os.path.join(project_root, "data", "twibot20_fixed_dataset")
    fixed_parquet_dir = os.path.join(project_root, "data", "twibot20_fixed_parquet")
    
    tokenized_dataset_dir = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
    tokenized_parquet_dir = os.path.join(project_root, "data", "twibot20_tokenized_parquet")
    
    # 1. Convert raw JSON data to Parquet
    print("\n=== Converting Raw JSON Data to Parquet ===")
    if os.path.exists(raw_data_dir):
        convert_json_to_parquet(raw_data_dir, raw_parquet_dir)
    else:
        print(f"Raw data directory not found: {raw_data_dir}")
    
    # 2. Convert fixed dataset to Parquet
    print("\n=== Converting Fixed Dataset to Parquet ===")
    if os.path.exists(fixed_dataset_dir):
        convert_hf_dataset_to_parquet(fixed_dataset_dir, fixed_parquet_dir)
    else:
        print(f"Fixed dataset directory not found: {fixed_dataset_dir}")
    
    # 3. Convert tokenized dataset to Parquet
    print("\n=== Converting Tokenized Dataset to Parquet ===")
    if os.path.exists(tokenized_dataset_dir):
        convert_hf_dataset_to_parquet(tokenized_dataset_dir, tokenized_parquet_dir)
    else:
        print(f"Tokenized dataset directory not found: {tokenized_dataset_dir}")
    
    # 4. Demonstrate loading Parquet files back as a Hugging Face dataset
    print("\n=== Demonstrating Loading Parquet Files as Hugging Face Dataset ===")
    if os.path.exists(fixed_parquet_dir):
        dataset = load_parquet_as_hf_dataset(fixed_parquet_dir)
        if dataset:
            print("\nSuccessfully loaded Parquet files as Hugging Face dataset!")
            print("Dataset splits:", list(dataset.keys()))
            print("Example features:", list(dataset[list(dataset.keys())[0]].features.keys()))
    
    print("\nConversion process completed!")
    print("\nNext steps:")
    print("1. Use the Parquet files for more efficient data loading")
    print("2. Update your data loading scripts to use the Parquet files")

if __name__ == "__main__":
    main()
