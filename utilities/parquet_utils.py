"""
Parquet Utilities

This module provides utility functions for working with Parquet files in the Twitter bot detection project.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence

def save_dataset_to_parquet(dataset, output_dir, compression='snappy'):
    """
    Save a Hugging Face dataset to Parquet format.

    Args:
        dataset (Dataset or DatasetDict): The dataset to save
        output_dir (str): Directory to save the Parquet files
        compression (str): Compression codec to use (default: 'snappy')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Handle both Dataset and DatasetDict
        if isinstance(dataset, Dataset):
            # Convert to pandas DataFrame
            df = dataset.to_pandas()

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df)

            # Save as Parquet
            output_file = os.path.join(output_dir, "dataset.parquet")
            pq.write_table(table, output_file, compression=compression)

            print(f"Saved dataset with {len(df)} rows to {output_file}")
            print(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")

        elif isinstance(dataset, DatasetDict):
            # Process each split
            for split_name, split_dataset in dataset.items():
                # Convert to pandas DataFrame
                df = split_dataset.to_pandas()

                # Convert to PyArrow Table
                table = pa.Table.from_pandas(df)

                # Save as Parquet
                output_file = os.path.join(output_dir, f"{split_name}.parquet")
                pq.write_table(table, output_file, compression=compression)

                print(f"Saved {split_name} split with {len(df)} rows to {output_file}")
                print(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        else:
            print(f"Unsupported dataset type: {type(dataset)}")
            return False

        return True

    except Exception as e:
        print(f"Error saving dataset to Parquet: {e}")
        return False

def load_parquet_as_dataset(parquet_path, split_name=None):
    """
    Load a Parquet file as a Hugging Face Dataset.

    Args:
        parquet_path (str): Path to the Parquet file or directory
        split_name (str, optional): Name of the split if loading a specific file

    Returns:
        Dataset or DatasetDict: The loaded dataset
    """
    try:
        # If parquet_path is a directory and split_name is not provided, load all splits
        if os.path.isdir(parquet_path) and split_name is None:
            # Get all Parquet files in the directory
            parquet_files = [f for f in os.listdir(parquet_path) if f.endswith('.parquet')]

            if not parquet_files:
                print(f"No Parquet files found in {parquet_path}")
                return None

            # Create a DatasetDict
            datasets_dict = {}

            for parquet_file in parquet_files:
                # Extract split name from filename (remove .parquet extension)
                file_split_name = os.path.splitext(parquet_file)[0]

                # Load Parquet file
                file_path = os.path.join(parquet_path, parquet_file)
                df = pd.read_parquet(file_path)

                # Define features with proper types
                features = None

                # Create features dictionary
                features = {}

                # If 'label' column exists, create a ClassLabel feature
                if 'label' in df.columns and df['label'].dtype == 'int64':
                    features['label'] = ClassLabel(num_classes=2, names=['human', 'bot'])

                # Add other columns dynamically
                for col in df.columns:
                    if col != 'label':
                        # Handle special columns
                        if col == 'input_ids' or col == 'attention_mask':
                            # These are sequences of integers
                            features[col] = Sequence(feature=Value('int32'))
                        else:
                            # Regular columns
                            features[col] = Value('string' if df[col].dtype == 'object' else str(df[col].dtype))

                # Create Features object
                if features:
                    features = Features(features)

                # Convert to Hugging Face Dataset
                datasets_dict[file_split_name] = Dataset.from_pandas(df, features=features, preserve_index=False)

                print(f"Loaded {file_split_name} split with {len(df)} rows")

            # Create DatasetDict
            return DatasetDict(datasets_dict)

        # If split_name is provided or parquet_path is a file, load a specific file
        else:
            # Determine the file path
            if os.path.isdir(parquet_path) and split_name:
                file_path = os.path.join(parquet_path, f"{split_name}.parquet")
            else:
                file_path = parquet_path

            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Parquet file not found: {file_path}")
                return None

            # Load Parquet file
            df = pd.read_parquet(file_path)

            # Define features with proper types
            features = None

            # Create features dictionary
            features = {}

            # If 'label' column exists, create a ClassLabel feature
            if 'label' in df.columns and df['label'].dtype == 'int64':
                features['label'] = ClassLabel(num_classes=2, names=['human', 'bot'])

            # Add other columns dynamically
            for col in df.columns:
                if col != 'label':
                    # Handle special columns
                    if col == 'input_ids' or col == 'attention_mask':
                        # These are sequences of integers
                        features[col] = Sequence(feature=Value('int32'))
                    else:
                        # Regular columns
                        features[col] = Value('string' if df[col].dtype == 'object' else str(df[col].dtype))

            # Create Features object
            if features:
                features = Features(features)

            # Convert to Hugging Face Dataset
            dataset = Dataset.from_pandas(df, features=features, preserve_index=False)

            print(f"Loaded dataset with {len(df)} rows from {file_path}")

            return dataset

    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return None

def convert_dataset_to_parquet(input_path, output_path, compression='snappy'):
    """
    Convert a Hugging Face dataset to Parquet format.

    Args:
        input_path (str): Path to the Hugging Face dataset
        output_path (str): Path to save the Parquet files
        compression (str): Compression codec to use (default: 'snappy')

    Returns:
        bool: True if successful, False otherwise
    """
    from datasets import load_from_disk

    try:
        # Load the dataset
        dataset = load_from_disk(input_path)
        print(f"Loaded dataset with splits: {list(dataset.keys())}")

        # Save to Parquet
        return save_dataset_to_parquet(dataset, output_path, compression)

    except Exception as e:
        print(f"Error converting dataset to Parquet: {e}")
        return False

def get_parquet_schema(parquet_path):
    """
    Get the schema of a Parquet file.

    Args:
        parquet_path (str): Path to the Parquet file

    Returns:
        pa.Schema: The schema of the Parquet file
    """
    try:
        # Read the schema from the Parquet file
        schema = pq.read_schema(parquet_path)
        return schema

    except Exception as e:
        print(f"Error reading Parquet schema: {e}")
        return None

def get_parquet_metadata(parquet_path):
    """
    Get metadata about a Parquet file.

    Args:
        parquet_path (str): Path to the Parquet file

    Returns:
        dict: Metadata about the Parquet file
    """
    try:
        # Open the Parquet file
        parquet_file = pq.ParquetFile(parquet_path)

        # Get metadata
        metadata = {
            'num_rows': parquet_file.metadata.num_rows,
            'num_row_groups': parquet_file.metadata.num_row_groups,
            'schema': str(parquet_file.schema),
            'file_size_mb': os.path.getsize(parquet_path) / (1024 * 1024),
            'columns': [parquet_file.schema.names[i] for i in range(parquet_file.schema.names)]
        }

        return metadata

    except Exception as e:
        print(f"Error getting Parquet metadata: {e}")
        return None

def read_parquet_sample(parquet_path, num_rows=5):
    """
    Read a sample of rows from a Parquet file.

    Args:
        parquet_path (str): Path to the Parquet file
        num_rows (int): Number of rows to read

    Returns:
        pd.DataFrame: A sample of rows from the Parquet file
    """
    try:
        # Read a sample of rows
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        return df.head(num_rows)

    except Exception as e:
        print(f"Error reading Parquet sample: {e}")
        return None
