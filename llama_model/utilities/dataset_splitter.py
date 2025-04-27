"""
Dataset Splitter for Twibot-20

This module provides functionality to split a Hugging Face DatasetDict containing 'train' and 'test' splits
into a new DatasetDict with 'train', 'validation', and 'test' splits.

The split is performed on the 'train' split, creating a new 'train' and 'validation' split with a 90/10 ratio,
stratified by the 'label' column to maintain the proportion of bots and humans in both splits.
"""

import numpy as np
from datasets import DatasetDict, ClassLabel, Features, Value

def split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label', random_state=42):
    """
    Split the existing 'train' split of the combined_dataset into a new 'train' and 'validation' split.

    Args:
        combined_dataset (DatasetDict): A Hugging Face DatasetDict containing at least 'train' and 'test' splits.
        test_size (float, optional): The proportion of the 'train' split to use for validation. Defaults to 0.1.
        stratify_by_column (str, optional): The column to use for stratification. Defaults to 'label'.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        DatasetDict: A new DatasetDict containing 'train', 'validation', and 'test' splits.
    """
    # Verify that the combined_dataset is a DatasetDict and contains the required splits
    if not isinstance(combined_dataset, DatasetDict):
        raise TypeError("combined_dataset must be a DatasetDict")

    if 'train' not in combined_dataset or 'test' not in combined_dataset:
        raise ValueError("combined_dataset must contain 'train' and 'test' splits")

    # Verify that the stratify_by_column exists in the dataset
    if stratify_by_column not in combined_dataset['train'].column_names:
        raise ValueError(f"Column '{stratify_by_column}' not found in the 'train' split")

    # Check if the stratify_by_column is a ClassLabel
    is_class_label = isinstance(combined_dataset['train'].features.get(stratify_by_column), ClassLabel)

    if is_class_label:
        # If it's already a ClassLabel, we can use stratify_by_column directly
        train_test_split = combined_dataset['train'].train_test_split(
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            seed=random_state
        )
    else:
        # If it's not a ClassLabel, we need to use a different approach
        # Get the labels for stratification
        labels = combined_dataset['train'][stratify_by_column]

        # Use sklearn's train_test_split for stratification
        from sklearn.model_selection import train_test_split as sklearn_split

        # Get indices for train and validation
        indices = np.arange(len(combined_dataset['train']))
        train_indices, val_indices = sklearn_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Split the dataset using the indices
        train_test_split = {
            'train': combined_dataset['train'].select(train_indices),
            'test': combined_dataset['train'].select(val_indices)
        }

    # Create the final DatasetDict
    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test'],  # The 'test' part of the split becomes validation
        'test': combined_dataset['test']
    })

    # Print information about the splits
    print(f"Original dataset: {combined_dataset}")
    print(f"Split dataset: {final_dataset}")
    print(f"Train split size: {len(final_dataset['train'])}")
    print(f"Validation split size: {len(final_dataset['validation'])}")
    print(f"Test split size: {len(final_dataset['test'])}")

    return final_dataset

if __name__ == "__main__":
    # Example usage
    # This is just a placeholder and won't run as-is
    # You would need to load your dataset first

    # Example:
    # from datasets import load_dataset
    # combined_dataset = load_dataset("your_dataset_name")
    # split_dataset = split_dataset(combined_dataset)

    print("To use this module, import it and call the split_dataset function with your DatasetDict.")
    print("Example:")
    print("from dataset_splitter import split_dataset")
    print("from datasets import load_dataset")
    print("combined_dataset = load_dataset('your_dataset_name')")
    print("split_dataset = split_dataset(combined_dataset)")
