"""
Parquet Utilities for Efficient Dataset Handling

This module provides utilities for efficiently working with datasets using Apache Parquet format,
which is more memory-efficient than standard formats, especially for large datasets.
"""

import os
import gc
import psutil
import pandas as pd
import time
from datasets import Dataset, DatasetDict
from typing import Dict, Union, Optional


def print_memory_usage():
    """Print current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.percent}% used, "
          f"{system_memory.available / (1024 * 1024):.2f} MB available")
    
    return memory_mb, system_memory.percent


def wait_for_memory(target_percent=75, max_wait=10, check_interval=1):
    """
    Wait for memory usage to drop below a target percentage.
    
    Args:
        target_percent: Target memory usage percentage to wait for
        max_wait: Maximum time to wait in seconds
        check_interval: How often to check memory in seconds
        
    Returns:
        bool: Whether memory dropped below target percentage
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        mem_info = psutil.virtual_memory()
        if mem_info.percent < target_percent:
            print(f"Memory usage dropped to {mem_info.percent}%, continuing...")
            return True
        print(f"Waiting for memory to drop below {target_percent}% (currently {mem_info.percent}%)...")
        time.sleep(check_interval)
    
    print(f"Memory still at {psutil.virtual_memory().percent}% after waiting {max_wait}s, continuing anyway")
    return False


def force_garbage_collection(pause_seconds=2):
    """
    Force garbage collection and pause to allow memory to be reclaimed.
    
    Args:
        pause_seconds: Number of seconds to pause after garbage collection
        
    Returns:
        float: Current memory usage in MB
    """
    print("Forcing garbage collection...")
    before_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    # Run garbage collection multiple times to ensure cyclic references are cleaned up
    gc.collect()
    time.sleep(pause_seconds / 2)  # Short pause to allow OS to catch up
    gc.collect()
    
    # Longer pause to let the OS reclaim the memory
    print(f"Pausing for {pause_seconds} seconds to allow memory cleanup...")
    time.sleep(pause_seconds)
    
    after_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Memory before: {before_mem:.2f} MB, after: {after_mem:.2f} MB, freed: {max(0, before_mem - after_mem):.2f} MB")
    
    # Wait for system memory to stabilize
    wait_for_memory()
    
    return after_mem

def save_dataset_to_parquet(dataset: Union[Dataset, DatasetDict], output_dir: str, 
                            batch_size: int = 1000, verbose: bool = True):
    """
    Save a Hugging Face dataset to Parquet format efficiently in batches.
    
    Args:
        dataset: The dataset or dataset dictionary to save
        output_dir: Directory to save the Parquet files
        batch_size: Number of rows to process at once for large datasets
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Saving dataset to Parquet at {output_dir}...")
        before_mem, _ = print_memory_usage()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write dataset info
    if isinstance(dataset, DatasetDict):
        splits_info = {split: len(ds) for split, ds in dataset.items()}
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            import json
            json.dump({'splits': splits_info}, f, indent=2)
        
        # Save each split to its own Parquet file
        for split_name, split_dataset in dataset.items():
            split_file = os.path.join(output_dir, f"{split_name}.parquet")
            
            if len(split_dataset) > batch_size:
                # For large datasets, convert to pandas and save in batches
                for i in range(0, len(split_dataset), batch_size):
                    end_idx = min(i + batch_size, len(split_dataset))
                    if verbose:
                        print(f"Processing {split_name} batch {i//batch_size + 1}/{(len(split_dataset)-1)//batch_size + 1} "
                              f"(rows {i}-{end_idx-1})...")
                    
                    # Convert batch to pandas dataframe
                    batch = split_dataset.select(range(i, end_idx))
                    df = pd.DataFrame(batch)
                    
                    # Write in append mode if not the first batch
                    if i == 0:
                        df.to_parquet(split_file, index=False)
                    else:
                        df.to_parquet(split_file, index=False, append=True)
                    
                    # Clean up to free memory with pause
                    del df
                    del batch
                    force_garbage_collection()
            else:
                # For small datasets, convert all at once
                df = pd.DataFrame(split_dataset)
                df.to_parquet(split_file, index=False)
                del df
                force_garbage_collection()
    else:
        # Save a single dataset
        output_file = os.path.join(output_dir, "dataset.parquet")
        
        if len(dataset) > batch_size:
            # For large datasets, convert to pandas and save in batches
            for i in range(0, len(dataset), batch_size):
                end_idx = min(i + batch_size, len(dataset))
                if verbose:
                    print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} "
                          f"(rows {i}-{end_idx-1})...")
                
                # Convert batch to pandas dataframe
                batch = dataset.select(range(i, end_idx))
                df = pd.DataFrame(batch)
                
                # Write in append mode if not the first batch
                if i == 0:
                    df.to_parquet(output_file, index=False)
                else:
                    df.to_parquet(output_file, index=False, append=True)
                
                # Clean up to free memory with pause
                del df
                del batch
                force_garbage_collection()
        else:
            # For small datasets, convert all at once
            df = pd.DataFrame(dataset)
            df.to_parquet(output_file, index=False)
            del df
            force_garbage_collection()
    
    if verbose:
        after_mem, _ = print_memory_usage()
        print(f"Saved dataset to {output_dir}")
        print(f"Memory change: {after_mem - before_mem:.2f} MB")


def load_parquet_as_dataset(input_dir: str, split: Optional[str] = None, 
                            batch_size: int = 5000, verbose: bool = True) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from Parquet format efficiently in batches.
    
    Args:
        input_dir: Directory containing the Parquet files
        split: Specific split to load (if None, loads all splits)
        batch_size: Number of rows to process at once for large datasets
        verbose: Whether to print progress information
    
    Returns:
        The loaded dataset or dataset dictionary
    """
    if verbose:
        print(f"Loading Parquet dataset from {input_dir}...")
        before_mem, _ = print_memory_usage()
    
    # Check if it's a multi-split dataset
    if split is not None:
        # Load specific split
        split_file = os.path.join(input_dir, f"{split}.parquet")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found")
        
        # Read Parquet file efficiently
        df = pd.read_parquet(split_file)
        dataset = Dataset.from_pandas(df)
        del df
        force_garbage_collection()
        
        if verbose:
            after_mem, _ = print_memory_usage()
            print(f"Loaded {split} split with {len(dataset)} samples")
            print(f"Memory change: {after_mem - before_mem:.2f} MB")
        
        return dataset
    else:
        # Try to load all splits
        splits = {}
        split_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
        
        if not split_files:
            raise FileNotFoundError(f"No Parquet files found in {input_dir}")
        
        # Check if it's a single dataset or multiple splits
        if len(split_files) == 1 and split_files[0] == "dataset.parquet":
            # Single dataset
            df = pd.read_parquet(os.path.join(input_dir, "dataset.parquet"))
            dataset = Dataset.from_pandas(df)
            del df
            force_garbage_collection()
            
            if verbose:
                after_mem, _ = print_memory_usage()
                print(f"Loaded dataset with {len(dataset)} samples")
                print(f"Memory change: {after_mem - before_mem:.2f} MB")
            
            return dataset
        else:
            # Multiple splits
            for split_file in split_files:
                if not split_file.endswith('.parquet'):
                    continue
                
                split_name = os.path.splitext(split_file)[0]
                if verbose:
                    print(f"Loading split: {split_name}...")
                
                # Read Parquet file efficiently
                df = pd.read_parquet(os.path.join(input_dir, split_file))
                splits[split_name] = Dataset.from_pandas(df)
                
                del df
                force_garbage_collection()
            
            dataset_dict = DatasetDict(splits)
            
            if verbose:
                after_mem, _ = print_memory_usage()
                print(f"Loaded dataset with splits: {', '.join(dataset_dict.keys())}")
                print(f"Memory change: {after_mem - before_mem:.2f} MB")
            
            return dataset_dict


def process_in_batches(function, data, batch_size=100, **kwargs):
    """
    Process data in batches to avoid memory issues.
    
    Args:
        function: The function to apply to each batch
        data: The data to process
        batch_size: The batch size
        **kwargs: Additional arguments to pass to the function
    
    Returns:
        The combined results
    """
    results = []
    
    for i in range(0, len(data), batch_size):
        end_idx = min(i + batch_size, len(data))
        batch = data[i:end_idx]
        batch_result = function(batch, **kwargs)
        results.extend(batch_result)
        
        # Clean up memory with pause
        del batch
        force_garbage_collection()
    
    return results