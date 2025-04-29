#!/usr/bin/env python3
"""
Benchmark script to compare the performance of Hugging Face and Parquet formats
for the Twibot-22 dataset.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
import psutil
import gc
import sys

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
twibot22_dir = os.path.dirname(script_dir)
sys.path.append(twibot22_dir)

# Import parquet utilities
from utilities.parquet_utils import load_parquet_as_dataset

def print_memory_usage():
    """Print current memory usage of the process and system."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    # Also get system memory info
    system_memory = psutil.virtual_memory()
    system_percent = system_memory.percent
    system_available_mb = system_memory.available / 1024 / 1024
    
    print(f"Memory usage: {memory_mb:.2f} MB")
    print(f"System memory: {system_percent:.1f}% used, {system_available_mb:.2f} MB available")
    
    return memory_mb

def force_garbage_collection():
    """Force garbage collection and print memory usage."""
    gc.collect()
    print_memory_usage()

def get_directory_size(path):
    """Get the size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def benchmark_loading(hf_path, parquet_path):
    """Benchmark loading time for both formats."""
    print("\n=== Benchmarking Loading Time ===")
    
    # Benchmark Hugging Face loading
    print("\nLoading Hugging Face dataset...")
    start_time = time.time()
    memory_before = print_memory_usage()
    
    hf_dataset = load_from_disk(hf_path)
    
    hf_loading_time = time.time() - start_time
    memory_after = print_memory_usage()
    memory_change = memory_after - memory_before
    
    print(f"Hugging Face loading time: {hf_loading_time:.2f} seconds")
    print(f"Memory change: {memory_change:.2f} MB")
    
    # Force garbage collection
    print("\nForcing garbage collection...")
    force_garbage_collection()
    
    # Benchmark Parquet loading
    print("\nLoading Parquet dataset...")
    start_time = time.time()
    memory_before = print_memory_usage()
    
    parquet_dataset = load_parquet_as_dataset(parquet_path)
    
    parquet_loading_time = time.time() - start_time
    memory_after = print_memory_usage()
    memory_change = memory_after - memory_before
    
    print(f"Parquet loading time: {parquet_loading_time:.2f} seconds")
    print(f"Memory change: {memory_change:.2f} MB")
    
    # Force garbage collection
    print("\nForcing garbage collection...")
    force_garbage_collection()
    
    return {
        'hf_loading_time': hf_loading_time,
        'parquet_loading_time': parquet_loading_time
    }

def benchmark_filtering(hf_path, parquet_path):
    """Benchmark filtering operations for both formats."""
    print("\n=== Benchmarking Filtering Operations ===")
    
    # Load datasets
    print("\nLoading datasets...")
    hf_dataset = load_from_disk(hf_path)
    parquet_dataset = load_parquet_as_dataset(parquet_path)
    
    # Benchmark Hugging Face filtering
    print("\nFiltering Hugging Face dataset...")
    start_time = time.time()
    
    filtered_hf = hf_dataset["train"].filter(lambda x: x["label"] == 1)
    
    hf_filtering_time = time.time() - start_time
    print(f"Hugging Face filtering time: {hf_filtering_time:.2f} seconds")
    print(f"Filtered dataset size: {len(filtered_hf)} samples")
    
    # Benchmark Parquet filtering
    print("\nFiltering Parquet dataset...")
    start_time = time.time()
    
    filtered_parquet = parquet_dataset["train"].filter(lambda x: x["label"] == 1)
    
    parquet_filtering_time = time.time() - start_time
    print(f"Parquet filtering time: {parquet_filtering_time:.2f} seconds")
    print(f"Filtered dataset size: {len(filtered_parquet)} samples")
    
    # Force garbage collection
    print("\nForcing garbage collection...")
    force_garbage_collection()
    
    return {
        'hf_filtering_time': hf_filtering_time,
        'parquet_filtering_time': parquet_filtering_time
    }

def benchmark_mapping(hf_path, parquet_path):
    """Benchmark mapping operations for both formats."""
    print("\n=== Benchmarking Mapping Operations ===")
    
    # Load datasets
    print("\nLoading datasets...")
    hf_dataset = load_from_disk(hf_path)
    parquet_dataset = load_parquet_as_dataset(parquet_path)
    
    # Define a simple mapping function
    def add_length(example):
        example["text_length"] = len(example["text"])
        return example
    
    # Benchmark Hugging Face mapping
    print("\nMapping Hugging Face dataset...")
    start_time = time.time()
    
    mapped_hf = hf_dataset["train"].map(add_length)
    
    hf_mapping_time = time.time() - start_time
    print(f"Hugging Face mapping time: {hf_mapping_time:.2f} seconds")
    
    # Benchmark Parquet mapping
    print("\nMapping Parquet dataset...")
    start_time = time.time()
    
    mapped_parquet = parquet_dataset["train"].map(add_length)
    
    parquet_mapping_time = time.time() - start_time
    print(f"Parquet mapping time: {parquet_mapping_time:.2f} seconds")
    
    # Force garbage collection
    print("\nForcing garbage collection...")
    force_garbage_collection()
    
    return {
        'hf_mapping_time': hf_mapping_time,
        'parquet_mapping_time': parquet_mapping_time
    }

def plot_results(results, output_dir):
    """Plot benchmark results."""
    print("\n=== Plotting Results ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage comparison
    storage_data = results['storage']
    labels = list(storage_data.keys())
    hf_sizes = [storage_data[label]['hf_size'] for label in labels]
    parquet_sizes = [storage_data[label]['parquet_size'] for label in labels]
    compression_ratios = [storage_data[label]['compression_ratio'] for label in labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, hf_sizes, width, label='Hugging Face')
    ax1.bar(x + width/2, parquet_sizes, width, label='Parquet')
    ax1.set_ylabel('Size (MB)')
    ax1.set_title('Storage Size Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    ax2.bar(x, compression_ratios, width)
    ax2.set_ylabel('Compression Ratio (HF/Parquet)')
    ax2.set_title('Compression Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'storage_comparison.png'))
    
    # Performance comparison
    performance_data = {
        'Loading Time': {
            'Hugging Face': results['loading']['hf_loading_time'],
            'Parquet': results['loading']['parquet_loading_time']
        },
        'Filtering Time': {
            'Hugging Face': results['filtering']['hf_filtering_time'],
            'Parquet': results['filtering']['parquet_filtering_time']
        },
        'Mapping Time': {
            'Hugging Face': results['mapping']['hf_mapping_time'],
            'Parquet': results['mapping']['parquet_mapping_time']
        }
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(performance_data))
    width = 0.35
    
    hf_times = [performance_data[op]['Hugging Face'] for op in performance_data]
    parquet_times = [performance_data[op]['Parquet'] for op in performance_data]
    
    ax.bar(x - width/2, hf_times, width, label='Hugging Face')
    ax.bar(x + width/2, parquet_times, width, label='Parquet')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(performance_data.keys())
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    
    # Generate markdown report
    report = f"""# Parquet vs Hugging Face Performance Comparison for Twibot-22

## Storage Efficiency

| Dataset | Hugging Face Size (MB) | Parquet Size (MB) | Compression Ratio |
|---------|------------------------|-------------------|-------------------|
"""
    
    for label in labels:
        report += f"| {label} | {storage_data[label]['hf_size']:.2f} | {storage_data[label]['parquet_size']:.2f} | {storage_data[label]['compression_ratio']:.2f}x |\n"
    
    report += f"""
## Performance Comparison

| Operation | Hugging Face (seconds) | Parquet (seconds) | Speedup |
|-----------|------------------------|-------------------|---------|
| Loading | {results['loading']['hf_loading_time']:.2f} | {results['loading']['parquet_loading_time']:.2f} | {results['loading']['hf_loading_time'] / results['loading']['parquet_loading_time']:.2f}x |
| Filtering | {results['filtering']['hf_filtering_time']:.2f} | {results['filtering']['parquet_filtering_time']:.2f} | {results['filtering']['hf_filtering_time'] / results['filtering']['parquet_filtering_time']:.2f}x |
| Mapping | {results['mapping']['hf_mapping_time']:.2f} | {results['mapping']['parquet_mapping_time']:.2f} | {results['mapping']['hf_mapping_time'] / results['mapping']['parquet_mapping_time']:.2f}x |

## Conclusion

### Storage Efficiency
Parquet format provides significant storage savings compared to the Hugging Face disk format:
- For the processed dataset: {storage_data['Processed Dataset']['compression_ratio']:.2f}x smaller
- For the tokenized dataset: {storage_data['Tokenized Dataset']['compression_ratio']:.2f}x smaller

### Performance
"""
    
    # Add performance conclusion based on results
    if results['loading']['hf_loading_time'] < results['loading']['parquet_loading_time']:
        report += "- **Loading**: Hugging Face format loads faster than Parquet.\n"
    else:
        report += "- **Loading**: Parquet format loads faster than Hugging Face.\n"
        
    if results['filtering']['hf_filtering_time'] < results['filtering']['parquet_filtering_time']:
        report += "- **Filtering**: Hugging Face format filters faster than Parquet.\n"
    else:
        report += "- **Filtering**: Parquet format filters faster than Hugging Face.\n"
        
    if results['mapping']['hf_mapping_time'] < results['mapping']['parquet_mapping_time']:
        report += "- **Mapping**: Hugging Face format maps faster than Parquet.\n"
    else:
        report += "- **Mapping**: Parquet format maps faster than Hugging Face.\n"
    
    report += """
### When to Use Each Format

- **Hugging Face Disk Format**: Recommended when processing speed is the priority and disk space is not a concern.
- **Apache Parquet Format**: Recommended when storage efficiency is important or when working with larger datasets.

The choice between formats involves a trade-off between storage efficiency and processing speed.
"""
    
    with open(os.path.join(output_dir, 'parquet_performance.md'), 'w') as f:
        f.write(report)
    
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark Hugging Face vs Parquet formats for Twibot-22')
    parser.add_argument('--hf-processed', type=str, default='data/twibot22_balanced_dataset',
                        help='Path to the processed dataset in Hugging Face format')
    parser.add_argument('--parquet-processed', type=str, default='data/twibot22_balanced_parquet',
                        help='Path to the processed dataset in Parquet format')
    parser.add_argument('--hf-tokenized', type=str, default='data/twibot22_balanced_tokenized',
                        help='Path to the tokenized dataset in Hugging Face format')
    parser.add_argument('--parquet-tokenized', type=str, default='data/twibot22_balanced_tokenized_parquet',
                        help='Path to the tokenized dataset in Parquet format')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results')
    args = parser.parse_args()
    
    # Resolve paths
    hf_processed = os.path.join(twibot22_dir, args.hf_processed)
    parquet_processed = os.path.join(twibot22_dir, args.parquet_processed)
    hf_tokenized = os.path.join(twibot22_dir, args.hf_tokenized)
    parquet_tokenized = os.path.join(twibot22_dir, args.parquet_tokenized)
    output_dir = os.path.join(twibot22_dir, args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Measure storage sizes
    print("\n=== Measuring Storage Sizes ===")
    
    hf_processed_size = get_directory_size(hf_processed)
    parquet_processed_size = get_directory_size(parquet_processed)
    hf_tokenized_size = get_directory_size(hf_tokenized)
    parquet_tokenized_size = get_directory_size(parquet_tokenized)
    
    print(f"Hugging Face processed dataset size: {hf_processed_size:.2f} MB")
    print(f"Parquet processed dataset size: {parquet_processed_size:.2f} MB")
    print(f"Compression ratio: {hf_processed_size / parquet_processed_size:.2f}x")
    
    print(f"Hugging Face tokenized dataset size: {hf_tokenized_size:.2f} MB")
    print(f"Parquet tokenized dataset size: {parquet_tokenized_size:.2f} MB")
    print(f"Compression ratio: {hf_tokenized_size / parquet_tokenized_size:.2f}x")
    
    storage_results = {
        'Processed Dataset': {
            'hf_size': hf_processed_size,
            'parquet_size': parquet_processed_size,
            'compression_ratio': hf_processed_size / parquet_processed_size
        },
        'Tokenized Dataset': {
            'hf_size': hf_tokenized_size,
            'parquet_size': parquet_tokenized_size,
            'compression_ratio': hf_tokenized_size / parquet_tokenized_size
        }
    }
    
    # Run benchmarks on processed dataset
    print("\n=== Running Benchmarks on Processed Dataset ===")
    loading_results = benchmark_loading(hf_processed, parquet_processed)
    filtering_results = benchmark_filtering(hf_processed, parquet_processed)
    mapping_results = benchmark_mapping(hf_processed, parquet_processed)
    
    # Collect results
    results = {
        'storage': storage_results,
        'loading': loading_results,
        'filtering': filtering_results,
        'mapping': mapping_results
    }
    
    # Plot results
    plot_results(results, output_dir)

if __name__ == "__main__":
    main()
