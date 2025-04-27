"""
Benchmark Parquet vs Hugging Face Dataset Performance

This script compares the performance of Apache Parquet and Hugging Face dataset formats
for the Twitter bot detection project, measuring:
1. Storage efficiency (file size)
2. Loading time
3. Processing time
4. Memory usage
"""

import os
import time
import sys
import gc
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
import numpy as np

# Add project root to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import parquet utilities
from utilities.parquet_utils import load_parquet_as_dataset

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_directory_size(path):
    """Get the size of a directory in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def benchmark_loading(hf_path, parquet_path, num_runs=5):
    """Benchmark dataset loading time"""
    hf_times = []
    parquet_times = []
    hf_memory = []
    parquet_memory = []

    print(f"Benchmarking loading time ({num_runs} runs each)...")

    for i in range(num_runs):
        # Clear memory
        gc.collect()
        initial_memory = get_memory_usage()

        # Benchmark Hugging Face
        start_time = time.time()
        hf_dataset = load_from_disk(hf_path)
        hf_load_time = time.time() - start_time
        hf_times.append(hf_load_time)
        hf_memory.append(get_memory_usage() - initial_memory)

        # Clear memory
        del hf_dataset
        gc.collect()
        initial_memory = get_memory_usage()

        # Benchmark Parquet
        start_time = time.time()
        parquet_dataset = load_parquet_as_dataset(parquet_path)
        parquet_load_time = time.time() - start_time
        parquet_times.append(parquet_load_time)
        parquet_memory.append(get_memory_usage() - initial_memory)

        # Clear memory
        del parquet_dataset
        gc.collect()

        print(f"  Run {i+1}: HF={hf_load_time:.3f}s, Parquet={parquet_load_time:.3f}s")

    return {
        'hf_load_time': np.mean(hf_times),
        'hf_load_std': np.std(hf_times),
        'parquet_load_time': np.mean(parquet_times),
        'parquet_load_std': np.std(parquet_times),
        'hf_memory': np.mean(hf_memory),
        'parquet_memory': np.mean(parquet_memory)
    }

def benchmark_processing(hf_path, parquet_path, num_runs=3):
    """Benchmark dataset processing time (filtering, mapping)"""
    hf_times = []
    parquet_times = []

    print(f"Benchmarking processing time ({num_runs} runs each)...")

    for i in range(num_runs):
        # Load datasets
        hf_dataset = load_from_disk(hf_path)
        parquet_dataset = load_parquet_as_dataset(parquet_path)

        # Benchmark Hugging Face processing
        start_time = time.time()
        # Filter for bot accounts
        filtered_hf = hf_dataset['train'].filter(lambda x: x['label'] == 1)
        # Map to extract text length
        mapped_hf = filtered_hf.map(lambda x: {'text_length': len(x['text'])})
        # Sort by text length
        sorted_hf = mapped_hf.sort('text_length', reverse=True)
        # Select top 100
        top_hf = sorted_hf.select(range(min(100, len(sorted_hf))))
        hf_process_time = time.time() - start_time
        hf_times.append(hf_process_time)

        # Clear memory
        del hf_dataset, filtered_hf, mapped_hf, sorted_hf, top_hf
        gc.collect()

        # Benchmark Parquet processing
        start_time = time.time()
        # Filter for bot accounts
        filtered_parquet = parquet_dataset['train'].filter(lambda x: x['label'] == 1)
        # Map to extract text length
        mapped_parquet = filtered_parquet.map(lambda x: {'text_length': len(x['text'])})
        # Sort by text length
        sorted_parquet = mapped_parquet.sort('text_length', reverse=True)
        # Select top 100
        top_parquet = sorted_parquet.select(range(min(100, len(sorted_parquet))))
        parquet_process_time = time.time() - start_time
        parquet_times.append(parquet_process_time)

        # Clear memory
        del parquet_dataset, filtered_parquet, mapped_parquet, sorted_parquet, top_parquet
        gc.collect()

        print(f"  Run {i+1}: HF={hf_process_time:.3f}s, Parquet={parquet_process_time:.3f}s")

    return {
        'hf_process_time': np.mean(hf_times),
        'hf_process_std': np.std(hf_times),
        'parquet_process_time': np.mean(parquet_times),
        'parquet_process_std': np.std(parquet_times)
    }

def benchmark_tokenized_loading(hf_path, parquet_path, num_runs=3):
    """Benchmark tokenized dataset loading time"""
    hf_times = []
    parquet_times = []

    print(f"Benchmarking tokenized dataset loading time ({num_runs} runs each)...")

    for i in range(num_runs):
        # Clear memory
        gc.collect()

        # Benchmark Hugging Face
        start_time = time.time()
        hf_dataset = load_from_disk(hf_path)
        hf_load_time = time.time() - start_time
        hf_times.append(hf_load_time)

        # Clear memory
        del hf_dataset
        gc.collect()

        # Benchmark Parquet
        start_time = time.time()
        parquet_dataset = load_parquet_as_dataset(parquet_path)
        parquet_load_time = time.time() - start_time
        parquet_times.append(parquet_load_time)

        # Clear memory
        del parquet_dataset
        gc.collect()

        print(f"  Run {i+1}: HF={hf_load_time:.3f}s, Parquet={parquet_load_time:.3f}s")

    return {
        'hf_tokenized_load_time': np.mean(hf_times),
        'hf_tokenized_load_std': np.std(hf_times),
        'parquet_tokenized_load_time': np.mean(parquet_times),
        'parquet_tokenized_load_std': np.std(parquet_times)
    }

def create_comparison_charts(results, output_dir):
    """Create comparison charts for the benchmark results"""
    os.makedirs(output_dir, exist_ok=True)

    # Storage comparison
    labels = ['Fixed Dataset', 'Tokenized Dataset']
    hf_sizes = [results['hf_fixed_size'], results['hf_tokenized_size']]
    parquet_sizes = [results['parquet_fixed_size'], results['parquet_tokenized_size']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, hf_sizes, width, label='Hugging Face')
    rects2 = ax.bar(x + width/2, parquet_sizes, width, label='Parquet')

    ax.set_ylabel('Size (MB)')
    ax.set_title('Storage Size Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add compression ratio labels
    for i, (hf, pq) in enumerate(zip(hf_sizes, parquet_sizes)):
        ratio = hf / pq
        ax.text(i, max(hf, pq) + 1, f'{ratio:.1f}x smaller',
                ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'storage_comparison.png'))

    # Loading time comparison
    labels = ['Fixed Dataset', 'Tokenized Dataset']
    hf_times = [results['hf_load_time'], results['hf_tokenized_load_time']]
    parquet_times = [results['parquet_load_time'], results['parquet_tokenized_load_time']]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, hf_times, width, label='Hugging Face')
    rects2 = ax.bar(x + width/2, parquet_times, width, label='Parquet')

    ax.set_ylabel('Loading Time (seconds)')
    ax.set_title('Loading Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add speedup labels
    for i, (hf, pq) in enumerate(zip(hf_times, parquet_times)):
        speedup = hf / pq
        ax.text(i, max(hf, pq) + 0.1, f'{speedup:.1f}x faster',
                ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loading_comparison.png'))

    # Processing time comparison
    labels = ['Processing Time']
    hf_times = [results['hf_process_time']]
    parquet_times = [results['parquet_process_time']]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, hf_times, width, label='Hugging Face')
    rects2 = ax.bar(x + width/2, parquet_times, width, label='Parquet')

    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Processing Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add speedup label
    speedup = hf_times[0] / parquet_times[0]
    ax.text(0, max(hf_times[0], parquet_times[0]) + 0.1, f'{speedup:.1f}x faster',
            ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'processing_comparison.png'))

    # Memory usage comparison
    labels = ['Memory Usage']
    hf_memory = [results['hf_memory']]
    parquet_memory = [results['parquet_memory']]

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, hf_memory, width, label='Hugging Face')
    rects2 = ax.bar(x + width/2, parquet_memory, width, label='Parquet')

    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add memory reduction label
    reduction = hf_memory[0] / parquet_memory[0]
    ax.text(0, max(hf_memory[0], parquet_memory[0]) + 5, f'{reduction:.1f}x less memory',
            ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))

    return True

def generate_markdown_report(results, output_file):
    """Generate a markdown report of the benchmark results"""
    with open(output_file, 'w') as f:
        f.write("# Parquet vs Hugging Face Performance Comparison\n\n")

        f.write("## 1. Storage Efficiency\n\n")
        f.write("| Dataset Type | Hugging Face Size (MB) | Parquet Size (MB) | Compression Ratio |\n")
        f.write("|--------------|------------------------|-------------------|-------------------|\n")
        f.write(f"| Fixed Dataset | {results['hf_fixed_size']:.2f} | {results['parquet_fixed_size']:.2f} | {results['hf_fixed_size']/results['parquet_fixed_size']:.2f}x |\n")
        f.write(f"| Tokenized Dataset | {results['hf_tokenized_size']:.2f} | {results['parquet_tokenized_size']:.2f} | {results['hf_tokenized_size']/results['parquet_tokenized_size']:.2f}x |\n\n")

        f.write("![Storage Comparison](benchmark_results/storage_comparison.png)\n\n")

        f.write("## 2. Loading Time\n\n")
        f.write("| Dataset Type | Hugging Face (seconds) | Parquet (seconds) | Speed Improvement |\n")
        f.write("|--------------|------------------------|-------------------|-------------------|\n")
        f.write(f"| Fixed Dataset | {results['hf_load_time']:.3f} ± {results['hf_load_std']:.3f} | {results['parquet_load_time']:.3f} ± {results['parquet_load_std']:.3f} | {results['hf_load_time']/results['parquet_load_time']:.2f}x |\n")
        f.write(f"| Tokenized Dataset | {results['hf_tokenized_load_time']:.3f} ± {results['hf_tokenized_load_std']:.3f} | {results['parquet_tokenized_load_time']:.3f} ± {results['parquet_tokenized_load_std']:.3f} | {results['hf_tokenized_load_time']/results['parquet_tokenized_load_time']:.2f}x |\n\n")

        f.write("![Loading Comparison](benchmark_results/loading_comparison.png)\n\n")

        f.write("## 3. Processing Performance\n\n")
        f.write("| Operation | Hugging Face (seconds) | Parquet (seconds) | Speed Improvement |\n")
        f.write("|-----------|------------------------|-------------------|-------------------|\n")
        f.write(f"| Filter, Map, Sort | {results['hf_process_time']:.3f} ± {results['hf_process_std']:.3f} | {results['parquet_process_time']:.3f} ± {results['parquet_process_std']:.3f} | {results['hf_process_time']/results['parquet_process_time']:.2f}x |\n\n")

        f.write("![Processing Comparison](benchmark_results/processing_comparison.png)\n\n")

        f.write("## 4. Memory Usage\n\n")
        f.write("| Dataset Type | Hugging Face (MB) | Parquet (MB) | Memory Reduction |\n")
        f.write("|--------------|-------------------|--------------|------------------|\n")
        f.write(f"| Fixed Dataset | {results['hf_memory']:.2f} | {results['parquet_memory']:.2f} | {results['hf_memory']/results['parquet_memory']:.2f}x |\n\n")

        f.write("![Memory Comparison](benchmark_results/memory_comparison.png)\n\n")

        f.write("## 5. Summary\n\n")
        f.write("Apache Parquet format provides significant advantages over the Hugging Face dataset format:\n\n")
        f.write(f"- **Storage**: {((results['hf_fixed_size'] + results['hf_tokenized_size']) / (results['parquet_fixed_size'] + results['parquet_tokenized_size'])):.1f}x smaller file sizes on average\n")
        f.write(f"- **Loading Speed**: {((results['hf_load_time'] + results['hf_tokenized_load_time']) / (results['parquet_load_time'] + results['parquet_tokenized_load_time'])):.1f}x faster loading times on average\n")
        f.write(f"- **Processing Speed**: {(results['hf_process_time'] / results['parquet_process_time']):.1f}x faster processing operations\n")
        f.write(f"- **Memory Efficiency**: {(results['hf_memory'] / results['parquet_memory']):.1f}x lower memory usage\n\n")

        f.write("These improvements are particularly valuable when working with larger datasets, where efficiency gains can significantly reduce processing time and resource requirements.\n")

    return True

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Fixed dataset paths
    hf_fixed_path = os.path.join(project_root, "data", "twibot20_fixed_dataset")
    parquet_fixed_path = os.path.join(project_root, "data", "twibot20_fixed_parquet")

    # Tokenized dataset paths
    hf_tokenized_path = os.path.join(project_root, "data", "twibot20_fixed_tokenized")
    parquet_tokenized_path = os.path.join(project_root, "data", "twibot20_tokenized_parquet")

    # Output paths
    output_dir = os.path.join(project_root, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    # Check if paths exist
    if not os.path.exists(hf_fixed_path) or not os.path.exists(parquet_fixed_path):
        print("Error: Fixed dataset paths not found. Please run scripts/1_fix_dataset.py first.")
        return

    if not os.path.exists(hf_tokenized_path) or not os.path.exists(parquet_tokenized_path):
        print("Error: Tokenized dataset paths not found. Please run scripts/2_tokenize_dataset.py first.")
        return

    # Collect results
    results = {}

    # 1. Measure storage size
    print("\n=== Measuring Storage Size ===")
    results['hf_fixed_size'] = get_directory_size(hf_fixed_path)
    results['parquet_fixed_size'] = get_directory_size(parquet_fixed_path)
    results['hf_tokenized_size'] = get_directory_size(hf_tokenized_path)
    results['parquet_tokenized_size'] = get_directory_size(parquet_tokenized_path)

    print(f"Fixed Dataset Size:")
    print(f"  Hugging Face: {results['hf_fixed_size']:.2f} MB")
    print(f"  Parquet: {results['parquet_fixed_size']:.2f} MB")
    print(f"  Compression Ratio: {results['hf_fixed_size'] / results['parquet_fixed_size']:.2f}x")

    print(f"Tokenized Dataset Size:")
    print(f"  Hugging Face: {results['hf_tokenized_size']:.2f} MB")
    print(f"  Parquet: {results['parquet_tokenized_size']:.2f} MB")
    print(f"  Compression Ratio: {results['hf_tokenized_size'] / results['parquet_tokenized_size']:.2f}x")

    # 2. Benchmark loading time
    print("\n=== Benchmarking Loading Time ===")
    loading_results = benchmark_loading(hf_fixed_path, parquet_fixed_path)
    results.update(loading_results)

    # 3. Benchmark processing time
    print("\n=== Benchmarking Processing Time ===")
    processing_results = benchmark_processing(hf_fixed_path, parquet_fixed_path)
    results.update(processing_results)

    # 4. Benchmark tokenized dataset loading
    print("\n=== Benchmarking Tokenized Dataset Loading ===")
    tokenized_loading_results = benchmark_tokenized_loading(hf_tokenized_path, parquet_tokenized_path)
    results.update(tokenized_loading_results)

    # 5. Create comparison charts
    print("\n=== Creating Comparison Charts ===")
    create_comparison_charts(results, output_dir)

    # 6. Generate markdown report
    print("\n=== Generating Markdown Report ===")
    report_path = os.path.join(output_dir, "parquet_performance.md")
    generate_markdown_report(results, report_path)

    print(f"\nBenchmark completed! Results saved to {output_dir}")
    print(f"Markdown report: {report_path}")

if __name__ == "__main__":
    main()
