# Tests and Bugs : Twibot-22 Extension

This document outlines the testing procedures and bugs encountered specifically for the Twibot-22 extension of the Twitter Bot Detection project. It focuses only on issues unique to the Twibot-22 implementation, without repeating general issues documented in the main project's `tests_and_bugs.md`.

## 1. Testing Procedures Specific to Twibot-22

### 1.1. Tweet Extraction Tests

#### 1.1.1. Balanced Extraction Test
- **Purpose :** Verify that exactly 500 bot tweets and 500 human tweets are extracted
- **Procedure :**
  - Run `scripts/1_extract_tweets.py` with `--bot-tweets 500 --human-tweets 500`
  - Check the output files to verify the counts
- **Expected Result :** Exactly 500 bot tweets and 500 human tweets are extracted and saved

#### 1.1.2. Memory Efficiency Test
- **Purpose :** Ensure that the extraction process doesn't exceed memory limits
- **Procedure :**
  - Run `scripts/1_extract_tweets.py` with memory monitoring enabled
  - Monitor memory usage during processing of large tweet files
- **Expected Result :** Memory usage should remain stable and below system limits, even when processing 10GB+ files

#### 1.1.3. Multiprocessing Test
- **Purpose :** Verify that parallel processing works correctly
- **Procedure :**
  - Run `scripts/1_extract_tweets.py` with different `--processes` values
  - Monitor CPU usage and extraction speed
- **Expected Result :** Higher process counts should utilize more CPU cores and potentially speed up extraction

### 1.2. Dataset Preparation Tests

#### 1.2.1. Class Balance Test
- **Purpose :** Ensure that the train/validation/test splits maintain class balance
- **Procedure :**
  - Run `scripts/prepare_dataset.py`
  - Check the reported class distribution in each split
- **Expected Result :** Each split should maintain approximately 50% bot and 50% human samples

#### 1.2.2. Split Ratio Test
- **Purpose :** Verify that the dataset is split according to the specified ratios
- **Procedure :**
  - Run `scripts/prepare_dataset.py` with different `--test-split` and `--validation-split` values
  - Check the reported split sizes
- **Expected Result :** Split sizes should match the specified ratios (default : 80% train, 10% validation, 10% test)

### 1.3. Benchmark Tests

#### 1.3.1. Format Comparison Test
- **Purpose :** Compare performance between Hugging Face and Parquet formats
- **Procedure :**
  - Run `scripts/benchmark_parquet.py`
  - Analyze the generated metrics and charts
- **Expected Result :** Parquet should show storage benefits but may have different performance characteristics for various operations

## 2. Bugs and Issues Specific to Twibot-22

### 2.1. Tweet Extraction Issues

#### 2.1.1. JSON Parsing Complexity
- **Issue :** Tweet JSON files were not in standard JSON Lines format and required special parsing
- **Impact :** Standard JSON parsing would fail due to objects spanning multiple lines or multiple objects per line
- **Resolution :** Implemented custom parsing using `json.JSONDecoder().raw_decode()` to handle non-standard JSON
- **Prevention :** Added robust error handling and progress tracking during parsing

#### 2.1.2. Data Imbalance
- **Issue :** The Twibot-22 dataset had a severe class imbalance with only about 10% bot content
- **Impact :** Training on the raw dataset would result in biased models that favor the majority class
- **Resolution :** Implemented balanced extraction to create a dataset with equal bot and human samples
- **Prevention :** Added explicit target counts for bot and human tweets with automatic stopping

#### 2.1.3. Memory Consumption with Large Files
- **Issue :** Processing the large tweet files (10GB+) caused excessive memory usage
- **Impact :** Scripts would crash or cause system swapping on machines with limited RAM
- **Resolution :** Implemented chunked processing, multiprocessing with controlled worker count, and aggressive garbage collection
- **Prevention :** Added memory monitoring that throttles processing when system memory usage exceeds thresholds

### 2.2. Performance Issues

#### 2.2.1. Slow JSON Processing
- **Issue :** Processing the large JSON files was extremely slow with standard methods
- **Impact :** Extraction could take hours or days with naive approaches
- **Resolution :** Implemented parallel processing with multiple worker processes and chunked file reading
- **Prevention :** Added progress reporting and early stopping once target counts are reached

#### 2.2.2. Parquet Loading Overhead
- **Issue :** While Parquet provided storage benefits, loading Parquet datasets was slower than Hugging Face format
- **Impact :** Initial loading time was significantly higher for Parquet format
- **Resolution :** Documented the trade-offs in the README and benchmark results
- **Prevention :** Provided both formats as options, allowing users to choose based on their priorities

### 2.3. Integration Issues

#### 2.3.1. Path Resolution
- **Issue :** Scripts had difficulty resolving paths correctly when run from different directories
- **Impact :** Scripts would fail to find modules or data files when not run from the expected location
- **Resolution :** Implemented absolute path resolution based on script location
- **Prevention :** Added robust path handling that works regardless of the current working directory

#### 2.3.2. Utility Module Import
- **Issue :** The Twibot-22 scripts needed to import utility modules from the main project
- **Impact :** Import errors occurred when the main project utilities weren't in the Python path
- **Resolution :** Added the project root to the Python path in each script
- **Prevention :** Standardized the import approach across all scripts

## 3. Optimization Strategies Specific to Twibot-22

### 3.1. Memory-Efficient Tweet Extraction

- **Strategy :** Process large JSON files in small chunks with controlled memory usage
- **Implementation :** 
  - Used 1MB chunk size for file reading
  - Implemented memory monitoring with `psutil`
  - Added throttling when memory usage exceeds 80%
  - Forced garbage collection between chunks
- **Result :** Successfully processed 10GB+ files on systems with limited RAM

### 3.2. Parallel Processing

- **Strategy :** Utilize multiple CPU cores for faster processing
- **Implementation :**
  - Created a worker pool with controlled number of processes
  - Distributed file chunks to workers
  - Collected results through a queue system
  - Automatically adjusted worker count based on available cores
- **Result :** Significantly faster extraction on multi-core systems

### 3.3. Early Stopping

- **Strategy :** Stop processing once target counts are reached
- **Implementation :**
  - Tracked bot and human tweet counts during extraction
  - Implemented early exit from file processing when targets are met
  - Added progress reporting to show completion percentage
- **Result :** Avoided unnecessary processing of the entire dataset

## 4. Lessons Learned Specific to Twibot-22

1. **Balanced Datasets Matter :** Creating a balanced dataset (500 bot/500 human tweets) led to better model performance than using the imbalanced raw data.

2. **Memory Efficiency is Critical :** When working with very large files (10GB+), memory-efficient processing techniques are essential, even on systems with substantial RAM.

3. **Parallel Processing Trade-offs :** While parallel processing speeds up extraction, it increases memory usage. Finding the right balance is important.

4. **Format Trade-offs :** Parquet format provides significant storage benefits but may have performance trade-offs for certain operations compared to the Hugging Face format.

5. **Custom JSON Parsing :** Standard JSON parsing libraries may struggle with non-standard JSON formats. Custom parsing approaches may be necessary for real-world data.

6. **Progress Monitoring :** For long-running processes, detailed progress monitoring and reporting are essential for debugging and user experience.

7. **Early Stopping :** When extracting a subset of data, implementing early stopping once targets are reached can save significant processing time.
