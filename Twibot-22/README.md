# Twibot-22 Extension: Twitter Bot Detection with Balanced Dataset

## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Structure](#2-project-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Data](#4-data)
    - 4.1. Source Files
    - 4.2. Generated Data Formats
    - 4.3. Data Preparation Steps
    - 4.4. Data Structures
5. [Methodology](#5-methodology)
    - 5.1. Preprocessing - Tokenization
    - 5.2. Model Architecture
    - 5.3. Training Process
6. [Results](#6-results)
7. [Discussion & Conclusion](#7-discussion--conclusion)
    - 7.1. Interpretation
    - 7.2. Limitations
    - 7.3. Conclusion
    - 7.4. Feature Engineering Opportunities
8. [Usage Instructions](#8-usage-instructions)
    - 8.1. Prerequisites
    - 8.2. Running the Pipeline (Default - HF Format)
    - 8.3. Running the Pipeline (Optional - Parquet Format)
    - 8.4. Benchmarking Parquet vs Hugging Face (Optional)
9. [API and Module Documentation](#9-api-and-module-documentation)
    - 9.1. `scripts/1_extract_tweets.py`
    - 9.2. `scripts/prepare_dataset.py`
    - 9.3. `scripts/2_tokenize_balanced_dataset.py`
    - 9.4. `scripts/3_train_model.py`
    - 9.5. `scripts/4_predict.py`
    - 9.6. `scripts/benchmark_parquet.py`
    - 9.7. `utilities/parquet_utils.py`
10. [Data Processing Workflow Details](#10-data-processing-workflow-details)
    - 10.1. Raw Data Loading
    - 10.2. Tweet Extraction and Cleaning
    - 10.3. Dataset Creation and Formatting
    - 10.4. Dataset Splitting
    - 10.5. Tokenization for Model Input
    - 10.6. Data Format Conversion (Optional)
    - 10.7. Data Flow Summary

---

## 1. Introduction

- **Purpose:** This is an optional extension to the main Twitter Bot Detection project, focusing on the Twibot-22 dataset.
- **Objective:** Train a DistilBERT model for Twitter bot detection using a balanced subset of the Twibot-22 dataset.
- **Approach:** Instead of using the entire imbalanced dataset, we create a balanced dataset with 500 bot tweets and 500 human tweets to improve model training.
- **Advantages:** This approach offers improved training stability, faster training times, better generalization, and reduced computational resource requirements.

## 2. Project Structure

```
/
├── Twibot-22/                      # Main directory for the extension project
│   ├── data/                       # Processed datasets
│   │   ├── twibot22_balanced_dataset/  # Balanced dataset in HF format (output from prepare_dataset.py)
│   │   ├── twibot22_balanced_tokenized/# Tokenized balanced dataset in HF format (output from 2_tokenize_balanced_dataset.py)
│   │   ├── twibot22_balanced_parquet/  # Balanced dataset in Parquet format
│   │   └── twibot22_balanced_tokenized_parquet/ # Tokenized balanced dataset in Parquet format
│   │
│   ├── models/                     # Trained models
│   │   └── bot_detection_model/    # DistilBERT model trained on balanced dataset
│   │
│   ├── scripts/                    # Pipeline scripts
│   │   ├── 1_extract_tweets.py     # Step 1: Extract 1000 tweets (500 bot, 500 human)
│   │   ├── prepare_dataset.py      # Step 2: Process extracted tweets and create train/val/test splits
│   │   ├── 2_tokenize_balanced_dataset.py # Step 3: Tokenize the balanced dataset
│   │   ├── 3_train_model.py        # Step 4: Train the model on tokenized data
│   │   ├── 4_predict.py            # Step 5: Make predictions with the trained model
│   │   └── benchmark_parquet.py    # Optional: Compare performance between formats
│   │
│   ├── utilities/                  # Helper modules
│   │   └── parquet_utils.py        # Apache Parquet utilities
│   │
│   ├── benchmark_results/          # Performance comparison results
│   │   ├── parquet_performance.md  # Detailed performance metrics
│   │   ├── performance_comparison.png # Performance comparison chart
│   │   └── storage_comparison.png  # Storage efficiency chart
│   │
│   ├── tweet_*.json                # Raw tweet data files
│   ├── label.csv                   # User labels (bot/human)
│   ├── split.csv                   # Dataset splits (train/test/dev)
│   ├── edge.csv                    # User follower/following relationships (optional)
│   ├── hashtag.json                # Hashtag data (optional)
│   ├── list.json                   # Twitter list data (optional)
│   └── README.md                   # This file
│
└── extracted_1000_tweets/          # Intermediate directory containing 1000 extracted tweets
    ├── tweets.txt                  # Raw text of 1000 tweets (500 bot, 500 human)
    ├── labels.txt                  # Labels for each tweet (1=bot, 0=human)
    └── dataset.csv                 # CSV format of the extracted tweets with labels
```

### 2.1. Additional Data Files (Optional)

The project includes several optional data files that are not used in the current pipeline but could be valuable for advanced analysis:

- **edge.csv** (6.2GB): Contains Twitter follower/following relationships between users. Useful for network-based bot detection approaches.
- **hashtag.json** (255MB): Contains hashtag data with IDs and tag names. Useful for hashtag-based analysis.
- **list.json** (4.7MB): Contains Twitter list data with details like name, description, and follower count. Provides additional context about user interests.

These files are not required for the basic pipeline but can be used to extend the model with additional features or for more sophisticated analysis approaches.

## 3. Pipeline Overview

The project follows a 5-step pipeline, executed via scripts in the `scripts/` directory:

1. **Data Extraction** (`1_extract_tweets.py`): Extracts exactly 500 bot tweets and 500 human tweets from the Twibot-22 dataset and saves them to text files.

2. **Dataset Preparation** (`prepare_dataset.py`): Processes the extracted tweets and creates a balanced dataset with train (80%), validation (10%), and test (10%) splits.

3. **Tokenization** (`2_tokenize_balanced_dataset.py`): Tokenizes the balanced dataset using the DistilBERT tokenizer.

4. **Model Training** (`3_train_model.py`): Trains a DistilBERT model on the tokenized balanced dataset, evaluates it, and saves the best model.

5. **Prediction** (`4_predict.py`): Loads the trained model and provides an interface for classifying new text samples.

Each step supports using either the standard Hugging Face dataset format or the Apache Parquet format via the `--use-parquet` flag for enhanced storage efficiency. Additionally, the `benchmark_parquet.py` script can be used to compare the performance of both formats.

## 4. Data

### 4.1. Source Files
- **Location:** Expected in the `Twibot-22/` directory
- **Files Used:**
  - `tweet_*.json`: Contains tweet data. These are large JSON files with tweet information.
  - `label.csv`: Maps user IDs to labels (bot/human).
  - `split.csv`: Defines original train/test/dev user ID lists.

### 4.2. Generated Data Formats
The pipeline generates processed and tokenized datasets, which can be stored in two formats:

1. **Hugging Face Disk Format:**
   - Default format. Stored in `data/twibot22_balanced_dataset/` and `data/twibot22_balanced_tokenized/`.
   - Consists of `dataset_dict.json`, `dataset_info.json`, and subfolders for each split containing Apache Arrow files (`.arrow`) and index files (`.idx`).

2. **Apache Parquet Format:**
   - Optional format, enabled with `--use-parquet`. Stored in `data/twibot22_balanced_parquet/` and `data/twibot22_balanced_tokenized_parquet/`.
   - Consists of subfolders for each split containing one or more `.parquet` files.

### 4.2.1. Format Comparison

| Dataset | Hugging Face Size (MB) | Parquet Size (MB) | Compression Ratio |
|---------|------------------------|-------------------|-------------------|
| Processed Dataset | 0.75 | 0.11 | 7.11x |
| Tokenized Dataset | 0.34 | 0.20 | 1.69x |

| Operation | Hugging Face (seconds) | Parquet (seconds) | Speedup |
|-----------|------------------------|-------------------|---------|
| Loading | 0.02 | 9.32 | 0.00x |
| Filtering | 0.02 | 0.01 | 3.34x |
| Mapping | 0.03 | 0.02 | 1.40x |

**Storage Efficiency**: Parquet format provides significant storage savings compared to the Hugging Face disk format, with the processed dataset being 7.11x smaller and the tokenized dataset being 1.69x smaller.

**Performance Trade-offs**:
- **Loading**: Hugging Face format loads significantly faster than Parquet.
- **Filtering**: Parquet format filters faster than Hugging Face.
- **Mapping**: Parquet format maps faster than Hugging Face.

**When to Use Each Format**:
- **Hugging Face Disk Format**: Recommended when loading speed is the priority and disk space is not a concern.
- **Apache Parquet Format**: Recommended when storage efficiency is important or when working with larger datasets and filtering/mapping operations are frequent.

### 4.3. Data Preparation Steps
*(Executed by `scripts/1_extract_tweets.py`)*
- Raw tweet data is loaded from the JSON files.
- User metadata is extracted from label.csv and split.csv.
- Exactly 500 bot tweets and 500 human tweets are extracted.
- Tweets are cleaned: URLs removed, extra whitespace normalized.
- Data is split into train (80%), validation (10%), and test (10%) sets.
- The processed data is saved in both Hugging Face and Parquet formats.
- **Final Dataset Statistics:**
  - Train: 800 samples (50% bots, 50% humans)
  - Validation: 100 samples (49% bots, 51% humans)
  - Test: 100 samples (50% bots, 50% humans)
  - Average tokens per sample: ~38 (train), ~38 (validation), ~34 (test)

### 4.4. Data Structures
- **Raw Data:** JSON files containing tweet data and CSV files for labels and splits.
- **Processed/Tokenized Data (in memory):** `datasets.DatasetDict`. This object holds multiple `datasets.Dataset` instances (one per split: train, validation, test).
- **`datasets.Dataset` Structure:** Key columns generated by the pipeline:
  - `user_id` (`string`): User identifier.
  - `text` (`string`): The tweet text.
  - `tweet_count` (`int64`): Number of tweets per user.
  - `label` (`ClassLabel(names=['human', 'bot'])`): Integer label (0 or 1).
  - `input_ids` (`Sequence(int32)`): *(Added after tokenization)* List of token IDs.
  - `attention_mask` (`Sequence(int8)`): *(Added after tokenization)* Mask indicating real tokens vs padding.

## 5. Methodology

### 5.1. Preprocessing - Tokenization
*(Executed by `scripts/2_tokenize_balanced_dataset.py`)*
- **Tokenizer:** `distilbert-base-uncased` from Hugging Face Transformers.
- **Process:** The `text` column of the processed dataset is tokenized.
- **Parameters:**
  - `truncation=True`: Sequences longer than the model's maximum input length (512 tokens for DistilBERT) are truncated.
  - `padding=False`: Padding is applied dynamically per batch during training.
- **Output:** Adds `input_ids` and `attention_mask` columns to the dataset.
- **Statistics:**
  - Average tokens per sample: ~38 (train), ~38 (validation), ~34 (test)
  - Maximum tokens in a sample: 228 (train), 216 (validation), 204 (test)
  - Samples exceeding max length: 0%
  - Samples with only special tokens: 0.5% (train), 0% (validation), 0% (test)

### 5.2. Model Architecture
- **Base Model:** `distilbert-base-uncased`. A smaller, faster version of BERT.
- **Task Adaptation:** Fine-tuned for sequence classification using `AutoModelForSequenceClassification`.
- **Configuration:** `num_labels=2`, `id2label={0: "human", 1: "bot"}`, `label2id={"human": 0, "bot": 1}`.

### 5.3. Training Process
*(Executed by `scripts/3_train_model.py`)*
- **Framework:** Hugging Face `Trainer` API with memory efficiency optimizations.
- **Optimizer:** AdamW (default).
- **Key Hyperparameters:**
  - Learning Rate: 5e-5
  - Batch Size: 16 per device
  - Epochs: 3
  - Weight Decay: 0.01
  - Gradient Accumulation Steps: 2
- **Evaluation:** Performed on the validation set after each epoch. Metrics: Accuracy, Precision, Recall, F1-Score.
- **Best Model Selection:** Based on the highest F1-score achieved on the validation set.
- **Early Stopping:** Training stops if the validation F1-score does not improve for 2 consecutive epochs.
- **Hardware Acceleration:** Automatically uses MPS (Apple Silicon) or CUDA (NVIDIA GPU) if available.

## 6. Results

- **Final Evaluation (Test Set):**

  | Metric          | Score  |
  |-----------------|--------|
  | Test Accuracy   | 0.79   |
  | Test Precision  | 0.80   |
  | Test Recall     | 0.79   |
  | Test F1-Score   | 0.79   |
  | Test Loss       | 0.40   |

- **Training Performance Trend:**
  - Peak validation F1-score of 0.86 was achieved at the end of epoch 3.
  - Training loss decreased steadily from 0.48 to 0.33 over 3 epochs.
  - Validation accuracy improved from 0.73 in epoch 1 to 0.86 in epoch 3.
- **Training Curves:** Visualizations of training/validation loss and metrics over epochs can be found in `models/bot_detection_model/training_curves.png`.

## 7. Discussion & Conclusion

### 7.1. Interpretation
- The fine-tuned DistilBERT model achieved a good F1-score of 0.79 and accuracy of 0.79 on the test set.
- The balanced dataset approach proved effective, with the model learning meaningful patterns to distinguish between bot and human tweets.
- The model shows strong performance on the validation set (0.86 F1-score) but slightly lower performance on the test set, indicating some potential overfitting.
- The model correctly identifies many tweets with personal experiences as human tweets, but sometimes misclassifies tweets with stock symbols ($TSLA) as human tweets.

### 7.2. Limitations
- **Data Scope:** The model is trained on a small, balanced subset of tweets, which may not capture the full diversity of bot and human behavior.
- **Feature Limitation:** The model relies solely on tweet text, without considering user metadata, behavioral patterns, or network information.
- **Domain Specificity:** The model may be biased toward the specific types of bots and humans represented in the Twibot-22 dataset.

### 7.3. Conclusion
- We successfully trained a DistilBERT model for Twitter bot detection using a balanced subset of the Twibot-22 dataset.
- The balanced approach led to good performance metrics (79% accuracy and F1-score on the test set).
- The proper train/validation/test split approach allowed for better model selection and evaluation, similar to the main project.
- This extension demonstrates that even with a relatively small, balanced dataset of 1000 tweets, transformer-based models can effectively distinguish between bot and human tweets.
- The integration of Apache Parquet provides flexibility for handling larger datasets or integration with other tools.

### 7.4. Feature Engineering Opportunities

While our current model achieves good performance using only tweet text, there are numerous opportunities for feature engineering that could significantly improve detection accuracy. The Twibot-22 dataset contains rich metadata that remains unexploited in our current approach.

#### 7.4.1. Hashtag Analysis

The `hashtag.json` file (255MB) contains valuable information about hashtag usage that could reveal bot behavior patterns:

- **Usage Frequency**: Bots often use hashtags more frequently than humans
- **Hashtag Categories**: Bots tend to use promotional, cryptocurrency, or trending hashtags
- **Temporal Patterns**: Bots may rapidly adopt trending hashtags or continue using outdated ones
- **Semantic Coherence**: Bots often use semantically unrelated hashtags together
- **Rare/Common Distribution**: Bots might use extremely rare hashtags for targeted campaigns

Implementation could involve creating features like hashtag density per tweet, unique hashtag ratio, or distribution across categories.

#### 7.4.2. Network Analysis

The `edge.csv` file (6.2GB) contains follower/following relationships that could reveal network-based signals:

- **Follow Ratio**: Bots often have unusual follower-to-following ratios
- **Network Centrality**: Bot accounts may have different centrality metrics in the social graph
- **Clustering Patterns**: Bots may form clusters or islands in the network
- **Connection to Known Bots**: Accounts connected to many known bots are more likely to be bots
- **Temporal Following Patterns**: Bots may gain followers in unusual patterns

Graph-based features could be extracted using network analysis libraries like NetworkX.

#### 7.4.3. User Metadata

User profile information could provide strong signals:

- **Account Age**: Many bots are relatively new accounts
- **Profile Completeness**: Bots often have incomplete profiles
- **Username Patterns**: Bots may have usernames with specific patterns (random strings, numbers)
- **Profile Picture Analysis**: Bots may lack profile pictures or use stock images
- **Description Keywords**: Certain keywords in descriptions may indicate bot accounts

#### 7.4.4. Behavioral Patterns

Temporal and behavioral features could be extracted:

- **Posting Frequency**: Bots often post at regular intervals or with unusual frequency
- **Activity Cycles**: Bots may show unnatural activity patterns across hours/days
- **Content Similarity**: Bots often post similar content repeatedly
- **Response Patterns**: Bots may have unusual patterns in how they interact with other users
- **Platform Usage**: Bots may use specific Twitter clients or APIs

#### 7.4.5. List Membership

The `list.json` file (4.7MB) contains information about Twitter lists:

- **List Membership**: Being on certain types of lists may correlate with bot status
- **List Creation**: Bots may create lists with specific characteristics
- **List Names/Descriptions**: The nature of lists an account belongs to may provide signals

#### 7.4.6. Multimodal Features

Combining different types of features could be particularly powerful:

- **Text + Network**: Combining content analysis with network position
- **Temporal + Content**: Analyzing how content changes over time
- **Metadata + Behavior**: Correlating profile information with behavioral patterns

#### 7.4.7. Implementation Considerations

When implementing these features, several considerations are important:

- **Memory Efficiency**: Large files like `edge.csv` (6.2GB) require chunked processing
- **Feature Selection**: Not all features will be equally valuable; feature selection methods should be applied
- **Interpretability**: Some features provide more interpretable signals than others
- **Generalization**: Features should capture fundamental bot behaviors rather than dataset-specific patterns

By incorporating these additional features, the model could potentially achieve significantly higher accuracy while maintaining good generalization to new data.

## 8. Usage Instructions

### 8.1. Prerequisites
1. Ensure you have the Twibot-22 dataset files (`tweet_*.json`, `label.csv`, `split.csv`) in the `Twibot-22/` directory.
2. Install required Python packages:
   ```bash
   pip install transformers datasets torch scikit-learn matplotlib pandas pyarrow psutil
   ```

### 8.2. Running the Pipeline (Default - HF Format)
Execute the scripts sequentially:
```bash
# Step 1: Extract tweets from the Twibot-22 dataset
python Twibot-22/scripts/1_extract_tweets.py --data-dir Twibot-22 --bot-tweets 500 --human-tweets 500 --output-dir ./extracted_1000_tweets

# Step 2: Prepare the dataset with train/validation/test splits
python Twibot-22/scripts/prepare_dataset.py --input-dir ./extracted_1000_tweets

# Step 3: Tokenize the balanced dataset
python Twibot-22/scripts/2_tokenize_balanced_dataset.py

# Step 4: Train the model
python Twibot-22/scripts/3_train_model.py

# Step 5: Make predictions with the trained model
python Twibot-22/scripts/4_predict.py # For interactive prediction
```

This will generate datasets in `data/twibot22_balanced_dataset/` and `data/twibot22_balanced_tokenized/`, and the trained model in `models/bot_detection_model/`.

### 8.3. Running the Pipeline (Optional - Parquet Format)
Use the `--use-parquet` flag for the tokenization and training steps:

```bash
# Step 1: Extract tweets from the Twibot-22 dataset
python Twibot-22/scripts/1_extract_tweets.py --data-dir Twibot-22 --bot-tweets 500 --human-tweets 500 --output-dir ./extracted_1000_tweets

# Step 2: Prepare the dataset with train/validation/test splits
python Twibot-22/scripts/prepare_dataset.py --input-dir ./extracted_1000_tweets

# Step 3: Tokenize the balanced dataset using Parquet format
python Twibot-22/scripts/2_tokenize_balanced_dataset.py --use-parquet

# Step 4: Train the model using Parquet format
python Twibot-22/scripts/3_train_model.py --use-parquet

# Step 5: Make predictions with the trained model
python Twibot-22/scripts/4_predict.py # Prediction script uses the saved model regardless of training data format
```

This will generate datasets in `data/twibot22_balanced_parquet/` and `data/twibot22_balanced_tokenized_parquet/`.

### 8.4. Benchmarking Parquet vs Hugging Face (Optional)
To compare the performance of Hugging Face and Parquet formats:

```bash
# Run the benchmark script
python Twibot-22/scripts/benchmark_parquet.py

# View the results
cat Twibot-22/benchmark_results/parquet_performance.md
```

This will generate performance metrics and charts in the `benchmark_results/` directory.

## 9. API and Module Documentation

### 9.1. `scripts/1_extract_tweets.py`
- **Functionality:** Extracts exactly 500 bot tweets and 500 human tweets from the Twibot-22 dataset.
- **Key Functions:** `load_user_metadata`, `process_json_object`, `worker_process_chunk`, `process_tweet_file`, `save_dataset_to_files`.
- **Input:** Path to the Twibot-22 directory containing tweet JSON files, label.csv, and split.csv.
- **Output:** Extracted tweets saved to text files (tweets.txt, labels.txt) and CSV (dataset.csv).

### 9.2. `scripts/prepare_dataset.py`
- **Functionality:** Prepares the extracted dataset with train/validation/test splits (80/10/10).
- **Key Functions:** `load_dataset_from_files`, `create_dataset_splits`, `main`.
- **Input:** Directory containing the extracted tweets and labels.
- **Output:** Processed dataset saved in Hugging Face format.

### 9.3. `scripts/2_tokenize_balanced_dataset.py`
- **Functionality:** Tokenizes the balanced dataset using the DistilBERT tokenizer.
- **Key Functions:** `preprocess_function`, `main`.
- **Input:** Path to the balanced dataset (HF or Parquet format).
- **Output:** Tokenized dataset saved in the specified format.

### 9.4. `scripts/3_train_model.py`
- **Functionality:** Trains a DistilBERT model on the tokenized balanced dataset.
- **Key Functions:** `compute_metrics`, `memory_monitor`, `MemoryEfficientTrainer`, `main`.
- **Input:** Path to the tokenized dataset (HF or Parquet format).
- **Output:** Trained model, evaluation results, and training curves.

### 9.5. `scripts/4_predict.py`
- **Functionality:** Uses the trained model to make predictions on new text inputs.
- **Key Functions:** `predict_bot_probability`, `main`.
- **Input:** Text to classify (via interactive prompt).
- **Output:** Prediction (human/bot) and confidence score.

### 9.6. `scripts/benchmark_parquet.py`
- **Functionality:** Benchmarks the performance of Hugging Face vs Parquet formats.
- **Key Functions:** `benchmark_loading`, `benchmark_filtering`, `benchmark_mapping`, `plot_results`.
- **Input:** Paths to datasets in both formats.
- **Output:** Performance metrics, charts, and a detailed markdown report.

### 9.7. `utilities/parquet_utils.py`
- **Functionality:** Utilities for working with Apache Parquet format.
- **Key Functions:** `print_memory_usage`, `force_garbage_collection`, `save_dataset_to_parquet`, `load_parquet_as_dataset`.
- **Usage:** Used by the pipeline scripts to save and load datasets in Parquet format.

## 10. Data Processing Workflow Details

This section details the sequence of operations transforming raw Twitter data into model-ready input, as implemented in the scripts, with specific function names and their operations.

### 10.1. Raw Data Loading

**Script:** `scripts/1_extract_tweets.py`

**Input:**
- `tweet_*.json`: Raw tweet data files
- `label.csv`: User labels (bot/human)
- `split.csv`: Dataset splits (train/test/dev)

**Key Functions:**
- `load_user_metadata(label_file, split_file)`: Parses CSV files to create dictionaries mapping user IDs to labels and splits
- `process_json_object(obj, user_metadata)`: Processes a single JSON object from the tweet files, extracting relevant data if the user is in our target set
- `worker_process_chunk(chunk, user_metadata, results_queue)`: Processes a chunk of JSON data in a worker process, handling JSON parsing errors gracefully
- `process_tweet_file(tweet_file, user_metadata, num_processes)`: Orchestrates multiprocessing to extract tweets from large JSON files using multiple CPU cores

**Process Flow:**
1. `load_user_metadata()` creates a dictionary of user IDs with their labels (0=human, 1=bot)
2. `process_tweet_file()` divides the large JSON files into chunks for parallel processing
3. Each worker process (`worker_process_chunk()`) extracts tweets from users in our target set
4. Results are collected through a queue system to avoid memory issues

### 10.2. Tweet Extraction and Cleaning

**Script:** `scripts/1_extract_tweets.py`

**Input:**
- Parsed tweet data and user metadata

**Key Functions:**
- `clean_text(text)`: Removes URLs, special characters, and normalizes whitespace using regex patterns
- `extract_tweet_text(tweet_obj)`: Extracts the text content from a tweet object
- `save_dataset_to_files(tweets, labels, output_dir)`: Saves the extracted tweets and labels to text files
- `monitor_extraction_progress(bot_count, human_count, bot_target, human_target)`: Tracks progress toward balanced dataset targets

**Process Flow:**
1. For each tweet, `extract_tweet_text()` retrieves the raw text
2. `clean_text()` applies regex patterns like `re.sub(r'http\S+', '', text)` to remove URLs
3. Tweets are filtered to maintain balance (500 bot, 500 human) using counters
4. `save_dataset_to_files()` writes the final balanced dataset to disk as text files

### 10.3. Dataset Creation and Formatting

**Script:** `scripts/prepare_dataset.py`

**Input:**
- Extracted and cleaned tweets with labels

**Key Functions:**
- `load_dataset_from_files(input_dir)`: Reads the text files containing tweets and labels
- `create_dataset_dict(train_df, validation_df, test_df)`: Creates a Hugging Face DatasetDict with proper feature typing
- `Dataset.from_pandas(df, features=features)`: Converts pandas DataFrames to Hugging Face Datasets
- `Features({...})`: Defines the schema with explicit typing for each column

**Process Flow:**
1. `load_dataset_from_files()` reads tweets.txt and labels.txt into memory
2. Data is loaded into pandas DataFrames with columns: 'text', 'label', 'user_id', 'tweet_count'
3. Features are explicitly typed using `Features({'user_id': Value('string'), 'text': Value('string'), ...})`
4. `Dataset.from_pandas()` converts each DataFrame to a Hugging Face Dataset with the defined schema

### 10.4. Dataset Splitting

**Script:** `scripts/prepare_dataset.py`

**Input:**
- The initial Dataset created in step 10.3

**Key Functions:**
- `random.shuffle(indices)`: Randomizes the order of dataset indices
- `df.iloc[indices]`: Selects rows from the DataFrame based on shuffled indices
- `DatasetDict({'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset})`: Creates a dictionary of datasets for each split

**Process Flow:**
1. All indices are shuffled using `random.shuffle()` with a fixed seed (42) for reproducibility
2. The dataset is split into train (80%), validation (10%), and test (10%) portions
3. Class balance is verified by counting labels in each split
4. The splits are combined into a `DatasetDict` with keys 'train', 'validation', and 'test'

### 10.5. Tokenization for Model Input

**Script:** `scripts/2_tokenize_balanced_dataset.py`

**Input:**
- The processed DatasetDict (from HF disk or Parquet)

**Key Functions:**
- `load_from_disk(dataset_path)` or `load_parquet_as_dataset(dataset_path)`: Loads the dataset from disk
- `AutoTokenizer.from_pretrained("distilbert-base-uncased")`: Initializes the DistilBERT tokenizer
- `preprocess_function(examples)`: Applies tokenization to batches of examples
- `dataset.map(preprocess_function, batched=True)`: Applies the tokenization function to the entire dataset
- `dataset.save_to_disk(output_dir)` or `save_dataset_to_parquet(dataset, output_dir)`: Saves the tokenized dataset

**Process Flow:**
1. The tokenizer is initialized with `AutoTokenizer.from_pretrained()`
2. `preprocess_function()` applies the tokenizer to batches of examples with parameters:
   - `truncation=True`: Truncates sequences longer than 512 tokens
   - `padding=False`: No padding is applied (done dynamically during training)
3. `dataset.map()` applies this function to all examples in batches for efficiency
4. The resulting dataset with 'input_ids' and 'attention_mask' columns is saved to disk

### 10.6. Data Format Conversion (Optional)

**Scripts:**
- All pipeline scripts with `--use-parquet` flag
- `scripts/benchmark_parquet.py` for performance comparison

**Key Functions:**
- `save_dataset_to_parquet(dataset_dict, output_dir)`: Converts and saves a DatasetDict to Parquet format
- `load_parquet_as_dataset(input_dir)`: Loads a DatasetDict from Parquet files
- `benchmark_loading()`, `benchmark_filtering()`, `benchmark_mapping()`: Compare performance between formats
- `plot_results()`: Generates visualizations of performance metrics

**Process Flow:**
1. `save_dataset_to_parquet()` iterates through each split in the DatasetDict
2. For each split, it converts the Dataset to a pandas DataFrame
3. The DataFrame is saved as a Parquet file using `df.to_parquet()`
4. Metadata is preserved in JSON files to maintain compatibility with the Hugging Face ecosystem

### 10.7. Data Flow Summary

Raw JSON/CSV → Extracted Tweets → Cleaned Text → Balanced Dataset → Split Dataset (train/val/test) → Tokenized Dataset → Model Input Tensors

A parallel path using Parquet for intermediate storage is available at each step after the initial extraction, implemented through the `--use-parquet` flag and the utility functions in `utilities/parquet_utils.py`.
