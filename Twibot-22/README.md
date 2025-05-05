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
- **Functionality**: extracts exactly 500 bot tweets and 500 human tweets from the Twibot-22 dataset.
- **Input**: path to the Twibot-22 directory containing tweet JSON files, `label.csv`, and `split.csv`.
- **Output**: extracted tweets saved to text files (`tweets.txt`, `labels.txt`) and CSV (`dataset.csv`).

- #### `print_memory_usage()`

    - **Description**: rints the current memory usage of the script and the system.
    
    - **Arguments**: _None_
    
    - **Returns**: `system_percent` (float): The percentage of system memory being used.
    
    - **Details**: it uses the `psutil` library to retrieve and display memory stats.

- #### `force_garbage_collection()`

    - **Description**: forces garbage collection to free up memory and prints memory usage statistics.
    
    - **Arguments**: _None_
    
    - **Returns**: _None_
    
    - **Details**: it uses Python's `gc.collect()` to trigger garbage collection and calls `print_memory_usage()`.



- #### `load_user_metadata(data_dir)`

    - **Description**: loads user metadata from a CSV file to create mappings of user IDs and labels.
    
    - **Arguments**: `data_dir` (str): The directory where the `label.csv` file is located.
    
    - **Returns**:  
        - `user_to_label` (dict): Maps user IDs to labels (1 for bot, 0 for human).  
        - `user_id_mapping` (dict): Maps numeric user IDs to IDs with the `'u'` prefix.
    
    - **Details**: Processes the CSV file to build two dictionaries for user label lookups.


- #### `process_json_object(obj, user_id_mapping, user_to_label)`

    - **Description**: Processes a single JSON tweet object to extract cleaned text and its corresponding label.
    
    - **Arguments**:  
        - `obj` (dict or str): JSON object, either as a string or a parsed dictionary.  
        - `user_id_mapping` (dict): Maps numeric user IDs to `'u'`-prefixed IDs.  
        - `user_to_label` (dict): Maps user IDs to labels.
    
    - **Returns**: `(text, label)` (tuple) if valid, `None` otherwise.

    - **Details**: cleans the tweet text and retrieves the matching label if available.


- #### `worker_process_chunk(chunk_data, user_id_mapping, user_to_label, bot_count, human_count, bot_target, human_target)`

    - **Description**: processes a chunk of tweet data to collect valid bot and human tweets.
    
    - **Arguments**:  
        - `chunk_data` (str): A block of raw tweet data.  
        - `user_id_mapping` (dict)  
        - `user_to_label` (dict)  
        - `bot_count` (int): Current bot tweet count.  
        - `human_count` (int): Current human tweet count.  
        - `bot_target` (int): Total target for bot tweets.  
        - `human_target` (int): Total target for human tweets.
    
    - **Returns**:  
    `dict` with keys:  
        - `'bot_tweets'` (list)  
        - `'human_tweets'` (list)  
        - `'bot_count'` (int)  
        - `'human_count'` (int)
    
    - **Details**: filters tweets and updates the count, useful in multiprocessing.



- #### `process_tweet_file(file_path, user_id_mapping, user_to_label, bot_tweets, human_tweets, bot_target, human_target, num_processes=None)`

    - **Description**: processes a tweet file to extract a target number of bot and human tweets.

    - **Arguments**:  
        - `file_path` (str): Path to the tweet JSON file.  
        - `user_id_mapping` (dict)  
        - `user_to_label` (dict)  
        - `bot_tweets` (list): Output container for bot tweets.  
        - `human_tweets` (list): Output container for human tweets.  
        - `bot_target` (int)  
        - `human_target` (int)  
        - `num_processes` (int, optional): Defaults to half of available CPU cores.

    - **Returns**: `True` if targets are met, `False` otherwise.
    
    - **Details**: it uses multiprocessing to speed up extraction from large tweet files.



- #### `save_dataset_to_files(bot_tweets, human_tweets, output_dir)`

    - **Description**: it saves the extracted tweets and their labels to `.txt` and `.csv` files.
    
    - **Arguments**:  
        - `bot_tweets` (list)  
        - `human_tweets` (list)  
        - `output_dir` (str): Output folder path.

    - **Returns**: _None_
    
    - **Details**:  
        - Creates three files: `tweets.txt`, `labels.txt`, and `dataset.csv`.  
        - Handles newline escaping and formatting for CSV compatibility.


- #### `main()`

    - **Description**: main function that orchestrates tweet extraction and dataset saving.

    - **Arguments**: _None_
    
    - **Returns**: _None_

    - **Details**:  
        - Parses CLI arguments, loads metadata, processes tweets, and saves output.  
        - Calls all other functions in the correct order.


### 9.2. `scripts/prepare_dataset.py`
- **Functionality**: prepares the extracted dataset with train/validation/test splits (80/10/10).
- **Key functions**: `main`
- **Input**: directory containing the extracted tweets and labels.
- **Output**: processed dataset saved in Hugging Face and Parquet formats.

- #### `main()`

    - **Description**: Prepares the dataset of labeled tweets for training and evaluation by loading, splitting, and saving the data.
    
    - **Arguments** (via CLI with `argparse`):
      - `--input-dir` (str, default=`"./extracted_1000_tweets"`): Path to the directory containing `tweets.txt` and `labels.txt`.
      - `--output-dir` (str, optional): Output directory to save the Hugging Face format dataset. Defaults to `"data/twibot22_balanced_dataset"` if not specified.
      - `--parquet-dir` (str, optional): Output directory to save the Apache Parquet format dataset. Defaults to `"data/twibot22_balanced_parquet"` if not specified.
      - `--test-split` (float, default=`0.1`): Fraction of the dataset used for testing.
      - `--validation-split` (float, default=`0.1`): Fraction of the dataset used for validation.
      - `--seed` (int, default=`42`): Random seed to ensure reproducibility of the dataset split.
    
    - **Returns**: _None_
    
    - **Details**:
      - Loads tweets and labels from `tweets.txt` and `labels.txt`.
      - Creates dummy user IDs (e.g., `user_0`, `user_1`, ...).
      - Shuffles the dataset and performs splitting using manual index slicing.
      - Constructs a `datasets.DatasetDict` object with the splits: `train`, `validation`, and `test`.
      - Adds a `tweet_count` field with constant value 1.
      - Saves the final dataset:
        - In Hugging Face format using `.save_to_disk(output_dir)`.
        - In Parquet format using `save_dataset_to_parquet()`.


### 9.3. `scripts/2_tokenize_balanced_dataset.py`
- **Functionality**: tokenizes the balanced dataset using the DistilBERT tokenizer.
- **Key functions**: `main`, `preprocess_function`
- **Input**: path to the balanced dataset (Hugging Face or Parquet format).
- **Output**: tokenized dataset saved in the specified format and tokenization statistics saved to `tokenization_info.json`.

- #### `main()`

    - **Description**: Main control function that handles dataset loading, tokenization, statistics computation, and saving the tokenized dataset.
    
    - **Arguments** (via `argparse`):
      - `--use-parquet` (bool): If set, loads the dataset from Apache Parquet format. Otherwise, loads from Hugging Face format.
    
    - **Returns**: _None_
    
    - **Details**:
      - Detects dataset format and loads it from either Hugging Face disk format or Apache Parquet using `load_from_disk()` or `load_parquet_as_dataset()`.
      - Loads the `distilbert-base-uncased` tokenizer via `AutoTokenizer`.
      - Applies the tokenizer using `dataset.map()` with `batched=True`.
      - Computes key statistics:
        - Average token length
        - Maximum token length
        - Percentage of truncated or empty samples
      - Saves the tokenized dataset to disk in the same format it was loaded from.
      - Stores tokenization metadata in a JSON file named `tokenization_info.json`.

- #### `preprocess_function(examples)`

    - **Description**: Tokenizes a batch of tweet texts using the DistilBERT tokenizer.
    
    - **Arguments**:
      - `examples` (_dict_): Dictionary of features containing the `"text"` key with a list of strings.
    
    - **Returns**: _dict_  
      - `input_ids`: List of token ID sequences.  
      - `attention_mask`: List of attention masks (1 = token, 0 = padding).
    
    - **Details**:
      - Uses `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
      - Tokenizes text with:
        - `truncation=True` (to 512 tokens)
        - `padding=False` (assumes dynamic padding during training)
      - Designed for use with `dataset.map()` and `batched=True` for efficiency.

### 9.4. `scripts/3_train_model.py`
- **Functionality**: trains a DistilBERT model on the tokenized balanced dataset.  
- **Key functions**: `compute_metrics`, `memory_monitor`, `MemoryEfficientTrainer`, `main`.  
- **Input**: path to the tokenized dataset (Hugging Face or Parquet format).  
- **Output**: trained model, evaluation results, and training curves.  

- #### `compute_metrics(pred)`
    - **description**: computes evaluation metrics (accuracy, precision, recall, f1-score) from model predictions  
    - **arguments**:  
      - `pred`: prediction output object containing `.label_ids` and `.predictions`  
    - **returns**:  
      - `dict`:  
        - `accuracy`: overall classification accuracy  
        - `precision`: weighted precision score  
        - `recall`: weighted recall score  
        - `f1`: weighted f1-score  
    - **details**:  
      - applies `argmax` over logits to derive predicted labels  
      - metrics computed using `sklearn.metrics`  

- #### `memory_monitor(threshold_percent=80, force=False)`
    - **description**: monitors system and process memory; triggers garbage collection and clears GPU memory if usage exceeds the threshold  
    - **arguments**:  
      - `threshold_percent` (`int`): maximum allowed memory percentage before triggering GC  
      - `force` (`bool`): if true, triggers GC regardless of current memory usage  
    - **returns**:  
      - `tuple`:  
        - `memory_mb`: memory used by the process (in MB)  
        - `memory_percent`: system-wide memory usage (in %)  
    - **details**:  
      - used before and after training/evaluation steps to prevent memory spikes  

- #### `class MemoryEfficientTrainer(Trainer)`
    - **description**: subclass of `transformers.Trainer` with memory monitoring before training, evaluation, and model saving  
    - **methods**:  
      - `__init__(self, memory_threshold, *args, **kwargs)`: adds a memory threshold to the standard trainer  
      - `training_step(self, model, inputs, num_items_in_batch=None)`: checks memory usage before training step  
      - `evaluate(self, *args, **kwargs)`: performs evaluation with GC check  
      - `save_model(self, *args, **kwargs)`: triggers GC before saving model  
    - **properties**:  
      - `memory_threshold` (`int`): maximum memory % before triggering garbage collection  
    - **details**:  
      - helps train on systems with limited RAM or VRAM (e.g. Apple M1/M2)  

- #### `main()`
    - **description**: main function to train a DistilBERT-based bot classifier on the balanced Twibot-22 dataset  
    - **arguments** (parsed via `argparse`):  
      - `--use-parquet`: use parquet format instead of Hugging Face format  
      - `--batch-size`: training and evaluation batch size  
      - `--learning-rate`: learning rate for the optimizer  
      - `--epochs`: number of training epochs  
      - `--weight-decay`: l2 regularization factor  
      - `--gradient-accumulation`: number of gradient accumulation steps  
      - `--fp16`: enable mixed-precision training  
      - `--memory-threshold`: memory usage % that triggers GC  
    - **returns**: _None_  
    - **details**:  
      - loads datasetdict with train/validation/test splits  
      - fine-tunes `distilbert-base-uncased` using `AutoModelForSequenceClassification`  
      - uses binary classification: `0 = human`, `1 = bot`  
      - evaluation metrics computed using `compute_metrics()`  
      - saves model to `models/bot_detection_model/best_model/`  
      - saves metrics to `test_results.json`  
      - saves training curves to `training_curves.png`  

### 9.5. `scripts/4_predict.py`
- **Functionality**: uses the trained model to make predictions on new text inputs.  
- **Key functions**: `predict_bot_probability`, `main`.  
- **Input**: text to classify (via interactive prompt).  
- **Output**: prediction (human or bot) and confidence score. 

- #### `predict_bot_probability(text, model, tokenizer, device)`
    - **description**: predicts whether a given text was written by a bot or a human using a fine-tuned distilbert model  
    - **arguments**:  
      - `text` (`str`): the input text to classify  
      - `model` (`AutoModelForSequenceClassification`): the trained classification model  
      - `tokenizer` (`PreTrainedTokenizer`): tokenizer used for encoding the input  
      - `device` (`torch.device`): device used for inference (`cpu`, `cuda`, or `mps`)  
    - **returns**:  
      - `tuple`:  
        - `prediction` (`str`): "human" or "bot"  
        - `probability` (`float`): confidence score of the predicted class  
    - **details**:  
      - tokenizes the input text  
      - performs inference with `torch.no_grad()`  
      - uses softmax to convert logits into probabilities  
      - selects the label with the highest probability via `argmax`  

- #### `main()`
    - **description**: loads the trained model, predicts on example tweets, and starts an interactive prediction prompt  
    - **arguments**: _None_  
    - **returns**: _None_  
    - **details**:  
      - loads the model from `models/bot_detection_model/best_model/`  
      - loads the `distilbert-base-uncased` tokenizer  
      - automatically selects device (`cuda`, `mps`, or `cpu`)  
      - makes predictions on 5 hardcoded sample tweets  
      - launches an interactive prompt for live predictions  
      - exits when user types `"q"` or `"quit"`  

### 9.6. `scripts/benchmark_parquet.py`
- **Functionality:** benchmarks the performance of hugging face vs parquet formats.
- **Key functions:** `benchmark_loading`, `benchmark_filtering`, `benchmark_mapping`, `plot_results`.
- **Input:** paths to datasets in both formats.
- **Output:** performance metrics, charts, and a detailed markdown report.

- #### `print_memory_usage()`
    - **description**: prints and returns the current memory usage of the system and the running process  
    - **arguments**: _none_  
    - **returns**:  
      - `memory_mb` (`float`): memory usage of the current process in MB  
    - **details**:  
      - uses `psutil` to retrieve system and process memory stats  
      - helpful for monitoring before and after data operations  

- #### `force_garbage_collection()`
    - **description**: forces python garbage collection and prints updated memory usage  
    - **arguments**: _none_  
    - **returns**: _none_  
    - **details**:  
      - uses `gc.collect()`  
      - calls `print_memory_usage()` to show reclaimed memory  

- #### `get_directory_size(path)`
    - **description**: calculates the total size of a directory (recursively) in megabytes  
    - **arguments**:  
      - `path` (`str`): path to the directory  
    - **returns**:  
      - `size_mb` (`float`): total directory size in MB  
    - **details**:  
      - uses `os.walk()` to sum the size of all files  
      - useful to compare disk usage of hf vs parquet formats  

- #### `benchmark_loading(hf_path, parquet_path)`
    - **description**: benchmarks the time and memory used to load datasets from hugging face and parquet formats  
    - **arguments**:  
      - `hf_path` (`str`): path to hugging face dataset  
      - `parquet_path` (`str`): path to parquet dataset  
    - **returns**: `dict` with keys:  
      - `hf_loading_time`, `parquet_loading_time`  
    - **details**:  
      - loads using `load_from_disk()` and `load_parquet_as_dataset()`  
      - measures load time and memory change  

- #### `benchmark_filtering(hf_path, parquet_path)`
    - **description**: benchmarks `.filter()` performance for both formats  
    - **arguments**:  
      - `hf_path` (`str`)  
      - `parquet_path` (`str`)  
    - **returns**: `dict` with keys:  
      - `hf_filtering_time`, `parquet_filtering_time`  
    - **details**:  
      - applies filter `lambda x: x['label'] == 1`  
      - compares duration and output size  

- #### `benchmark_mapping(hf_path, parquet_path)`
    - **description**: benchmarks `.map()` performance by adding a computed column  
    - **arguments**:  
      - `hf_path` (`str`)  
      - `parquet_path` (`str`)  
    - **returns**: `dict` with keys:  
      - `hf_mapping_time`, `parquet_mapping_time`  
    - **details**:  
      - adds a `text_length` column using `lambda x: len(x["text"])`  

- #### `plot_results(results, output_dir)`
    - **description**: visualizes and saves benchmark charts and markdown reports  
    - **arguments**:  
      - `results` (`dict`): contains all benchmark metrics  
      - `output_dir` (`str`): directory to save plots and report  
    - **returns**: _none_  
    - **details**:  
      - saves `storage_comparison.png`, `performance_comparison.png`  
      - generates `parquet_performance.md` with tables and figures  

- #### `main()`
    - **description**: entry point that runs all benchmarks and saves results  
    - **arguments**: _none_ (uses `argparse` for cli inputs)  
    - **returns**: _none_  
    - **details**:  
      - expects paths via CLI: `--hf-processed`, `--parquet-processed`, `--hf-tokenized`, `--parquet-tokenized`, `--output-dir`  
      - calls:  
        - `get_directory_size()`  
        - `benchmark_loading()`  
        - `benchmark_filtering()`  
        - `benchmark_mapping()`  
        - `plot_results()`  

### 9.7. `utilities/parquet_utils.py`
- **Functionality:** utilities for working with apache parquet format.
- **Key functions:** `print_memory_usage`, `force_garbage_collection`, `save_dataset_to_parquet`, `load_parquet_as_dataset`.
- **Usage:** used by the pipeline scripts to save and load datasets in parquet format.

- #### `print_memory_usage()`
    - **description**: prints current memory usage of the process and available system memory  
    - **arguments**: _none_  
    - **returns**:  
      - `memory_mb` (`float`): current process memory in mb  
      - `system_percent` (`float`): system-wide memory usage percentage  
    - **details**: uses `psutil` to fetch memory statistics  

- #### `wait_for_memory(target_percent=75, max_wait=10, check_interval=1)`
    - **description**: waits until system memory usage drops below a target threshold  
    - **arguments**:  
      - `target_percent` (`int`, default=75): memory usage threshold to wait for  
      - `max_wait` (`int`, default=10): maximum time in seconds to wait  
      - `check_interval` (`int`, default=1): time interval between checks  
    - **returns**: `bool` — whether memory dropped below target before timeout  
    - **details**: checks system memory in a loop with optional timeout  

- #### `force_garbage_collection(pause_seconds=2)`
    - **description**: triggers garbage collection and waits briefly to allow cleanup  
    - **arguments**:  
      - `pause_seconds` (`int`, default=2): sleep time after gc to allow memory stabilization  
    - **returns**: `float` — memory usage in mb after cleanup  
    - **details**: helpful before/after loading/saving large datasets  

- #### `save_dataset_to_parquet(dataset, output_dir, batch_size=1000, verbose=True)`
    - **description**: saves a hugging face dataset or datasetdict to parquet files efficiently using batch-wise writing  
    - **arguments**:  
      - `dataset` (`Dataset` or `DatasetDict`): dataset to save  
      - `output_dir` (`str`): output directory for parquet files  
      - `batch_size` (`int`, default=1000): number of rows per write batch  
      - `verbose` (`bool`, default=True): log progress if true  
    - **returns**: _none_  
    - **output format**:  
      - for `DatasetDict`: each split saved as `<split>.parquet`  
      - for `Dataset`: saved as `dataset.parquet`  
      - also saves `dataset_info.json` with metadata  

- #### `load_parquet_as_dataset(input_dir, split=None, batch_size=5000, verbose=True)`
    - **description**: loads datasets from parquet format into hugging face dataset or datasetdict  
    - **arguments**:  
      - `input_dir` (`str`): directory containing parquet files  
      - `split` (`str`, optional): if provided, loads a specific split  
      - `batch_size` (`int`, default=5000): unused but reserved  
      - `verbose` (`bool`, default=True): log memory and progress  
    - **returns**: `Dataset` or `DatasetDict` — loaded data  
    - **input format**:  
      - `<split>.parquet` files or a single `dataset.parquet`  

- #### `process_in_batches(function, data, batch_size=100, **kwargs)`
    - **description**: applies a function to data in memory-efficient batches  
    - **arguments**:  
      - `function` (`callable`): the function to apply to each batch  
      - `data` (`list` or indexable): the data to process  
      - `batch_size` (`int`, default=100): number of items per batch  
      - `**kwargs`: passed to the function  
    - **returns**: `list` — concatenated results from all batches  
    - **details**:  
      - garbage collection is called between batches to manage memory  
      - useful for applying transforms on large lists or arrays 

## 10. Data Processing Workflow Details

This section details the sequence of operations transforming raw Twitter data into model-ready input, as implemented in the scripts, with specific function names and their operations.

### 10.1. Raw Data Loading

**Script:** `scripts/1_extract_tweets.py`

**Input:**
- `tweet_*.json`: Raw tweet data files
- `label.csv`: User labels (bot/human)
- `split.csv`: Dataset splits (train/test/dev)

**Key functions:**
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

**Key functions:**
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

**Key functions:**
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

**Key functions:**
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

**Key functions:**
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

**Key functions:**
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
