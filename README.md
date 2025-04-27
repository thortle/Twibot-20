# Twitter Bot Detection using Profile Text and DistilBERT

## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Structure](#2-project-structure)
3. [Pipeline](#3-pipeline)
4. [Data](#4-data)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Discussion & Conclusion](#7-discussion--conclusion)
8. [Usage Instructions](#8-usage-instructions)
9. [Script Documentation](#9-script-documentation)
10. [Data Processing Workflow](#10-data-processing-workflow)
11. [Parquet vs Hugging Face Performance Comparison](#11-parquet-vs-hugging-face-performance-comparison)
12. [Requirements](#12-requirements)

## 1. Introduction

- **Objective:** This project aims to build a machine learning model to classify Twitter accounts as either 'human' or 'bot' using the Twibot-20 dataset.
- **Approach:** We fine-tuned a pre-trained DistilBERT model for sequence classification using the Hugging Face Transformers library.
- **Data Focus:** Classification is based on user profile text (username, name, description, location) and up to 5 recent tweets when available.

## 2. Project Structure

```
/
├── scripts/                      # Main pipeline scripts
│   ├── 1_fix_dataset.py          # Data extraction and preprocessing
│   ├── 2_tokenize_dataset.py     # Text tokenization
│   ├── 3_train_model.py          # Model training
│   ├── 4_predict.py              # Making predictions
│   ├── convert_to_parquet.py     # Convert datasets to Parquet format
│   └── benchmark_parquet.py      # Benchmark Parquet vs Hugging Face performance
│
├── utilities/                    # Helper modules
│   ├── dataset_splitter.py       # Dataset splitting functionality
│   ├── parquet_utils.py          # Apache Parquet utilities
│   └── README_PARQUET.md         # Documentation for Parquet implementation
│
├── models/                       # Trained models
│   └── distilbert-bot-detector/  # Trained model files
│
├── benchmark_results/            # Performance comparison results
│   ├── storage_comparison.png    # Storage efficiency charts
│   ├── loading_comparison.png    # Loading time charts
│   ├── processing_comparison.png # Processing time charts
│   ├── memory_comparison.png     # Memory usage charts
│   └── parquet_performance.md    # Detailed benchmark report
│
└── README.md                     # This file
```

## 3. Pipeline

The project follows a 4-step pipeline:

1. **Data Extraction** (`scripts/1_fix_dataset.py`): Extracts and preprocesses profile text from the Twibot-20 dataset.
2. **Tokenization** (`scripts/2_tokenize_dataset.py`): Tokenizes the text data using the DistilBERT tokenizer.
3. **Training** (`scripts/3_train_model.py`): Fine-tunes a DistilBERT model on the tokenized data.
4. **Prediction** (`scripts/4_predict.py`): Uses the trained model to make predictions on new data.

All pipeline steps support both Hugging Face dataset format and Apache Parquet format for efficient data storage and processing.

## 4. Data

- **Source:** The Twibot-20 dataset, a large-scale Twitter bot detection benchmark.
- **Files Used:**
  - `node_new.json`: Contains user profile information and tweets
  - `label_new.json`: Contains labels (bot or human) for users
  - `split_new.json`: Contains train/test splits
- **Directory Structure:**
  - `data/Twibot-20/`: Contains the original dataset files
    - `edge_new.json`: Network connections between users (not used in this model)
    - `label_new.json`: Binary labels (bot/human) for all users
    - `node_new.json`: Raw user profile data and tweet content
    - `split_new.json`: Original dataset partitioning
  - `data/twibot20_fixed_dataset/`: Contains the processed data with HuggingFace structure
    - `dataset_dict.json`: Stores dataset configuration
    - `dataset_info.json`: Contains detailed statistics for all splits
    - Separate folders for each data split (train/validation/test)
  - `data/twibot20_fixed_parquet/`: Contains the processed data in Apache Parquet format
    - `train.parquet`: Train split data in columnar format
    - `validation.parquet`: Validation split data in columnar format
    - `test.parquet`: Test split data in columnar format
  - `data/twibot20_fixed_tokenized/`: Contains the tokenized dataset in HuggingFace format
    - `dataset_dict.json`: Stores tokenized dataset configuration
    - Separate folders for each tokenized split (train/validation/test)
  - `data/twibot20_tokenized_parquet/`: Contains the tokenized dataset in Apache Parquet format
    - `train.parquet`: Tokenized train split in columnar format
    - `validation.parquet`: Tokenized validation split in columnar format
    - `test.parquet`: Tokenized test split in columnar format
- **Preparation Steps:**
  - Raw JSON data is loaded from the three files
  - Text is extracted by combining username, name, description, location, and up to 5 tweets
  - The extracted text is cleaned by removing URLs and extra whitespace
  - The data is converted to Hugging Face `DatasetDict` format
  - The train set is split into train and validation sets (90/10 ratio) with stratification by label
  - The processed data is saved in both Hugging Face format and Apache Parquet format
  - Final dataset statistics:
    - Train: 7,450 samples (56.1% bots, 43.9% humans)
    - Validation: 828 samples (56.0% bots, 44.0% humans)
    - Test: 1,183 samples (54.1% bots, 45.9% humans)
    - Average text length: ~150 characters per user
    - Storage efficiency with Parquet: 14.5x reduction (16MB → 1.1MB)

## 5. Methodology

- **Preprocessing - Tokenization:**
  - Profile and tweet text is tokenized using the `distilbert-base-uncased` tokenizer
  - Truncation is applied to handle sequences longer than the model's max length (512 tokens)
  - Dynamic padding is used during training (via `DataCollatorWithPadding`)
  - Tokenization statistics:
    - Average tokens per sample: ~41 tokens
    - Maximum tokens in a sample: ~300 tokens
    - Less than 1% of samples exceeded the model's max length
    - Only ~2% of samples had just special tokens (indicating empty text)
  - The tokenized dataset is saved in both Hugging Face format (`twibot20_fixed_tokenized`) and Apache Parquet format (`twibot20_tokenized_parquet`)
  - Storage efficiency with Parquet for tokenized data: 3.4x reduction (10MB → 2.9MB)

- **Model:**
  - Base model: `distilbert-base-uncased` loaded via `AutoModelForSequenceClassification`
  - Configuration: `num_labels=2` with label mapping `{0: "human", 1: "bot"}`
  - The model has ~66 million parameters

- **Training:**
  - Training is performed using the Hugging Face `Trainer` API
  - Hyperparameters:
    - Learning rate: 5e-5
    - Batch size: 16 per device
    - Maximum epochs: 3 (with early stopping)
    - Weight decay: 0.01
    - Optimizer: AdamW (default in Trainer)
  - Training can utilize Apple Silicon MPS backend for acceleration when available
  - Early stopping is implemented with patience of 2 epochs
  - Evaluation metrics: accuracy, precision, recall, and F1-score

## 6. Results

- **Quantitative Results:**

  | Metric          | Score  |
  |-----------------|--------|
  | Test Accuracy   | 0.78   |
  | Test Precision  | 0.77   |
  | Test Recall     | 0.78   |
  | Test F1-Score   | 0.77   |
  | Test Loss       | 0.52   |

- **Validation Performance:**
  - Peak validation F1-score: 0.79 (reached at epoch 2)
  - Early stopping activated after epoch 3 (no improvement in F1-score)

## 7. Discussion & Conclusion

- **Interpretation:**
  - The model achieved a 77% F1-score classifying bots based only on profile text, indicating that user profiles contain strong signals for bot detection.
  - Profile information (username, name, description, location) provides sufficient context for the model to distinguish between bots and humans with reasonable accuracy.

- **Limitations:**
  - Tweet content was not available/used in this model, which could potentially improve performance.
  - The model may struggle with sophisticated bots that have human-like profiles.
  - Performance could be improved with additional features like user behavior patterns, network structure, or temporal activity.

- **Conclusion:**
  - We successfully fine-tuned a DistilBERT model on Twitter profile data to detect bots with 78% accuracy.
  - This demonstrates that transformer-based models can effectively leverage the limited text in user profiles for bot detection.
  - The model provides a strong baseline that could be enhanced with additional data sources in future work.

## 8. Usage Instructions

### 1. Data Extraction

```bash
python scripts/1_fix_dataset.py
```

This script extracts profile text from the Twibot-20 dataset and creates a new dataset with better text content for training. The data is saved in both Hugging Face and Parquet formats.

### 2. Tokenization

```bash
# Using Hugging Face format
python scripts/2_tokenize_dataset.py

# Using Parquet format
python scripts/2_tokenize_dataset.py --use-parquet
```

This script tokenizes the fixed dataset using the DistilBERT tokenizer. You can choose between Hugging Face format or Parquet format.

### 3. Training

```bash
# Using Hugging Face format
python scripts/3_train_model.py

# Using Parquet format
python scripts/3_train_model.py --use-parquet
```

This script fine-tunes a DistilBERT model on the tokenized data. You can choose between Hugging Face format or Parquet format.

### 4. Prediction

```bash
python scripts/4_predict.py
```

This script loads the trained model and allows you to make predictions on new text data.

### 5. Converting to Parquet

```bash
python scripts/convert_to_parquet.py
```

This script converts existing datasets to Parquet format for more efficient storage and faster loading.

## 9. Script Documentation

Each script in this project serves a specific purpose in the data processing and model training pipeline:

### utilities/dataset_splitter.py
- **Purpose**: Splits datasets into train, validation, and test sets with stratification
- **Input**: A Hugging Face DatasetDict with 'train' and 'test' splits
- **Output**: A DatasetDict with 'train', 'validation', and 'test' splits
- **When to use**: Used by 1_fix_dataset.py to create validation splits

### utilities/parquet_utils.py
- **Purpose**: Provides utility functions for working with Apache Parquet files
- **Key features**:
  - Convert Hugging Face datasets to Parquet format
  - Load Parquet files as Hugging Face datasets
  - Get metadata and schema information from Parquet files
  - Read samples from Parquet files
- **When to use**: Used by scripts to save and load data in Parquet format

### scripts/1_fix_dataset.py
- **Purpose**: Processes raw Twibot-20 data files and extracts text from user profiles and tweets
- **Input**: Raw JSON files from the Twibot-20 dataset
- **Output**: Creates the `twibot20_fixed_dataset` directory with processed data and `twibot20_fixed_parquet` with Parquet data
- **Key features**:
  - Extracts username, name, description, location and up to 5 tweets when available
  - Cleans text by removing URLs and extra whitespace
  - Uses dataset_splitter.py to create train/validation/test splits
  - Saves data in both Hugging Face and Parquet formats
- **Next step**: Run 2_tokenize_dataset.py after this script

### scripts/2_tokenize_dataset.py
- **Purpose**: Tokenizes the processed dataset for input to the DistilBERT model
- **Input**: The fixed dataset from `twibot20_fixed_dataset` or `twibot20_fixed_parquet` directory
- **Output**: Creates the `twibot20_fixed_tokenized` or `twibot20_tokenized_parquet` directory with tokenized data
- **Key features**:
  - Uses DistilBERT tokenizer to process the text
  - Provides detailed tokenization statistics
  - Shows examples of tokenized output
  - Supports both Hugging Face and Parquet formats with `--use-parquet` flag
- **Next step**: Run 3_train_model.py after this script

### scripts/3_train_model.py
- **Purpose**: Fine-tunes a DistilBERT model on the tokenized data for bot detection
- **Input**: The tokenized dataset from `twibot20_fixed_tokenized` or `twibot20_tokenized_parquet` directory
- **Output**: Trained model saved to the `models/distilbert-bot-detector` directory
- **Key features**:
  - Supports MPS for Apple Silicon or CUDA for NVIDIA GPUs
  - Implements early stopping to prevent overfitting
  - Evaluates on test set after training
  - Supports both Hugging Face and Parquet formats with `--use-parquet` flag
- **Next step**: Use the trained model for predictions

### scripts/4_predict.py
- **Purpose**: Uses the trained model to make predictions on new text inputs
- **Input**: The trained model from `models/distilbert-bot-detector` directory
- **Output**: Bot/human predictions for provided text inputs
- **Usage**: Provides an interactive interface and sample predictions
- **When to use**: After training the model to make predictions on new data

### scripts/convert_to_parquet.py
- **Purpose**: Converts existing datasets to Apache Parquet format
- **Input**: Datasets in Hugging Face format
- **Output**: Datasets in Parquet format
- **Key features**:
  - Converts raw JSON data to Parquet
  - Converts fixed dataset to Parquet
  - Converts tokenized dataset to Parquet
  - Provides file size comparisons
- **When to use**: When you want to optimize storage space and loading times

### scripts/benchmark_parquet.py
- **Purpose**: Benchmarks performance of Parquet vs Hugging Face dataset formats
- **Input**: Datasets in both Hugging Face and Parquet formats
- **Output**: Performance metrics and visualizations in the `benchmark_results` directory
- **Key features**:
  - Measures storage efficiency (file size)
  - Benchmarks loading time for both formats
  - Compares processing performance (filtering, mapping, sorting)
  - Analyzes memory usage
  - Generates charts and a detailed markdown report
- **When to use**: When you want to evaluate the performance trade-offs between formats

## 10. Data Processing Workflow

This section provides a detailed explanation of how raw Twitter data is processed, transformed, and prepared for the bot detection model. The workflow follows a series of well-defined steps, each handled by specific functions across multiple scripts.

### 10.1 Raw Data Loading and Initial Processing

**Script**: `scripts/1_fix_dataset.py`
**Key Function**: `load_twibot20_data(data_dir)`

This function loads the raw Twibot-20 dataset from three JSON files:
- `node_new.json`: Contains user profile information and tweets
- `label_new.json`: Contains binary labels (bot/human) for each user
- `split_new.json`: Contains the original train/test split information

```python
def load_twibot20_data(data_dir):
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
```

The function returns three Python dictionaries:
1. `nodes`: Maps user IDs to their profile data and tweets
2. `labels`: Maps user IDs to their labels ('bot' or 'human')
3. `splits`: Maps split names ('train', 'test') to lists of user IDs

### 10.2 Text Extraction and Cleaning

**Script**: `scripts/1_fix_dataset.py`
**Key Functions**: `extract_text_from_user(user_data)` and `clean_text(text)`

These functions transform the raw user data into clean, meaningful text for the model:

1. **Text Extraction** (`extract_text_from_user`):
   - Extracts and formats profile information (username, name, description, location)
   - Extracts up to 5 recent tweets from each user
   - Combines all text parts with appropriate labels and formatting

```python
def extract_text_from_user(user_data):
    text_parts = []

    # Extract profile information
    if 'username' in user_data and user_data['username']:
        text_parts.append(f"Username: {user_data['username']}")

    # [Similar extraction for name, description, location]

    # Extract tweets (up to 5)
    if 'tweet' in user_data and isinstance(user_data['tweet'], list):
        tweet_texts = []
        for tweet in user_data['tweet'][:5]:
            if 'text' in tweet and tweet['text']:
                tweet_texts.append(tweet['text'])

        if tweet_texts:
            text_parts.append("Tweets:")
            for i, tweet_text in enumerate(tweet_texts):
                text_parts.append(f"  Tweet {i+1}: {tweet_text}")

    return "\n".join(text_parts)
```

2. **Text Cleaning** (`clean_text`):
   - Removes URLs using regular expressions
   - Normalizes whitespace
   - Handles edge cases like non-string inputs

```python
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

### 10.3 Dataset Creation and Formatting

**Script**: `scripts/1_fix_dataset.py`
**Key Function**: `convert_to_hf_dataset(nodes, labels, splits)`

This function transforms the processed data into a structured Hugging Face `DatasetDict` format:

1. Creates separate datasets for each split ('train', 'test')
2. For each user in a split:
   - Extracts and cleans text using the functions above
   - Converts labels to integers (0 for human, 1 for bot)
   - Stores the full user features as a JSON string
3. Defines proper feature types for the dataset
4. Creates a `Dataset` object for each split
5. Returns a `DatasetDict` containing all splits

```python
def convert_to_hf_dataset(nodes, labels, splits):
    datasets = {}

    for split_name in ['train', 'test']:
        user_ids = []
        texts = []
        features_list = []
        label_list = []

        for user_id in splits[split_name]:
            if user_id in nodes and user_id in labels:
                # Extract and clean text
                user_text = extract_text_from_user(nodes[user_id])
                cleaned_text = clean_text(user_text)

                # Store data
                user_ids.append(user_id)
                texts.append(cleaned_text)
                features_list.append(json.dumps(nodes[user_id]))
                label_value = 1 if labels[user_id] == 'bot' else 0
                label_list.append(label_value)

        # Define features with proper types
        features = Features({
            'user_id': Value('string'),
            'text': Value('string'),
            'features': Value('string'),
            'label': ClassLabel(num_classes=2, names=['human', 'bot'])
        })

        # Create Dataset
        datasets[split_name] = Dataset.from_dict({
            'user_id': user_ids,
            'text': texts,
            'features': features_list,
            'label': label_list
        }, features=features)

    return DatasetDict(datasets)
```

### 10.4 Dataset Splitting

**Script**: `utilities/dataset_splitter.py`
**Key Function**: `split_dataset(combined_dataset, test_size, stratify_by_column, random_state)`

This utility function splits the 'train' split into 'train' and 'validation' splits:

1. Verifies that the input is a valid `DatasetDict` with required splits
2. Checks if the stratification column is a `ClassLabel` type
3. Performs stratified splitting to maintain label distribution:
   - If the column is a `ClassLabel`, uses Hugging Face's built-in `train_test_split`
   - Otherwise, uses scikit-learn's `train_test_split` with manual stratification
4. Creates a new `DatasetDict` with 'train', 'validation', and 'test' splits

```python
def split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label', random_state=42):
    # Check if the stratify_by_column is a ClassLabel
    is_class_label = isinstance(combined_dataset['train'].features.get(stratify_by_column), ClassLabel)

    if is_class_label:
        # Use Hugging Face's built-in stratified split
        train_test_split = combined_dataset['train'].train_test_split(
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            seed=random_state
        )
    else:
        # Use scikit-learn for stratification
        from sklearn.model_selection import train_test_split as sklearn_split

        indices = np.arange(len(combined_dataset['train']))
        train_indices, val_indices = sklearn_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=combined_dataset['train'][stratify_by_column]
        )

        train_test_split = {
            'train': combined_dataset['train'].select(train_indices),
            'test': combined_dataset['train'].select(val_indices)
        }

    # Create final DatasetDict
    return DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test'],
        'test': combined_dataset['test']
    })
```

### 10.5 Tokenization for Model Input

**Script**: `scripts/2_tokenize_dataset.py`
**Key Function**: `preprocess_function(examples)`

This function prepares the text data for input to the DistilBERT model:

1. Loads the DistilBERT tokenizer
2. Defines a preprocessing function that:
   - Tokenizes the text data
   - Applies truncation to handle sequences longer than the model's max length
   - Returns input IDs and attention masks
3. Applies the tokenization to the entire dataset using `dataset.map()`

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define preprocessing function
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Padding will be handled later by DataCollator
        max_length=tokenizer.model_max_length
    )

# Apply tokenization
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    desc="Tokenizing dataset"
)
```

### 10.6 Data Format Conversion (Optional)

**Script**: `utilities/parquet_utils.py`
**Key Functions**: `save_dataset_to_parquet(dataset, output_dir)` and `load_parquet_as_dataset(parquet_path)`

These utility functions convert between Hugging Face datasets and Apache Parquet format:

1. **Saving to Parquet** (`save_dataset_to_parquet`):
   - Converts Hugging Face datasets to pandas DataFrames
   - Converts DataFrames to PyArrow Tables
   - Writes Tables to Parquet files with compression

2. **Loading from Parquet** (`load_parquet_as_dataset`):
   - Reads Parquet files into pandas DataFrames
   - Reconstructs proper feature types (ClassLabel, Sequence, etc.)
   - Converts DataFrames back to Hugging Face Datasets
   - Handles both individual datasets and dataset dictionaries

### 10.7 Data Flow Summary

The complete data processing workflow follows these steps:

1. **Raw Data Loading**: JSON files → Python dictionaries
2. **Text Extraction**: User profiles and tweets → Structured text
3. **Text Cleaning**: Raw text → Cleaned text (URLs removed, whitespace normalized)
4. **Dataset Creation**: Cleaned data → Hugging Face DatasetDict
5. **Dataset Splitting**: Original splits → Train/Validation/Test splits
6. **Tokenization**: Text data → Model-ready token IDs and attention masks
7. **Format Conversion** (Optional): Hugging Face format ↔ Apache Parquet format

This workflow transforms raw, unstructured Twitter data into a clean, structured, and tokenized format that's ready for input to the DistilBERT model for bot detection.

## 11. Parquet vs Hugging Face Performance Comparison

We conducted detailed benchmarks to compare the performance of Apache Parquet and Hugging Face dataset formats in our Twitter bot detection pipeline. The results highlight the trade-offs between storage efficiency and processing performance.

### 11.1 Storage Efficiency

| Dataset Type | Hugging Face Size (MB) | Parquet Size (MB) | Compression Ratio |
|--------------|------------------------|-------------------|-------------------|
| Fixed Dataset | 15.95 | 1.09 | 14.59x |
| Tokenized Dataset | 10.31 | 1.83 | 5.65x |

Parquet provides exceptional storage efficiency, reducing the fixed dataset size by 14.59x and the tokenized dataset size by 5.65x. This makes it ideal for storing large datasets.

### 11.2 Loading and Processing Performance

| Operation | Hugging Face | Parquet | Notes |
|-----------|--------------|---------|-------|
| Loading Fixed Dataset | Faster | Slower | HF is optimized for quick in-memory loading |
| Loading Tokenized Dataset | Faster | Slower | Parquet requires additional deserialization |
| Processing Operations | Faster | Slower | HF has optimized in-memory operations |

While Hugging Face datasets load and process faster in our benchmarks, this is primarily due to the small size of our test dataset (Twibot-20). For larger datasets, Parquet's columnar format would likely show performance advantages as it allows reading only the required columns and benefits from efficient compression.

### 11.3 When to Use Each Format

- **Use Hugging Face format when**:
  - Working with small to medium datasets
  - Performing frequent in-memory operations
  - Quick iteration during development

- **Use Parquet format when**:
  - Working with large datasets in general
  - Storage space is a concern
  - Performing column-specific operations
  - Sharing data with other tools and frameworks

### 11.4 Conclusion

For the Twibot-20 dataset, the storage benefits of Parquet (5-15x reduction) outweigh the slight performance overhead. For larger datasets, Parquet would likely provide both storage and performance benefits, especially when working with limited memory resources.

## 12. Requirements

```
torch>=1.12.0
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
ijson>=3.1.4
pyarrow>=8.0.0
pandas>=1.4.0
```

For Apple Silicon users, ensure you have PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

For Parquet support, ensure you have PyArrow installed:
```bash
pip install pyarrow pandas
```
