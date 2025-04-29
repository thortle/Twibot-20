# Twitter Bot Detection using Profile Text and DistilBERT

## Table of Contents
1. [Introduction](#1-introduction)
    - 1.1. Main Project: Twibot-20
    - 1.2. Extension: Twibot-22
    - 1.3. Key Differences
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
8. [Usage Instructions](#8-usage-instructions)
    - 8.1. Prerequisites
    - 8.2. Running the Pipeline (Default - HF Format)
    - 8.3. Running the Pipeline (Optional - Parquet Format)
    - 8.4. Optional Scripts
9. [API and Module Documentation](#9-api-and-module-documentation)
    - 9.1. `scripts/1_fix_dataset.py`
    - 9.2. `scripts/2_tokenize_dataset.py`
    - 9.3. `scripts/3_train_model.py`
    - 9.4. `scripts/4_predict.py`
    - 9.5. `scripts/convert_to_parquet.py`
    - 9.6. `scripts/benchmark_parquet.py`
    - 9.7. `utilities/dataset_splitter.py`
    - 9.8. `utilities/parquet_utils.py`
10. [Data Processing Workflow Details](#10-data-processing-workflow-details)
    - 10.1. Raw Data Loading
    - 10.2. Text Extraction and Cleaning
    - 10.3. Dataset Creation and Formatting
    - 10.4. Dataset Splitting
    - 10.5. Tokenization for Model Input
    - 10.6. Data Format Conversion (Optional)
    - 10.7. Data Flow Summary
11. [Parquet vs Hugging Face Performance Comparison](#11-parquet-vs-hugging-face-performance-comparison)
    - 11.1. Storage Efficiency
    - 11.2. Loading and Processing Performance
    - 11.3. When to Use Each Format
    - 11.4. Conclusion
12. [Requirements](#12-requirements)

---

## 1. Introduction

This project is divided into two distinct parts, each focusing on a different Twitter bot detection dataset:

### 1.1. Main Project: Twibot-20

- **Objective:** Build a machine learning model to classify Twitter accounts as either 'human' or 'bot' using the Twibot-20 dataset.
- **Approach:** Fine-tune a pre-trained DistilBERT model for sequence classification using the Hugging Face Transformers library.
- **Data Focus:** Classification based on user profile text (username, name, description, location) and up to 5 recent tweets when available.
- **Dataset Size:** Approximately 9,500 users (train/validation/test combined).
- **Features:** Uses combined profile information as the primary input.

### 1.2. Extension: Twibot-22

- **Objective:** Train a similar bot detection model using the newer and larger Twibot-22 dataset.
- **Approach:** Create a balanced subset of the imbalanced Twibot-22 dataset and fine-tune DistilBERT.
- **Data Focus:** Classification based solely on tweet content rather than profile information.
- **Dataset Size:** Uses a balanced subset of 1,000 tweets (500 bot, 500 human).
- **Features:** Uses only tweet text as input, without profile information.

### 1.3. Key Differences

| Feature | Twibot-20 (Main) | Twibot-22 (Extension) |
|---------|------------------|------------------------|
| Input Data | Profile + Tweets | Tweet text only |
| Dataset Size | ~9,500 users | 1,000 tweets (balanced subset) |
| Data Structure | User-centric | Tweet-centric |
| Split Strategy | Train/Val/Test (original + validation split) | Train/Val/Test (created from scratch) |
| Class Balance | Slightly imbalanced (56% bots) | Perfectly balanced (50% bots) |

Both implementations support the standard Hugging Face dataset format and the efficient Apache Parquet format, with detailed performance comparisons between the two storage approaches.

## 2. Project Structure

The project is organized into two main parts: Twibot-20 (main project) and Twibot-22 (extension).

```
/
├── scripts/                        # Main pipeline scripts for Twibot-20
│   ├── 1_fix_dataset.py            # Data extraction and preprocessing
│   ├── 2_tokenize_dataset.py       # Text tokenization
│   ├── 3_train_model.py            # Model training
│   ├── 4_predict.py                # Making predictions
│   ├── convert_to_parquet.py       # Convert datasets to Parquet format
│   └── benchmark_parquet.py        # Benchmark Parquet vs Hugging Face performance
│
├── utilities/                      # Helper modules for Twibot-20
│   ├── dataset_splitter.py         # Dataset splitting functionality
│   ├── parquet_utils.py            # Apache Parquet utilities
│   └── README_PARQUET.md           # Documentation for Parquet implementation
│
├── models/                         # Trained models for Twibot-20
│   └── distilbert-bot-detector/    # Trained DistilBERT model files
│
├── data/                           # Datasets for Twibot-20
│   ├── Twibot-20/                  # Original dataset files (node_new.json, etc.)
│   ├── twibot20_fixed_dataset/     # (Generated - HF Format)
│   ├── twibot20_fixed_tokenized/   # (Generated - HF Format)
│   ├── twibot20_llama_tokenized/   # (Generated - Alternative tokenizer for T5 model)
│   ├── twibot20_fixed_parquet/     # (Optional - Parquet Format)
│   └── twibot20_tokenized_parquet/ # (Optional - Parquet Format)
│
├── benchmark_results/              # Performance comparison results for Twibot-20
│   ├── storage_comparison.png      # Storage efficiency charts
│   ├── loading_comparison.png      # Loading time charts
│   ├── processing_comparison.png   # Processing time charts
│   ├── memory_comparison.png       # Memory usage charts
│   └── parquet_performance.md      # Detailed benchmark report
│
├── llama_model/                    # Alternative model implementation using T5 (Llama substitute)
│   ├── scripts/                    # Pipeline scripts for T5 model
│   ├── utilities/                  # Helper modules for T5 model
│   ├── models/                     # Trained T5 model files
│   └── README.md                   # Documentation for T5 implementation
│
├── Twibot-22/                      # Extension project directory (see Twibot-22/README.md for details)
│
└── README.md                       # Main project documentation
```


## 3. Pipeline Overview

The project follows a 4-step pipeline, executed via scripts in the `scripts/` directory:

1.  **Data Extraction & Preprocessing** (`1_fix_dataset.py`): Loads raw data, extracts profile/tweet text, cleans it, splits into train/validation/test sets, and saves the processed dataset.
2.  **Tokenization** (`2_tokenize_dataset.py`): Loads the processed dataset and converts the text into tokens suitable for the DistilBERT model.
3.  **Model Training** (`3_train_model.py`): Loads the tokenized dataset and fine-tunes the DistilBERT model for bot classification. Evaluates the model.
4.  **Prediction** (`4_predict.py`): Loads the fine-tuned model and provides an interface for classifying new text samples.

Each step (`1`, `2`, `3`) supports using either the standard Hugging Face dataset format or the Apache Parquet format via the `--use-parquet` flag for enhanced storage efficiency.

## 4. Data

### 4.1. Source Files
- **Location:** Expected in `data/Twibot-20/`
- **Files Used:**
    - `node_new.json`: Contains user profile information and potentially tweets. Structure: Dict mapping `user_id` (str) to user data (dict).
        ```json
        { "u17461978": { "description": "...", "name": "SHAQ", ... }, ... }
        ```
    - `label_new.json`: Maps `user_id` (str) to label ('human' or 'bot').
        ```json
        { "u17461978": "human", "u1297437077403885568": "bot", ... }
        ```
    - `split_new.json`: Defines original train/test user ID lists.
        ```json
        { "train": ["u17461978", ...], "test": [...], "dev": [...] }
        ```

### 4.2. Generated Data Formats
The pipeline generates processed and tokenized datasets, which can be stored in two formats:

1.  **Hugging Face Disk Format:**
    - Default format. Stored in `data/twibot20_fixed_dataset/` and `data/twibot20_fixed_tokenized/`.
    - Consists of `dataset_dict.json`, `dataset_info.json`, and subfolders for each split containing Apache Arrow files (`.arrow`) and index files (`.idx`). Optimized for fast loading and certain operations within the `datasets` library.
2.  **Apache Parquet Format:**
    - Optional format, enabled with `--use-parquet`. Stored in `data/twibot20_fixed_parquet/` and `data/twibot20_tokenized_parquet/`.
    - Consists of subfolders for each split containing one or more `.parquet` files. Parquet is a columnar storage format offering high compression and efficiency for large datasets. Metadata is stored alongside in `dataset_info.json` and `state.json`.

### 4.3. Data Preparation Steps
*(Executed by `scripts/1_fix_dataset.py`)*
- Raw JSON data is loaded.
- Text is extracted by combining: `Username`, `Name`, `Description`, `Location`, and up to 5 recent `Tweets` (if available).
- Text is cleaned: URLs removed, extra whitespace normalized.
- Data is converted to a Hugging Face `DatasetDict`.
- The initial 'train' split is further divided into 'train' (90%) and 'validation' (10%) splits using stratified sampling based on the 'label' column (via `utilities/dataset_splitter.py`).
- The final `DatasetDict` (containing 'train', 'validation', 'test' splits) is saved to disk (either HF format or Parquet).
- **Final Dataset Statistics:**
    - Train: 7,450 samples (56.1% bots, 43.9% humans)
    - Validation: 828 samples (56.0% bots, 44.0% humans)
    - Test: 1,183 samples (54.1% bots, 45.9% humans)
    - Average combined text length: ~150 characters per user.

### 4.4. Data Structures
- **Raw Data:** Primarily Python dictionaries loaded from JSON.
- **Processed/Tokenized Data (in memory):** `datasets.DatasetDict`. This object holds multiple `datasets.Dataset` instances (one per split: train, validation, test).
- **`datasets.Dataset` Structure:** Represents a table-like structure. Key columns generated by the pipeline:
    - `user_id` (`string`): User identifier.
    - `text` (`string`): The combined, cleaned text from profile and tweets.
    - `features` (`string`): JSON string of the raw node data (for potential future use).
    - `label` (`ClassLabel(names=['human', 'bot'])`): Integer label (0 or 1).
    - `input_ids` (`Sequence(int32)`): *(Added after tokenization)* List of token IDs.
    - `attention_mask` (`Sequence(int8)`): *(Added after tokenization)* Mask indicating real tokens vs padding.

## 5. Methodology

### 5.1. Preprocessing - Tokenization
*(Executed by `scripts/2_tokenize_dataset.py`)*
- **Tokenizer:** `distilbert-base-uncased` from Hugging Face Transformers. It converts text into sequences of numerical IDs.
- **Process:** The `text` column of the processed dataset is tokenized.
- **Parameters:**
    - `truncation=True`: Sequences longer than the model's maximum input length (512 tokens for DistilBERT) are truncated.
    - `padding=False`: Padding is applied dynamically per batch during training by the `DataCollatorWithPadding`.
- **Output:** Adds `input_ids` and `attention_mask` columns to the dataset.
- **Statistics (Train Split):**
    - Average tokens per sample: ~41 tokens.
    - Maximum tokens in a sample: ~300 tokens.
    - Samples exceeding max length (truncated): < 1%.
    - Samples with essentially empty text (≤ 2 tokens): ~2%.
- **Storage:** The tokenized dataset is saved (HF format or Parquet).

### 5.2. Model Architecture
- **Base Model:** `distilbert-base-uncased`. A smaller, faster version of BERT, maintaining good performance. Uses the Transformer architecture.
- **Task Adaptation:** Fine-tuned for sequence classification using `AutoModelForSequenceClassification`. A classification head (a linear layer) is placed on top of the base DistilBERT model's pooled output.
- **Configuration:** `num_labels=2`, `id2label={0: "human", 1: "bot"}`, `label2id={"human": 0, "bot": 1}`.

### 5.3. Training Process
*(Executed by `scripts/3_train_model.py`)*
- **Framework:** Hugging Face `Trainer` API.
- **Optimizer:** AdamW (default).
- **Key Hyperparameters:**
    - Learning Rate: 5e-5
    - Batch Size: 16 per device
    - Max Epochs: 3
    - Weight Decay: 0.01
- **Evaluation:** Performed on the validation set after each epoch. Metrics: Accuracy, Precision, Recall, F1-Score (weighted).
- **Best Model Selection:** Based on the highest F1-score achieved on the validation set.
- **Early Stopping:** Training stops if the validation F1-score does not improve for 2 consecutive epochs.
- **Hardware Acceleration:** Automatically uses MPS (Apple Silicon) or CUDA (NVIDIA GPU) if available, otherwise CPU.

## 6. Results

- **Final Evaluation (Test Set):** The best model checkpoint (selected based on validation F1) was evaluated on the held-out test set.

  | Metric          | Score  |
  |-----------------|--------|
  | Test Accuracy   | 0.78   |
  | Test Precision  | 0.77   |
  | Test Recall     | 0.78   |
  | Test F1-Score   | 0.77   |
  | Test Loss       | 0.52   |

- **Validation Performance Trend:**
  - Peak validation F1-score of 0.79 was achieved at the end of epoch 2.
  - Early stopping triggered after epoch 3, indicating no further improvement.
- **Training Curves:** Visualizations of training/validation loss and metrics over epochs can be found in `models/distilbert-bot-detector/training_curves.png`. *(Ensure this file is generated and saved there by `3_train_model.py` or move it)*

## 7. Discussion & Conclusion

### 7.1. Interpretation
- The fine-tuned DistilBERT model achieved a respectable F1-score of 77% and accuracy of 78% on the test set using only profile text and limited tweet data.
- This indicates that the textual content available in user profiles (and a few recent tweets) contains significant signals that the model can learn to distinguish between human and bot accounts within the Twibot-20 dataset context.

### 7.2. Alternative Model Comparison (T5 vs. DistilBERT)
- As an alternative approach, we also implemented a T5 model (as a substitute for Llama) for the same task. The implementation is available in the `llama_model/` directory.
- The T5 model achieved an accuracy of 72.44% and F1-score of 72.41%, which is approximately 5 percentage points lower than DistilBERT.
- Despite being a larger model (220M parameters vs. 66M for DistilBERT), T5 performed worse on this specific task, suggesting that smaller, task-specific models can outperform larger, more general models for specialized classification tasks.
- The T5 model also showed less decisive prediction behavior, with a tendency to classify most inputs as "Human" with moderate confidence (51-70%).
- For detailed comparison metrics and analysis, see the `llama_model/README.md` file.

### 7.3. Limitations
- **Data Scope:** The primary limitation is the reliance on limited textual data. Performance could likely be improved by incorporating user metadata (account age, follower/following ratio), behavioral patterns (posting frequency, content type), or network information (connections to known bots/humans), which were not used here.
- **Tweet Availability:** The `node_new.json` file did not consistently contain tweet data for all users, limiting the model's exposure to actual user-generated content beyond the profile.
- **Sophisticated Bots:** The model might struggle against advanced bots designed to mimic human profiles closely or those with very sparse profiles.
- **Generalization:** Performance on different Twitter datasets or newer bot types may vary. The Twibot-20 dataset has specific characteristics.
- **Text Cleaning & Tokenization:** Basic cleaning was applied. More advanced NLP techniques (e.g., handling emojis, non-standard characters, language detection) were not implemented. Truncation affects a small percentage (<1%) of very long profiles/tweet combinations.

### 7.4. Conclusion
- We successfully fine-tuned DistilBERT for Twitter bot detection using profile/tweet text from Twibot-20, achieving 78% accuracy.
- The project demonstrates the viability of using Transformer models on limited text data for this task and establishes a solid baseline.
- The integration of Apache Parquet provides significant storage savings (~5-15x) and offers flexibility for handling larger datasets or integration with other tools, although processing speed trade-offs exist compared to the native Hugging Face format for this specific dataset size.

## 8. Usage Instructions

### 8.1. Prerequisites

1. Clone the repository.
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Twibot-20 dataset files (`node_new.json`, `label_new.json`, `split_new.json`) and place them inside the `data/Twibot-20/` directory.

### 8.2. Running the Pipeline (Default - HF Format)

Execute the scripts sequentially:

```bash
# Step 1: Process the raw data
python scripts/1_fix_dataset.py

# Step 2: Tokenize the processed dataset
python scripts/2_tokenize_dataset.py

# Step 3: Train the model
python scripts/3_train_model.py

# Step 4: Make predictions with the trained model
python scripts/4_predict.py # For interactive prediction
```

This will generate datasets in `data/twibot20_fixed_dataset/` and `data/twibot20_fixed_tokenized/`, and the trained model in `models/distilbert-bot-detector/`.

### 8.3. Running the Pipeline (Optional - Parquet Format)

Use the `--use-parquet` flag for the processing and training steps:

```bash
# Step 1: Process the raw data using Parquet format
python scripts/1_fix_dataset.py --use-parquet

# Step 2: Tokenize the processed dataset using Parquet format
python scripts/2_tokenize_dataset.py --use-parquet

# Step 3: Train the model using Parquet format
python scripts/3_train_model.py --use-parquet

# Step 4: Make predictions with the trained model
python scripts/4_predict.py # Prediction script uses the saved model regardless of training data format
```

This will generate datasets in `data/twibot20_fixed_parquet/` and `data/twibot20_tokenized_parquet/`. The model is saved in the same location (`models/distilbert-bot-detector/`).

### 8.4. Optional Scripts

Convert existing HF datasets to Parquet:

```bash
# Convert the processed dataset
python scripts/convert_to_parquet.py --input_dir data/twibot20_fixed_dataset --output_dir data/twibot20_fixed_parquet

# Convert the tokenized dataset
python scripts/convert_to_parquet.py --input_dir data/twibot20_fixed_tokenized --output_dir data/twibot20_tokenized_parquet
```

Run performance benchmarks (requires datasets in both formats):

```bash
# Run the benchmark script
python scripts/benchmark_parquet.py

# Results will appear in console and benchmark_results/ directory
```

## 9. API and Module Documentation

This section provides a detailed description of inputs, outputs, and key functions for each script and module.

### 9.1. `scripts/1_fix_dataset.py`

- **Functionality:** Loads raw JSON data, extracts and cleans profile/tweet text, creates a DatasetDict with columns user_id, text, features (JSON string of the node), label (0/1), performs train/validation split, and saves the processed dataset.

- **Key Functions:** `load_twibot20_data()`, `extract_text_from_user()`, `clean_text()`, `convert_to_hf_dataset()`, `main()`.

- **Input:** Path to the directory containing the original JSON files (`data/Twibot-20/`). Optional argument `--use-parquet`.

- **Output:** Dataset (DatasetDict) saved to disk (`data/twibot20_fixed_dataset/`) OR in Parquet format (`data/twibot20_fixed_parquet/`).

### 9.2. `scripts/2_tokenize_dataset.py`

- **Functionality:** Loads the processed dataset (HF or Parquet), applies DistilBERT tokenization, and saves the resulting dataset.

- **Key Functions:** `preprocess_function()`, `main()`.

- **Input:** Path to the processed dataset (`data/twibot20_fixed_dataset/` or `data/twibot20_fixed_parquet/`). Optional argument `--use-parquet`.

- **Output:** Tokenized dataset (DatasetDict) saved to disk (`data/twibot20_fixed_tokenized/`) OR in Parquet format (`data/twibot20_tokenized_parquet/`).

### 9.3. `scripts/3_train_model.py`

- **Functionality:** Loads the tokenized dataset (HF or Parquet), configures and fine-tunes the DistilBERT model, evaluates and saves the model.

- **Key Functions:** `compute_metrics()`, `main()`.

- **Input:** Path to the tokenized dataset (`data/twibot20_fixed_tokenized/` or `data/twibot20_tokenized_parquet/`). Optional argument `--use-parquet`.

- **Output:** Fine-tuned model (`models/distilbert-bot-detector/`), evaluation results (console), training curves (`training_curves.png`).

### 9.4. `scripts/4_predict.py`

- **Functionality:** Loads the fine-tuned model for interactive predictions.

- **Key Functions:** `predict_bot_probability()`, `main()`.

- **Input:** Text to classify (via prompt), path to model (`models/distilbert-bot-detector/`).

- **Output:** Prediction ('Human'/'Bot') and confidence (console).

### 9.5. `scripts/convert_to_parquet.py`

- **Functionality:** Converts a Hugging Face disk format dataset to Parquet format.

- **Arguments:** `--input_dir`, `--output_dir`.

- **Usage:** Manual tool for format conversion.

### 9.6. `scripts/benchmark_parquet.py`

- **Functionality:** Compares the performance of HF disk and Parquet formats.

- **Input:** Paths to datasets in both formats.

- **Output:** Results (console), markdown report and charts (`benchmark_results/`).

### 9.7. `utilities/dataset_splitter.py`

- **Functionality:** Utility module for stratified splitting of a Hugging Face dataset.

- **API:** `split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label', random_state=42) -> DatasetDict`.

### 9.8. `utilities/parquet_utils.py`

- **Functionality:** Utility module for saving and loading Hugging Face datasets in Parquet format.

- **API:** `save_dataset_to_parquet(dataset_dict, output_dir)`, `load_parquet_as_dataset(input_dir) -> DatasetDict`.

## 10. Data Processing Workflow Details

This section details the sequence of operations transforming raw data into model-ready input, as implemented in the scripts, with specific function names and their operations.

### 10.1. Raw Data Loading

**Script:** `scripts/1_fix_dataset.py`

**Input:**
- `node_new.json`: Contains user profile information and tweets
- `label_new.json`: Maps user IDs to labels (human/bot)
- `split_new.json`: Defines original train/test user ID lists

**Key Functions:**
- `load_twibot20_data(data_dir)`: Loads and parses the three JSON files into Python dictionaries
- `json.load(file)`: Built-in JSON parser used to convert JSON files to Python dictionaries
- `check_data_integrity(nodes, labels, splits)`: Verifies that the loaded data is consistent and complete
- `print_dataset_stats(nodes, labels, splits)`: Outputs statistics about the dataset for verification

**Process Flow:**
1. `load_twibot20_data()` opens each JSON file and parses it into memory
2. The function returns three dictionaries:
   - `nodes`: Maps user IDs to user data (profile info and tweets)
   - `labels`: Maps user IDs to labels ("human" or "bot")
   - `splits`: Contains lists of user IDs for "train", "test", and "dev" sets
3. Data integrity checks ensure all referenced user IDs exist across dictionaries

### 10.2. Text Extraction and Cleaning

**Script:** `scripts/1_fix_dataset.py`

**Input:**
- Parsed dictionaries from step 10.1

**Key Functions:**
- `extract_text_from_user(user_data)`: Extracts and combines text from user profile and tweets
- `clean_text(text)`: Removes URLs, special characters, and normalizes whitespace
- `get_user_tweets(user_data)`: Extracts up to 5 most recent tweets from user data
- `format_profile_text(username, name, description, location)`: Formats profile fields into a single string

**Process Flow:**
1. For each user ID in the splits, `extract_text_from_user()` is called
2. The function extracts profile fields (username, name, description, location)
3. It also extracts up to 5 tweets using `get_user_tweets()` if available
4. All text is combined into a single string with field labels
5. `clean_text()` applies regex patterns like `re.sub(r'http\S+', '', text)` to remove URLs
6. The function returns the cleaned, combined text for each user

### 10.3. Dataset Creation and Formatting

**Script:** `scripts/1_fix_dataset.py`

**Input:**
- Cleaned text for each user, labels dictionary, splits dictionary

**Key Functions:**
- `convert_to_hf_dataset(user_texts, labels, splits)`: Creates a Hugging Face DatasetDict
- `Features(...)`: Defines the schema with explicit typing for each column
- `Dataset.from_dict(...)`: Creates a Dataset from a dictionary of lists
- `DatasetDict(...)`: Creates a dictionary of datasets for different splits

**Process Flow:**
1. `convert_to_hf_dataset()` organizes the data into dictionaries for each split
2. For each split, it creates lists of user_ids, texts, features, and labels
3. Features are explicitly typed using:
   ```python
   features = Features({
       'user_id': Value('string'),
       'text': Value('string'),
       'features': Value('string'),  # JSON string of raw node data
       'label': ClassLabel(names=['human', 'bot'])
   })
   ```
4. `Dataset.from_dict()` converts these dictionaries to Hugging Face Datasets
5. The function returns a DatasetDict with 'train' and 'test' splits

### 10.4. Dataset Splitting

**Script:** `scripts/1_fix_dataset.py` (using `utilities/dataset_splitter.py`)

**Input:**
- The initial DatasetDict with 'train' and 'test' splits

**Key Functions:**
- `split_dataset(dataset, test_size, stratify_by_column, random_state)`: Splits a dataset while preserving class distribution
- `sklearn.model_selection.train_test_split()`: Performs the actual splitting with stratification
- `Dataset.select()`: Creates a new dataset from selected indices
- `np.random.RandomState()`: Ensures reproducible random splits

**Process Flow:**
1. `split_dataset()` from `utilities/dataset_splitter.py` is called on the 'train' split
2. The function calculates indices for the new train/validation split using:
   ```python
   train_idx, val_idx = train_test_split(
       range(len(dataset)),
       test_size=test_size,
       stratify=dataset[stratify_by_column],
       random_state=random_state
   )
   ```
3. It creates new datasets using `Dataset.select(indices)` for both splits
4. The function returns a new DatasetDict with 'train' (90%), 'validation' (10%), and the original 'test' split

### 10.5. Tokenization for Model Input

**Script:** `scripts/2_tokenize_dataset.py`

**Input:**
- The processed DatasetDict with 'train', 'validation', and 'test' splits

**Key Functions:**
- `load_from_disk(dataset_path)` or `load_parquet_as_dataset(dataset_path)`: Loads the dataset
- `AutoTokenizer.from_pretrained("distilbert-base-uncased")`: Initializes the tokenizer
- `preprocess_function(examples)`: Applies tokenization to batches of examples
- `dataset.map(preprocess_function, batched=True)`: Applies the function to the entire dataset
- `compute_token_statistics(dataset)`: Analyzes token length distribution and potential truncation

**Process Flow:**
1. The dataset is loaded using the appropriate function based on format
2. The tokenizer is initialized with `AutoTokenizer.from_pretrained()`
3. `preprocess_function()` applies the tokenizer with specific parameters:
   ```python
   return tokenizer(
       examples["text"],
       truncation=True,
       padding=False,  # Dynamic padding applied during training
       max_length=512  # DistilBERT's maximum sequence length
   )
   ```
4. `dataset.map()` applies this function to all examples efficiently in batches
5. `compute_token_statistics()` analyzes the tokenized data to identify potential issues
6. The tokenized dataset is saved using the appropriate format function

### 10.6. Data Format Conversion (Optional)

**Scripts:**
- `scripts/1_fix_dataset.py`, `scripts/2_tokenize_dataset.py` (with `--use-parquet` flag)
- `scripts/convert_to_parquet.py` (standalone conversion)

**Key Functions:**
- `save_dataset_to_parquet(dataset_dict, output_dir)`: Converts and saves a DatasetDict to Parquet
- `load_parquet_as_dataset(input_dir)`: Loads a DatasetDict from Parquet files
- `dataset_to_pandas(dataset)`: Converts a Dataset to a pandas DataFrame
- `pandas_to_dataset(df, features)`: Converts a pandas DataFrame back to a Dataset

**Process Flow:**
1. `save_dataset_to_parquet()` iterates through each split in the DatasetDict
2. For each split, it converts the Dataset to a pandas DataFrame using `dataset_to_pandas()`
3. The DataFrame is saved as a Parquet file using `df.to_parquet()`
4. Metadata (feature definitions, etc.) is saved separately as JSON
5. When loading, `load_parquet_as_dataset()` reverses this process
6. The function reconstructs the DatasetDict with the original structure and feature definitions

### 10.7. Data Flow Summary

Raw JSON → Python Dictionaries → Cleaned Text Strings → HF DatasetDict (Initial Splits) → HF DatasetDict (Final Splits) → Tokenized HF DatasetDict → Model Input Tensors

A parallel path using Parquet for intermediate storage is available at each step after the initial dataset creation, implemented through the `--use-parquet` flag and the utility functions in `utilities/parquet_utils.py`.

## 11. Parquet vs Hugging Face Performance Comparison

This section summarizes the findings from `scripts/benchmark_parquet.py`. For full details and charts, see `benchmark_results/`.

### 11.1. Storage Efficiency

Apache Parquet demonstrates significant storage savings compared to the default Hugging Face disk format:

- **Fixed Dataset**: Up to 14.59x smaller (e.g., 16MB HF → 1.1MB Parquet).
- **Tokenized Dataset**: Up to 5.65x smaller (e.g., 10MB HF → 1.8MB Parquet).
- **Conclusion**: Parquet is highly effective for reducing disk space usage.

### 11.2. Loading and Processing Performance

- **Loading Time**: For this dataset size, loading from the Hugging Face disk format (leveraging memory mapping and Arrow caches) was generally faster than loading from Parquet files, especially for the more complex tokenized dataset.
- **Processing Operations**: Simple operations like filtering (`.filter()`) and sorting (`.sort()`) were often faster using the optimized Hugging Face format. Mapping operations (`.map()`) showed variable performance.
- **Note**: These speed comparisons might favor Parquet more significantly on much larger datasets where reading only necessary columns becomes a major advantage or when I/O becomes the bottleneck.

### 11.3. When to Use Each Format

- **Hugging Face Disk Format**: Recommended for small-to-medium datasets, rapid prototyping, and when peak processing speed for common datasets operations is prioritized.
- **Apache Parquet Format**: Recommended for large datasets, scenarios where disk space is limited, long-term archival, and interoperability with other data processing tools (Spark, Pandas, Dask).

### 11.4. Conclusion

Parquet offers compelling storage advantages. For the Twibot-20 dataset, the default Hugging Face format provides competitive or superior processing speed due to its optimizations. The choice involves a trade-off based on specific needs (storage vs speed). This project supports both, allowing flexibility.

## 12. Requirements

Ensure the following dependencies are installed:

```bash
torch>=1.12.0
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
pyarrow>=8.0.0  # For Parquet support
pandas>=1.4.0   # Dependency for Parquet utils
```

Install these dependencies using pip:

```bash
pip install -r requirements.txt
```

For Apple Silicon users, ensure PyTorch with MPS support is installed.

## 13. Dataset Attribution and Citation

### Twibot-20 and Twibot-22 Datasets

The datasets used in this project were created by researchers at Xi'an Jiaotong University (XJTU) and are available through their official repositories:

- **Twibot-20**: The original dataset used in the main project. Created by Shangbin Feng et al.
- **Twibot-22**: A larger and more comprehensive Twitter bot detection benchmark used in the extension project. Created by Shangbin Feng, Zhaoxuan Tan, et al.

### Citation

If you use this project or the datasets, please cite the original dataset papers:

```
@inproceedings{fengtwibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and others},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

### Dataset Repository

The original dataset repositories can be found at:
- Twibot-20: https://github.com/BunsenFeng/TwiBot-20
- Twibot-22: https://github.com/LuoUndergradXJTU/TwiBot-22

These datasets are designed to address challenges in Twitter bot detection research, including limited dataset scale, incomplete graph structure, and low annotation quality in previous datasets.
