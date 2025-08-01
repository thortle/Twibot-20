# Twitter bot detection using profile text and DistilBERT

## Table of contents
1. [Project specifications](#1-project-specifications)  
    1.1. [Introduction](#11-introduction)  
        - 1.1.1. Main project : Twibot-20  
        - 1.1.2. Extension : Twibot-22  
        - 1.1.3. Key differences  
    1.2. [Project structure](#12-project-structure)  
    1.3. [Pipeline overview](#13-pipeline-overview)  
    1.4. [Data](#14-data)  
        - 1.4.1. Source files  
        - 1.4.2. Generated data formats  
        - 1.4.3. Data preparation steps  
        - 1.4.4. Data structures  
    1.5. [Methodology](#15-methodology)  
        - 1.5.1. Preprocessing - Tokenization  
        - 1.5.2. Model architecture  
        - 1.5.3. Training process

2. [Development outcomes](#2-development-outcomes)  
    2.1. [Results](#21-results)  
    2.2. [Discussion and conclusion](#22-discussion-and-conclusion)  
        - 2.2.1. Interpretation  
        - 2.2.2. Alternative model comparison (T5 vs. DistilBERT)  
        - 2.2.3. Limitations  
        - 2.2.4. Conclusion  

3. [Testing, bugs and performance optimizations](#3-testing-bugs-and-performance-optimizations)  
    3.1. [Testing procedures](#31-testing-procedures)  
        - 3.1.1. Data processing tests  
            - 3.1.1.1. Dataset loading test  
            - 3.1.1.2. Text extraction test  
            - 3.1.1.3. Dataset split test  
        - 3.1.2. Tokenization tests  
            - 3.1.2.1. Tokenizer loading test  
            - 3.1.2.2. Tokenization quality test  
            - 3.1.2.3. Format compatibility test  
        - 3.1.3. Model training tests  
            - 3.1.3.1. Model loading test  
            - 3.1.3.2. Training progress test  
            - 3.1.3.3. Device compatibility test  
        - 3.1.4. Prediction tests  
            - 3.1.4.1. Model loading for inference test  
            - 3.1.4.2. Prediction quality test  
        - 3.1.5. Parquet performance tests  
            - 3.1.5.1. Storage efficiency test  
            - 3.1.5.2. Loading speed test  
            - 3.1.5.3. Processing performance test  

    3.2. [Bugs and issues encountered](#32-bugs-and-issues-encountered)  
        - 3.2.1. Data processing issues  
            - 3.2.1.1. User ID format mismatch  
            - 3.2.1.2. Empty text fields  
            - 3.2.1.3. ClassLabel type requirement  
            - 3.2.1.4. JSON parsing errors  
            - 3.2.1.5. Tweet data structure inconsistency  
        - 3.2.2. Tokenization issues  
            - 3.2.2.1. Token length imbalance  
            - 3.2.2.2. Truncation of long texts  
        - 3.2.3. Memory and resource issues  
            - 3.2.3.1. Excessive memory usage with large files  
        - 3.2.4. Model training issues  
            - 3.2.4.1. Device detection errors  
            - 3.2.4.2. Memory usage spikes  
        - 3.2.5. Parquet conversion issues  
            - 3.2.5.1. Feature type preservation  
            - 3.2.5.2. Sequence column handling  

    3.3. [Performance optimizations](#33-performance-optimizations)  
        - 3.3.1. Memory efficiency improvements  
            - 3.3.1.1. Incremental processing  
            - 3.3.1.2. Garbage collection  
        - 3.3.2. Speed improvements  
            - 3.3.2.1. Parquet format  
            - 3.3.2.2. Batch processing  

    3.4. [Lessons learned](#34-lessons-learned)

4. [Usage instructions](#4-usage-instructions)  
    4.1. Prerequisites  
    4.2. Running the pipeline (default - HF format)  
    4.3. Running the pipeline (optional - Parquet format)  
    4.4. Optional scripts  

5. [API and module documentation](#5-api-and-module-documentation)  
    5.1. `llama_model/scripts/1_fix_dataset.py`  
    5.2. `llama_model/scripts/2_tokenize_dataset.py`  
    5.3. `llama_model/scripts/3_train_model.py`  
    5.4. `llama_model/scripts/4_predict.py`  
   5.5. `llama_model/scripts/sample_prediction.py`    
    5.6. `llama_model/utilities/dataset_splitter.py`    
    5.7. `scripts/1_fix_dataset.py`    
    5.8. `scripts/2_tokenize_dataset.py`  
    5.9  `scripts/3_train_model.py`  
    5.10 `scripts/4_predict.py`  
    5.11. `scripts/benchmark_parquet.py`  
    5.12. `scripts/convert_to_parquet.py`  
    5.13. `utilities/dataset_splitter.py`  
    5.14. `utilities/parquet_utils.py`  

7. [Data processing workflow details](#6-data-processing-workflow-details)  
    6.1. Raw data loading  
    6.2. Text extraction and cleaning  
    6.3. Dataset creation and formatting  
    6.4. Dataset splitting  
    6.5. Tokenization for model input  
    6.6. Data format conversion (optional)  
    6.7. Data flow summary  

8. [Parquet vs Hugging Face performance comparison](#7-parquet-vs-hugging-face-performance-comparison)  
    7.1. Storage efficiency  
    7.2. Loading and processing performance  
    7.3. When to use each format  
    7.4. Conclusion  

9. [Requirements](#8-requirements)
10. [Dataset attribution and citation](#9-dataset-attribution-and-citation)

---

## 1. Project specification

### 1.1 Introduction

This project is divided into two distinct parts, each focusing on a different Twitter bot detection dataset :

#### 1.1.1 Main project : Twibot-20  

- **Objective :** build a machine learning model to classify Twitter accounts as either 'human' or 'bot' using the Twibot-20 dataset.
- **Approach :** fine-tune a pre-trained DistilBERT model for sequence classification using the Hugging Face Transformers library.
- **Data focus :** classification based on user profile text (username, name, description, location) and up to 5 recent tweets when available.
- **Dataset size :** approximately 9,500 users (train/validation/test combined).
- **Features :** uses combined profile information as the primary input.

#### 1.1.2 Extension : Twibot-22  

- **Objective :** train a similar bot detection model using the newer and larger Twibot-22 dataset.
- **Approach :** create a balanced subset of the imbalanced Twibot-22 dataset and fine-tune DistilBERT.
- **Data focus :** classification based solely on tweet content rather than profile information.
- **Dataset size :** uses a balanced subset of 1,000 tweets (500 bot, 500 human).
- **Features :** uses only tweet text as input, without profile information.

#### 1.1.3 Key differences

| Feature | Twibot-20 (Main) | Twibot-22 (Extension) |
|---------|------------------|------------------------|
| Input data | Profile + tweets | Tweet text only |
| Dataset size | ~9,500 users | 1,000 tweets (balanced subset) |
| Data structure | User-centric | Tweet-centric |
| Split strategy | Train/Val/Test (original + validation split) | Train/Val/Test (created from scratch) |
| Class balance | Slightly imbalanced (56% bots) | Perfectly balanced (50% bots) |

Both implementations support the standard Hugging Face dataset format and the efficient Apache Parquet format, with detailed performance comparisons between the two storage approaches.

### 1.2 Project structure

The project is organized into two main parts : Twibot-20 (main project) and Twibot-22 (extension).

```
/
├── scripts/                        # Main pipeline scripts for Twibot-20
│   ├── 1_fix_dataset.py            # Data extraction and preprocessing
│   ├── 2_tokenize_dataset.py       # Text tokenization
│   ├── 3_train_model.py            # Model training
│   ├── 4_predict.py                # Making predictions
│   ├── convert_to_parquet.py       # Convert datasets to parquet format
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
│   ├── twibot20_llama_tokenized/   # (Generated - alternative tokenizer for T5 model)
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


### 1.3 Pipeline overview

The project follows a 4-step pipeline, executed via scripts in the `scripts/` directory :

1.  **Data extraction and preprocessing** (`1_fix_dataset.py`): loads raw data, extracts profile/tweet text, cleans it, splits into train/validation/test sets, and saves the processed dataset.
2.  **Tokenization** (`2_tokenize_dataset.py`): loads the processed dataset and converts the text into tokens suitable for the DistilBERT model.
3.  **Model training** (`3_train_model.py`): loads the tokenized dataset and fine-tunes the DistilBERT model for bot classification. Evaluates the model.
4.  **Prediction** (`4_predict.py`): loads the fine-tuned model and provides an interface for classifying new text samples.

Each step (`1`, `2`, `3`) supports using either the standard Hugging Face dataset format or the Apache Parquet format via the `--use-parquet` flag for enhanced storage efficiency.

### 1.4 Data

#### 1.4.1 Source files
- **Location :** expected in `data/Twibot-20/`
- **Files used :**
    - `node_new.json`: contains user profile information and potentially tweets. Structure : dict mapping `user_id` (str) to user data (dict).
        ```json
        { "u17461978": { "description": "...", "name": "SHAQ", ... }, ... }
        ```
    - `label_new.json`: maps `user_id` (str) to label ('human' or 'bot').
        ```json
        { "u17461978": "human", "u1297437077403885568": "bot", ... }
        ```
    - `split_new.json`: defines original train/test user ID lists.
        ```json
        { "train": ["u17461978", ...], "test": [...], "dev": [...] }
        ```

#### 1.4.2 Generated data formats
The pipeline generates processed and tokenized datasets, which can be stored in two formats :

1.  **Hugging Face disk format :**
    - Default format. Stored in `data/twibot20_fixed_dataset/` and `data/twibot20_fixed_tokenized/`.
    - Consists of `dataset_dict.json`, `dataset_info.json`, and subfolders for each split containing Apache Arrow files (`.arrow`) and index files (`.idx`). Optimized for fast loading and certain operations within the `datasets` library.
2.  **Apache Parquet format :**
    - Optional format, enabled with `--use-parquet`. Stored in `data/twibot20_fixed_parquet/` and `data/twibot20_tokenized_parquet/`.
    - Consists of subfolders for each split containing one or more `.parquet` files. Parquet is a columnar storage format offering high compression and efficiency for large datasets. Metadata is stored alongside in `dataset_info.json` and `state.json`.

#### 1.4.3 Data preparation steps
*(Executed by `scripts/1_fix_dataset.py`)*
- Raw JSON data is loaded.
- Text is extracted by combining : `Username`, `Name`, `Description`, `Location`, and up to 5 recent `Tweets` (if available).
- Text is cleaned : URLs removed, extra whitespace normalized.
- Data is converted to a Hugging Face `DatasetDict`.
- The initial 'train' split is further divided into 'train' (90%) and 'validation' (10%) splits using stratified sampling based on the 'label' column (via `utilities/dataset_splitter.py`).
- The final `DatasetDict` (containing 'train', 'validation', 'test' splits) is saved to disk (either HF format or Parquet).
- **Final dataset statistics :**
    - Train : 7,450 samples (56.1% bots, 43.9% humans)
    - Validation : 828 samples (56.0% bots, 44.0% humans)
    - Test : 1,183 samples (54.1% bots, 45.9% humans)
    - Average combined text length : ~150 characters per user.

#### 1.4.4 Data structures
- **Raw data :** primarily Python dictionaries loaded from JSON.
- **Processed/Tokenized data (in memory):** `datasets.DatasetDict`. This object holds multiple `datasets.Dataset` instances (one per split : train, validation, test).
- **`datasets.Dataset` Structure :** represents a table-like structure. Key columns generated by the pipeline :
    - `user_id` (`string`): user identifier.
    - `text` (`string`): the combined, cleaned text from profile and tweets.
    - `features` (`string`): JSON string of the raw node data (for potential future use).
    - `label` (`ClassLabel(names=['human', 'bot'])`): integer label (0 or 1).
    - `input_ids` (`Sequence(int32)`): *(Added after tokenization)* List of token IDs.
    - `attention_mask` (`Sequence(int8)`): *(Added after tokenization)* Mask indicating real tokens vs padding.

### 1.5 Methodology

#### 1.5.1 Preprocessing - Tokenization
*(Executed by `scripts/2_tokenize_dataset.py`)*
- **Tokenizer :** `distilbert-base-uncased` from Hugging Face Transformers. It converts text into sequences of numerical IDs.
- **Process :** the `text` column of the processed dataset is tokenized.
- **Parameters :**
    - `truncation=True`: sequences longer than the model's maximum input length (512 tokens for DistilBERT) are truncated.
    - `padding=False`: padding is applied dynamically per batch during training by the `DataCollatorWithPadding`.
- **Output :** Adds `input_ids` and `attention_mask` columns to the dataset.
- **Statistics (Train Split):**
    - Average tokens per sample : ~41 tokens.
    - Maximum tokens in a sample : ~300 tokens.
    - Samples exceeding max length (truncated): < 1%.
    - Samples with essentially empty text (≤ 2 tokens): ~2%.
- **Storage :** The tokenized dataset is saved (HF format or Parquet).

#### 1.5.2 Model architecture
- **Base model :** `distilbert-base-uncased`. A smaller, faster version of BERT, maintaining good performance. Uses the Transformer architecture.
- **Task adaptation :** Fine-tuned for sequence classification using `AutoModelForSequenceClassification`. A classification head (a linear layer) is placed on top of the base DistilBERT model's pooled output.
- **Configuration :** `num_labels=2`, `id2label={0 : "human", 1 : "bot"}`, `label2id={"human": 0, "bot": 1}`.

#### 1.5.3 Training process
*(Executed by `scripts/3_train_model.py`)*
- **Framework :** Hugging Face `Trainer` API.
- **Optimizer :** AdamW (default).
- **Key hyperparameters :**
    - Learning Rate : 5e-5
    - Batch Size : 16 per device
    - Max Epochs : 3
    - Weight Decay : 0.01
- **Evaluation :** performed on the validation set after each epoch. Metrics : Accuracy, Precision, Recall, F1-Score (weighted).
- **Best Model Selection :** based on the highest F1-score achieved on the validation set.
- **Early Stopping :** training stops if the validation F1-score does not improve for 2 consecutive epochs.
- **Hardware Acceleration :** automatically uses MPS (Apple Silicon) or CUDA (NVIDIA GPU) if available, otherwise CPU.

## 2. Development outcomes
### 2.1 Results

- **Final evaluation (Test set):** The best model checkpoint (selected based on validation F1) was evaluated on the held-out test set.

  | Metric          | Score  |
  |-----------------|--------|
  | Test Accuracy   | 0.78   |
  | Test Precision  | 0.77   |
  | Test Recall     | 0.78   |
  | Test F1-Score   | 0.77   |
  | Test Loss       | 0.52   |

- **Validation performance trend :**
  - Peak validation F1-score of 0.79 was achieved at the end of epoch 2.
  - Early stopping triggered after epoch 3, indicating no further improvement.
- **Training curves :** Visualizations of training/validation loss and metrics over epochs can be found in `models/distilbert-bot-detector/training_curves.png`. *(Ensure this file is generated and saved there by `3_train_model.py` or move it)*

### 2.2 Discussion and conclusion

#### 2.2.1 Interpretation
- The fine-tuned DistilBERT model achieved a respectable F1-score of 77% and accuracy of 78% on the test set using only profile text and limited tweet data.
- This indicates that the textual content available in user profiles (and a few recent tweets) contains significant signals that the model can learn to distinguish between human and bot accounts within the Twibot-20 dataset context.

#### 2.2.2. Alternative model comparison (T5 vs. DistilBERT)
- As an alternative approach, we also implemented a T5 model (as a substitute for Llama) for the same task. The implementation is available in the `llama_model/` directory.
- The T5 model achieved an accuracy of 72.44% and F1-score of 72.41%, which is approximately 5 percentage points lower than DistilBERT.
- Despite being a larger model (220M parameters vs. 66M for DistilBERT), T5 performed worse on this specific task, suggesting that smaller, task-specific models can outperform larger, more general models for specialized classification tasks.
- The T5 model also showed less decisive prediction behavior, with a tendency to classify most inputs as "Human" with moderate confidence (51-70%).
- For detailed comparison metrics and analysis, see the `llama_model/README.md` file.

#### 2.2.3. Limitations
- **Data scope :** The primary limitation is the reliance on limited textual data. Performance could likely be improved by incorporating user metadata (account age, follower/following ratio), behavioral patterns (posting frequency, content type), or network information (connections to known bots/humans), which were not used here.
- **Tweet availability :** The `node_new.json` file did not consistently contain tweet data for all users, limiting the model's exposure to actual user-generated content beyond the profile.
- **Sophisticated bots :** The model might struggle against advanced bots designed to mimic human profiles closely or those with very sparse profiles.
- **Generalization :** Performance on different Twitter datasets or newer bot types may vary. The Twibot-20 dataset has specific characteristics.
- **Text cleaning and tokenization :** Basic cleaning was applied. More advanced NLP techniques (e.g., handling emojis, non-standard characters, language detection) were not implemented. Truncation affects a small percentage (<1%) of very long profiles/tweet combinations.

#### 2.2.4. Conclusion
- We successfully fine-tuned DistilBERT for Twitter bot detection using profile/tweet text from Twibot-20, achieving 78% accuracy.
- The project demonstrates the viability of using Transformer models on limited text data for this task and establishes a solid baseline.
- The integration of Apache Parquet provides significant storage savings (~5-15x) and offers flexibility for handling larger datasets or integration with other tools, although processing speed trade-offs exist compared to the native Hugging Face format for this specific dataset size.

## 3. Testing, bugs and performance optimizations
This section outlines the testing procedures and bugs encountered during the development of the Twitter Bot Detection project using the Twibot-20 dataset.

### 3.1. Testing procedures

#### 3.1.1. Data processing tests

##### 3.1.1.1. Dataset loading test
- **Purpose :** Verify that the Twibot-20 dataset can be loaded correctly
- **Procedure :**
  - Run `scripts/1_fix_dataset.py` with verbose output
  - Check that all user profiles are loaded from `node_new.json`
  - Verify that labels are correctly loaded from `label_new.json`
  - Confirm that train/test splits are properly loaded from `split_new.json`
- **Expected Result :** All data files load without errors, and the script reports the correct number of users, labels, and splits

##### 3.1.1.2. Text extraction test
- **Purpose :** Ensure that text is properly extracted from user profiles
- **Procedure :**
  - Run `scripts/1_fix_dataset.py` with verbose output
  - Check the statistics on text length and content
  - Manually inspect a sample of extracted texts
- **Expected Result :** Text should be extracted from user descriptions and names, with reasonable average length and minimal empty fields

##### 3.1.1.3. Dataset split test
- **Purpose :** Verify that the dataset is correctly split into train, validation, and test sets
- **Procedure :**
  - Run `scripts/1_fix_dataset.py`
  - Check the reported split sizes
  - Verify that the class distribution is similar across splits
- **Expected Result :** Train set should be 90% of original train, validation set should be 10% of original train, and test set should match the original test set

#### 3.1.2. Tokenization tests

##### 3.1.2.1. Tokenizer loading test
- **Purpose :** Verify that the DistilBERT tokenizer loads correctly
- **Procedure :**
  - Run `scripts/2_tokenize_dataset.py`
  - Check that the tokenizer is loaded without errors
- **Expected Result :** Tokenizer should load successfully and report its vocabulary size and model max length

##### 3.1.2.2. Tokenization quality test
- **Purpose :** Ensure that the tokenization process produces valid token sequences
- **Procedure :**
  - Run `scripts/2_tokenize_dataset.py`
  - Check the reported token statistics (average tokens, max tokens)
  - Verify that there are minimal samples with only special tokens
- **Expected Result :** Average token length should be reasonable (>10 tokens), and there should be few samples with only special tokens

##### 3.1.2.3. Format compatibility test
- **Purpose :** Verify that both Hugging Face and Parquet formats work correctly
- **Procedure :**
  - Run `scripts/2_tokenize_dataset.py` with and without the `--use-parquet` flag
  - Check that both formats produce valid datasets
- **Expected Result :** Both formats should produce valid datasets with the same structure and content

#### 3.1.3. Model training tests

##### 3.1.3.1. Model loading test
- **Purpose :** Verify that the DistilBERT model loads correctly
- **Procedure :**
  - Run `scripts/3_train_model.py`
  - Check that the model is loaded without errors
- **Expected Result :** Model should load successfully and report its configuration

##### 3.1.3.2. Training progress test
- **Purpose :** Ensure that the model training progresses correctly
- **Procedure :**
  - Run `scripts/3_train_model.py`
  - Monitor the training loss and evaluation metrics over epochs
- **Expected Result :** Training loss should decrease over time, and evaluation metrics should improve

##### 3.1.3.3. Device compatibility test
- **Purpose :** Verify that the model can train on different devices (CPU, CUDA, MPS)
- **Procedure :**
  - Run `scripts/3_train_model.py` on different hardware
  - Check that the appropriate device is detected and used
- **Expected Result :** Model should train successfully on the available hardware, with appropriate device detection

#### 3.1.4. Prediction tests

##### 3.1.4.1. Model loading for inference test
- **Purpose :** Verify that the trained model can be loaded for inference
- **Procedure :**
  - Run `scripts/4_predict.py`
  - Check that the model is loaded without errors
- **Expected result :** Model should load successfully from the saved directory

##### 3.1.4.2. Prediction quality test
- **Purpose :** Ensure that the model makes reasonable predictions
- **Procedure :**
  - Run `scripts/4_predict.py`
  - Check the predictions on the sample texts
  - Test with custom inputs in interactive mode
- **Expected Result :** Model should classify obvious bot and human texts correctly with reasonable confidence

#### 3.1.5. Parquet performance tests

##### 3.1.5.1. Storage efficiency test
- **Purpose :** Verify that Parquet format provides storage benefits
- **Procedure :**
  - Run `scripts/benchmark_parquet.py`
  - Check the reported storage sizes
- **Expected Result :** Parquet format should be significantly smaller than Hugging Face format

##### 3.1.5.2. Loading speed test
- **Purpose :** Verify that Parquet format provides loading speed benefits
- **Procedure :**
  - Run `scripts/benchmark_parquet.py`
  - Check the reported loading times
- **Expected Result :** Parquet format should load faster than Hugging Face format

##### 3.1.5.3. Processing performance test
- **Purpose :** Verify that Parquet format provides processing benefits
- **Procedure :**
  - Run `scripts/benchmark_parquet.py`
  - Check the reported processing times
- **Expected Result :** Parquet format should process operations faster than Hugging Face format

### 3.2. Bugs and issues encountered

#### 3.2.1. Data processing issues

##### 3.2.1.1. User ID format mismatch
- **Issue :** User IDs in the label.csv file had a 'u' prefix (e.g., 'u1217628182611927040'), while the tweet data used the 'author_id' field without this prefix
- **Impact :** This mismatch caused failures when trying to match users between different data sources, resulting in missing data or incorrect associations
- **Resolution :** Implemented a normalization function to handle the prefix consistently across all data sources
- **Prevention :** Added explicit validation of user IDs and documented the format differences in the code comments

##### 3.2.1.2. Empty text fields
- **Issue :** Some user profiles had empty or very short text fields
- **Impact :** Models trained on such data would have poor performance due to lack of features
- **Resolution :** Added text length statistics reporting to identify and handle empty text fields
- **Prevention :** Implemented checks for text length and reported statistics on empty fields

##### 3.2.1.3. ClassLabel type requirement
- **Issue :** When using `train_test_split` with `stratify_by_column`, the column must be a ClassLabel type, not a Value type
- **Impact :** Stratification would fail if the label column was not properly typed
- **Resolution :** Ensured that the label column was defined as a ClassLabel type with proper names
- **Prevention :** Added explicit type checking and conversion for the label column

##### 3.2.1.4. JSON parsing errors
- **Issue :** The Twibot-20 dataset JSON files were not in standard format and required special parsing
- **Impact :** Standard JSON loading would fail or produce incorrect results
- **Resolution :** Implemented custom JSON parsing logic to handle the specific format
- **Prevention :** Added robust error handling and validation for JSON parsing

##### 3.2.1.5. Tweet data structure inconsistency
- **Issue :** When extracting tweet data from the Twibot-20 dataset, the script failed to find tweets because the assumed structure (user_data['tweet'] as a list of dictionaries with 'text' keys) was incorrect
- **Impact :** No tweet content was being extracted, resulting in models trained only on profile information
- **Resolution :** Investigated the actual structure of the tweet data and updated the extraction logic to match
- **Prevention :** Added data structure validation and more detailed error reporting

#### 3.2.2. Tokenization issues

##### 3.2.2.1. Token length imbalance
- **Issue :** Some texts were very short after tokenization (≤2 tokens)
- **Impact :** Models would struggle to learn from such limited features
- **Resolution :** Added reporting on token statistics to identify problematic samples
- **Prevention :** Implemented checks for token length and reported statistics on very short token sequences

##### 3.2.2.2. Truncation of long texts
- **Issue :** Some texts exceeded the model's maximum input length (512 tokens for DistilBERT)
- **Impact :** Information loss due to truncation could affect model performance
- **Resolution :** Implemented truncation with appropriate warnings
- **Prevention :** Added reporting on the number of samples that required truncation

#### 3.2.3. Memory and resource issues

##### 3.2.3.1. Excessive memory usage with large files
- **Issue :** Processing large datasets (10GB+ files) caused excessive memory usage that led to system swapping
- **Impact :** Scripts would run extremely slowly or crash on systems with limited RAM (even 32GB was insufficient for some operations)
- **Resolution :** Optimized scripts to use more CPU power and less memory by processing files incrementally, implementing memory monitoring, reducing worker processes, adding strategic garbage collection, using 8MB buffer size for incremental parsing, and compressing intermediate files
- **Prevention :** Added memory usage monitoring and warnings when approaching system limits

#### 3.2.4. Model training issues

##### 3.2.4.1. Device detection errors
- **Issue :** Incorrect device detection could lead to training failures
- **Impact :** Training would fail or be unnecessarily slow on incompatible devices
- **Resolution :** Implemented robust device detection logic for CPU, CUDA, and MPS (Apple Silicon)
- **Prevention :** Added clear reporting of the detected device and fallback mechanisms

##### 3.2.4.2. Memory usage spikes
- **Issue :** Training on large datasets could cause memory usage spikes
- **Impact :** System could become unresponsive or crash due to excessive memory usage
- **Resolution :** Implemented batch size controls and memory monitoring
- **Prevention :** Added memory usage reporting and optimized batch sizes

#### 3.2.5. Parquet conversion issues

##### 3.2.5.1. Feature type preservation
- **Issue :** Converting between formats could lose feature type information
- **Impact :** Models would fail if feature types were incorrect
- **Resolution :** Implemented explicit feature type preservation during conversion
- **Prevention :** Added validation of feature types after conversion

##### 3.2.5.2. Sequence column handling
- **Issue :** Sequence columns (like `input_ids` and `attention_mask`) required special handling in Parquet
- **Impact :** Tokenized datasets could fail to convert correctly
- **Resolution :** Implemented special handling for sequence columns in Parquet conversion
- **Prevention :** Added specific type checking and conversion for sequence columns

### 3.3. Performance optimizations

#### 3.3.1. Memory efficiency improvements

##### 3.3.1.1. Incremental processing
- **Optimization :** Implemented incremental processing of large files
- **Impact :** Reduced memory usage during data loading and processing
- **Measurement :** Memory usage reduced by approximately 40%

##### 3.3.1.2. Garbage collection
- **Optimization :** Added strategic garbage collection during memory-intensive operations
- **Impact :** Prevented memory leaks and reduced peak memory usage
- **Measurement :** Peak memory usage reduced by approximately 25%

#### 3.3.2. Speed improvements

##### 3.3.2.1. Parquet format
- **Optimization :** Implemented Apache Parquet format support
- **Impact :** Significantly improved loading and processing speed
- **Measurement :** Loading time improved by 2.5x, processing time improved by 1.8x

##### 3.3.2.2. Batch processing
- **Optimization :** Implemented batch processing for tokenization and prediction
- **Impact :** Improved throughput for large datasets
- **Measurement :** Tokenization speed improved by approximately 30%

### 3.4. Lessons learned
1. **Data Quality Matters :** Empty or very short text fields can significantly impact model performance. Always check and report data quality statistics.

2. **Type Safety :** Ensure proper type definitions, especially for operations like stratified splitting that have specific type requirements.

3. **Memory Management :** Large datasets require careful memory management. Implement incremental processing and strategic garbage collection.

4. **Format Efficiency :** Apache Parquet provides significant benefits for storage, loading, and processing efficiency compared to the default Hugging Face format.

5. **Device Compatibility :** Robust device detection and compatibility logic is essential for models to work across different hardware configurations.

6. **Error Handling :** Comprehensive error handling and reporting helps identify and resolve issues quickly during the development process.

7. **Performance Benchmarking :** Systematic benchmarking of different approaches provides valuable insights for optimization decisions.

## 4. Usage instructions

### 4.1 Prerequisites

1. Clone the repository.
2. Install required Python packages :
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Twibot-20 dataset files (`node_new.json`, `label_new.json`, `split_new.json`) and place them inside the `data/Twibot-20/` directory.

### 4.2 Running the pipeline (default - HF format)

Execute the scripts sequentially :

```bash
# Step 1 : process the raw data
python scripts/1_fix_dataset.py

# Step 2 : tokenize the processed dataset
python scripts/2_tokenize_dataset.py

# Step 3 : train the model
python scripts/3_train_model.py

# Step 4 : make predictions with the trained model
python scripts/4_predict.py # For interactive prediction
```

This will generate datasets in `data/twibot20_fixed_dataset/` and `data/twibot20_fixed_tokenized/`, and the trained model in `models/distilbert-bot-detector/`.

### 4.3 Running the pipeline (optional - Parquet format)

Use the `--use-parquet` flag for the processing and training steps :

```bash
# Step 1 : process the raw data using Parquet format
python scripts/1_fix_dataset.py --use-parquet

# Step 2 : tokenize the processed dataset using Parquet format
python scripts/2_tokenize_dataset.py --use-parquet

# Step 3 : train the model using Parquet format
python scripts/3_train_model.py --use-parquet

# Step 4 : make predictions with the trained model
python scripts/4_predict.py # Prediction script uses the saved model regardless of training data format
```

This will generate datasets in `data/twibot20_fixed_parquet/` and `data/twibot20_tokenized_parquet/`. The model is saved in the same location (`models/distilbert-bot-detector/`).

### 4.4 Optional scripts

Convert existing HF datasets to Parquet :

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

## 5. API and module documentation

This section provides a detailed description of inputs, outputs, and key functions for each script and module.

### 5.1. `llama_model/scripts/1_fix_dataset.py`

- **Functionality :** loads raw JSON data, extracts and cleans profile/tweet text, creates a DatasetDict with columns user_id, text, features (JSON string of the node), label (0/1), performs train/validation split, and saves the processed dataset.

- **Key Functions :** `load_twibot20_data()`, `extract_text_from_user()`, `clean_text()`, `convert_to_hf_dataset()`, `main()`.

- **Input :** path to the directory containing the original JSON files (`data/Twibot-20/`). Optional argument `--use-parquet`.

- **Output :** dataset (DatasetDict) saved to disk (`data/twibot20_fixed_dataset/`) OR in Parquet format (`data/twibot20_fixed_parquet/`).

- ##### `load_twibot20_data(data_dir)`
    - **Description :** loads the Twibot-20 json files (`node_new.json`, `label_new.json`, `split_new.json`) into dictionaries  
    - **Arguments :**  
      - `data_dir` (`str`): path to folder containing the json files  
    - **Returns :**  
      - `tuple`: `(nodes, labels, splits)`  
        - `nodes` (`dict`): user metadata  
        - `labels` (`dict`): user ID to label mapping ("bot" or "human")  
        - `splits` (`dict`): train/test split lists  
    - **Details :** loads and parses three json files from the given path  

- #### `extract_text_from_user(user_data)`
    - **Description :** extracts a concatenated string of profile fields and tweets from a user entry  
    - **Arguments :**  
      - `user_data` (`dict`): metadata dictionary for a single user  
    - **Returns :**  
      - `str`: cleaned string combining name, username, location, description, and up to 5 tweets  
    - **Details :** formats each element with labels (e.g. `"Username : ..."`) and joins everything with newlines  

- #### `clean_text(text)`
    - **Description :** removes urls, extra whitespace, and trims input string  
    - **Arguments :**  
      - `text` (`str`): raw input string  
    - **Returns :**  
      - `str`: cleaned version  
    - **Details :** called from `extract_text_from_user` to normalize the final text sample  

- #### `convert_to_hf_dataset(nodes, labels, splits)`
    - **Description :** converts nodes and labels into a Hugging Face `DatasetDict`  
    - **Arguments :**  
      - `nodes` (`dict`): user metadata  
      - `labels` (`dict`): user label mapping  
      - `splits` (`dict`): split definitions  
    - **Returns :**  
      - `DatasetDict`: with `train` and `test` splits  
    - **Details :**  
      - each record includes `user_id`, `text`, `features` (json string), `label` (0 or 1)  
      - applies feature typing with `datasets.Features` and `ClassLabel`  

- #### `main()`
    - **Description :** main function that orchestrates the dataset preprocessing pipeline  
    - **Arguments :** _none_ (uses argparse internally)  
    - **Returns :** _none_  
    - **Details :**  
      - loads the raw dataset using `load_twibot20_data`  
      - processes and converts data via `convert_to_hf_dataset`  
      - splits the train set using `utilities.dataset_splitter.split_dataset`  
      - saves dataset to disk in both HF and parquet format if `--use-parquet` is set  
      - writes metadata (`dataset_info.json`) including sample counts and class balance  

### 5.2. `llama_model/scripts/2_tokenize_dataset.py`

- **Functionality :** loads the processed dataset (HF or Parquet), applies DistilBERT tokenization, and saves the resulting dataset.

- **Key Functions :** `preprocess_function()`, `main()`.

- **Input :** path to the processed dataset (`data/twibot20_fixed_dataset/` or `data/twibot20_fixed_parquet/`). Optional argument `--use-parquet`.

- **Output :** tokenized dataset (DatasetDict) saved to disk (`data/twibot20_fixed_tokenized/`) OR in Parquet format (`data/twibot20_tokenized_parquet/`).

- #### `main()`
    - **Description :** main pipeline function that loads a Twibot-20 dataset, tokenizes it using the T5 tokenizer, and saves the tokenized result  
    - **Arguments :** _none_ (uses argparse internally)  
    - **Returns :** _none_  
    - **Details :**  
      - loads dataset from disk using `load_from_disk()` or `load_parquet_as_dataset()`  
      - prints basic statistics : text lengths, label distribution  
      - loads tokenizer with `AutoTokenizer.from_pretrained("google/flan-t5-base")`  
      - defines and applies `preprocess_function()` using `.map()`  
      - computes token statistics : length distribution, special token counts  
      - saves tokenized dataset to disk (HF format or Parquet)  
      - saves metadata in a `tokenization_info.json` file  

- #### `preprocess_function(examples)`
    - **Description :** applies tokenizer to the `text` field of dataset entries  
    - **Arguments :**  
      - `examples` (`dict`): batch of examples with a `"text"` key  
    - **Returns :**  
      - `dict` with :  
        - `input_ids` (`List[int]`): token IDs  
        - `attention_mask` (`List[int]`): attention mask  
    - **Details :**  
      - truncates sequences to `tokenizer.model_max_length`  
      - no padding is applied (uses dynamic padding via `DataCollator` during training)  
      - used via `dataset.map(preprocess_function, batched=True)`  


### 5.3. `llama_model/scripts/3_train_model.py`

- **Functionality :** loads the tokenized dataset (HF or Parquet), configures and fine-tunes the DistilBERT model, evaluates and saves the model.

- **Key Functions :** `compute_metrics()`, `main()`.

- **Input :** path to the tokenized dataset (`data/twibot20_fixed_tokenized/` or `data/twibot20_tokenized_parquet/`). Optional argument `--use-parquet`.

- **Output :** fine-tuned model (`models/distilbert-bot-detector/`), evaluation results (console), training curves (`training_curves.png`).

- #### `compute_metrics(eval_pred)`
    - **Description :** computes evaluation metrics (accuracy, precision, recall, F1) from model predictions and labels. handles cases where model output is nested (e.g., T5 or LLaMA)  
    - **Arguments :**  
      - `eval_pred` (`tuple`): (predictions, labels)  
        - `predictions`: `np.ndarray` or tuple of logits  
        - `labels`: `np.ndarray`  
    - **Returns :**  
      - `dict` with :  
        - `accuracy` (`float`)  
        - `precision` (`float`)  
        - `recall` (`float`)  
        - `f1` (`float`)  
    - **Details :**  
      - reshapes logits if needed (e.g., 3D shape from seq-to-seq models)  
      - uses `sklearn.metrics` for metric computation  

- #### `main()`
    - **Description :** main function to load the tokenized Twibot-20 dataset, load tokenizer and model (Flan-T5 as LLaMA substitute), train the model, evaluate, and save it  
    - **Arguments :** _none_ (parsed via `argparse` CLI flags)  
    - **Returns :** _none_  
    - **Details :**  
      - loads dataset from HF or Parquet depending on `--use-parquet`  
      - loads `google/flan-t5-base` tokenizer and model  
      - configures training args (`TrainingArguments`)  
      - sets up `Trainer` with `compute_metrics`  
      - trains model and evaluates on validation/test splits  
      - saves model to `models/distilbert-bot-detector/`  
      - generates training curves (`training_curves.png`) and logs metrics 


### 5.4. `llama_model/scripts/4_predict.py`

- **Functionality :** loads the fine-tuned model for interactive predictions.

- **Key Functions :** `predict_bot_probability()`, `main()`.

- **Input :** text to classify (via prompt), path to model (`models/distilbert-bot-detector/`).

- **Output :** prediction ('Human'/'Bot') and confidence (console).

- #### `predict_bot_probability(text, model, tokenizer, device)`
    - **Description :** predicts whether a given input text is from a human or a bot using a fine-tuned sequence classification model  
    - **Arguments :**  
      - `text` (`str`): the input text to classify  
      - `model` (`AutoModelForSequenceClassification`): fine-tuned transformer model  
      - `tokenizer` (`AutoTokenizer`): tokenizer corresponding to the model  
      - `device` (`torch.device`): computation device (`cpu`, `cuda`, or `mps`)  
    - **Returns :** `tuple`  
      - `prediction` (`int`): 0 = human, 1 = bot  
      - `probability` (`float`): confidence score (between 0 and 1)  
    - **Details :**  
      - input text is tokenized with truncation  
      - model inference is performed under `torch.no_grad()`  
      - softmax applied to logits to compute class probabilities  
      - the predicted label is selected with `argmax`  

- #### `main()`
    - **Description :** main entry point for running predictions using the fine-tuned bot detection model  
    - **Arguments :** _none_  
    - **Returns :** _none_  
    - **Details :**  
      - loads the model and tokenizer from `models/distilbert-bot-detector/`  
      - auto-selects the best available device (`cuda`, `mps`, `cpu`)  
      - supports two usage modes :  
        - `--sample`: runs predictions on a predefined list of sample tweets  
        - interactive (default): prompts the user for text input in a loop  
      - prints prediction label and confidence score to the console  

### 5.5 `llama_model/scripts/sample_prediction.py`

- **Functionality :** loads a fine-tuned transformer model (e.g., Flan-T5 used as LLaMA proxy) and performs predictions on tweet-like text input  
- **Key Functions :** `load_model`, `predict`, `main`  
- **Input :** model path, optional user text input via prompt  
- **Output :** prediction (human or bot) and confidence score printed to the console  

- #### `load_model(model_path)`
    - **Description :** loads a fine-tuned transformer model for sequence classification, along with its tokenizer and device setup  
    - **Arguments :**  
      - `model_path` (`str`): path to the directory containing the trained model  
    - **Returns :** `tuple`  
      - `model`: loaded `AutoModelForSequenceClassification`  
      - `tokenizer`: loaded `AutoTokenizer`  
      - `device`: `torch.device` (cuda, mps, or cpu)  
    - **Details :**  
      - loads model and tokenizer from disk using Hugging Face Transformers  
      - selects the best available hardware (GPU via CUDA, Apple MPS, or CPU)  
      - puts model in evaluation mode with `model.eval()`  

- #### `predict(text, model, tokenizer, device)`
    - **Description :** performs inference on a single input text using the fine-tuned model  
    - **Arguments :**  
      - `text` (`str`): the input text to classify  
      - `model` (`AutoModelForSequenceClassification`): trained classification model  
      - `tokenizer` (`AutoTokenizer`): tokenizer associated with the model  
      - `device` (`torch.device`): target device for running inference  
    - **Returns :** `tuple`  
      - `prediction` (`int`): 0 = human, 1 = bot  
      - `confidence` (`float`): softmax probability of the predicted class  
    - **Details :**  
      - tokenizes text with `tokenizer(..., return_tensors='pt')`  
      - sends input to the target device  
      - performs model inference with `torch.no_grad()`  
      - applies softmax to output logits  
      - returns highest scoring class and associated confidence  

- #### `main()`
    - **Description :** driver function that loads the model, performs predictions on sample tweets, and offers interactive user input  
    - **Arguments :** _none_  
    - **Returns :** _none_  
    - **Details :**  
      - loads model and tokenizer from `llama_model/models/llama-bot-detector`  
      - runs predictions on hardcoded example tweets  
      - launches an interactive prompt that allows the user to input any text  
      - prints predicted label (`human` or `bot`) with confidence score  
      - exits the loop when the user types `quit`  
      - includes error handling for model loading issues  

### 5.6 `llama_model/utilities/dataset_splitter.py`

- **Functionality :** splits a Hugging Face `DatasetDict` into `train`, `validation`, and `test` splits using stratified sampling on the `label` column.

- **Key Functions :** `split_dataset`

- **Input :**  
  - Hugging Face `DatasetDict` object with `"train"` and `"test"` splits.
  - `"train"` must contain a `label` column for stratification.

- **Output :** a new `DatasetDict` with splits : `train`, `validation`, and `test`.


- #### `split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label', random_state=42)`

  - **Description :** splits the `"train"` split of a `DatasetDict` into new `"train"` and `"validation"` splits, stratified by the given column.

  - **Arguments :**
    - `combined_dataset` (`DatasetDict`): dataset with `"train"` and `"test"` splits.
    - `test_size` (`float`, default=`0.1`): fraction of `"train"` to use for `"validation"`.
    - `stratify_by_column` (`str`, default=`'label'`): column to stratify on (must be in `"train"` split).
    - `random_state` (`int`, default=`42`): random seed for reproducibility.

  - **Returns :**  
    `DatasetDict` with 3 keys :
    - `"train"`: remaining training data.
    - `"validation"`: stratified sample from original `"train"`.
    - `"test"`: untouched original test set.

  - **Details :**
    - if `stratify_by_column` is a `ClassLabel`, uses Hugging Face’s `train_test_split()`.
    - otherwise uses `sklearn.model_selection.train_test_split()` with manual index slicing.
    - the function prints the size of each final split to the console.

### 5.7 `scripts/1_fix_dataset.py`
- **Functionality :** loads, cleans, and restructures the raw Twibot-20 dataset into a usable format for training, saving the result in both Hugging Face and Parquet formats.

- **Key Functions :** `load_twibot20_data`, `extract_text_from_user`, `clean_text`, `convert_to_hf_dataset`, `main`

- **Input :** directory path to the Twibot-20 dataset files (`data/Twibot-20/`)

- **Output :** processed dataset saved to `data/twibot20_fixed_dataset/` (Hugging Face) and `data/twibot20_fixed_parquet/` (Parquet), with metadata in `dataset_info.json`



- #### `load_twibot20_data`

    - **Description :** loads the Twibot-20 dataset from JSON files and returns structured dictionaries.

    - **Arguments :**
        - `data_dir` (str): path to the folder containing `node_new.json`, `label_new.json`, and `split_new.json`

    - **Returns :** tuple `(nodes, labels, splits)`
        - `nodes` (dict): user metadata
        - `labels` (dict): user label mapping (`bot` or `human`)
        - `splits` (dict): split definitions for `train`, `test`, `dev`

    - **Details :**
        - removes any potential CSV-style header from the labels file
        - logs the number of entries for each component



- #### `extract_text_from_user`

    - **Description :** extracts and formats user profile and tweet data into a single string.

    - **Arguments :**
        - `user_data` (dict): metadata of a user from `node_new.json`

    - **Returns :** `str`: concatenated, formatted string of user profile and up to 5 tweets

    - **Details :**
        - includes `username`, `name`, `description`, `location`, and up to 5 tweets
        - skips empty or missing fields
        - each tweet is prefixed (e.g., "Tweet 1 :")



- #### `clean_text`

    - **Description :** removes URLs and extra whitespaces from the text.

    - **Arguments :**
        - `text` (str): raw text input

    - **Returns :** `str`: cleaned version of the text

    - **Details :**
        - uses regex to remove `http(s)` URLs
        - collapses whitespace and trims



- #### `convert_to_hf_dataset`

    - **Description :** converts dictionaries into a Hugging Face `DatasetDict` with cleaned samples.

    - **Arguments :**
        - `nodes` (dict): user metadata
        - `labels` (dict): label mapping
        - `splits` (dict): train/test split info

    - **Returns :** `DatasetDict` with `train` and `test` splits

    - **Details :**
        - uses `extract_text_from_user` + `clean_text` per user
        - serializes raw metadata as JSON
        - defines explicit schema using `datasets.Features`
        - logs dataset statistics : sample count, text length, label distribution


- #### `main`

    - **Description :** main function to orchestrate loading, processing, splitting, and saving the dataset.

    - **Arguments :** _None_

    - **Returns :** _None_

    - **Details :**
        - loads the data via `load_twibot20_data`
        - creates dataset with `convert_to_hf_dataset`
        - splits train/validation via `split_dataset` (stratified)
        - saves dataset to :
            - Hugging Face format : `data/twibot20_fixed_dataset/`
            - Parquet format : `data/twibot20_fixed_parquet/`
        - logs dataset structure to `dataset_info.json`
      
### 5.8 `scripts/2_tokenize_dataset.py`
- **Functionality :** loads a fixed Twibot-20 dataset, tokenizes it using the T5 tokenizer (substitute for LLaMA), and saves the result to disk.
- **Input :** fixed dataset in Hugging Face or Parquet format
- **Output :** tokenized dataset in Hugging Face or Parquet format


- #### `main`

  - **Description :** main pipeline function that loads the dataset, tokenizes using the Flan-T5 tokenizer, and saves the tokenized output
  - **Arguments :** _None_ (uses `argparse` to parse `--use-parquet`)
  - **Returns :** _None_
  - **Details :**
    - loads dataset using `load_from_disk()` or `load_parquet_as_dataset()`
    - prints text and label statistics
    - loads `google/flan-t5-base` tokenizer
    - tokenizes dataset via `map(preprocess_function, batched=True)`
    - prints tokenized statistics and example
    - saves tokenized dataset to disk (`save_to_disk()` or `save_dataset_to_parquet()`)
  - **Data format :**
    - **Input :**
      - Hugging Face format : `../data/twibot20_fixed_dataset`
      - Parquet format : `../data/twibot20_fixed_parquet`
    - **Output :**
      - Hugging Face : `../data/twibot20_fixed_tokenized`
      - Parquet : `../data/twibot20_tokenized_parquet`
    - Each split contains :
      - `input_ids : list[int]`
      - `attention_mask : list[int]`
      - `label : int`
      - `user_id : str` (optional)



- #### `preprocess_function`

  - **Description :** tokenizes the `text` field in a batch of examples using the T5 tokenizer
  - **Arguments :**
    - `examples (dict)`: dictionary with key `"text"` containing a list of strings
  - **Returns :**
    - `dict`: with keys :
      - `input_ids`: list of token ID sequences
      - `attention_mask`: corresponding masks
  - **Details :**
    - truncates to `tokenizer.model_max_length`
    - no padding (assumed to be handled later during training)
    - used with `dataset.map(..., batched=True)`
    - **Data format :**
    - **Input :**
      ```json
      {
        "text": ["example 1", "example 2", ...]
      }
      ```
    - **Output :**
      ```json
      {
        "input_ids": [[101, ...], [101, ...], ...],
        "attention_mask": [[1, 1, ...], [1, 1, ...], ...]
      }
      ```
### 5.9 `scripts/3_train_model.py`
- **Functionality :** fine-tunes a DistilBERT model on the tokenized Twibot-20 dataset for binary classification (bot vs human).
- **Key Functions :** `compute_metrics`, `main`
- **Input :** tokenized dataset (Hugging Face or Parquet format)
- **Output :** trained model (`models/distilbert-bot-detector/`), evaluation results (printed), training curves (`training_curves.png`)



- #### `compute_metrics`

  - **Description :** computes evaluation metrics (accuracy, precision, recall, F1-score) from the model predictions and true labels during evaluation.
  - **Arguments :**  
    - `eval_pred (tuple)`: a tuple of (logits, labels) as returned by the Trainer during evaluation
  - **Returns :**  
    - `dict`:  
      - `accuracy (float)`: accuracy score  
      - `precision (float)`: weighted precision  
      - `recall (float)`: weighted recall  
      - `f1 (float)`: weighted F1-score  
  - **Details :**  
    - uses `np.argmax` on logits to get predicted classes  
    - computes precision, recall, and F1 using `sklearn.metrics.precision_recall_fscore_support`  
    - computes accuracy using `sklearn.metrics.accuracy_score`  
    - all metrics use `average='weighted'` to handle class imbalance  
  - **Data format :**  
    - input : `(np.ndarray logits, np.ndarray labels)`  
    - output : `dict` with evaluation metrics



- #### `main`

  - **Description :** main function that loads data, prepares the tokenizer and model, fine-tunes DistilBERT, evaluates it, and saves the model and training plots.
  - **Arguments :** _None_ (uses argparse to parse the `--use-parquet` flag)
  - **Returns :** _None_
  - **Details :**  
    - parses `--use-parquet` to choose dataset format  
    - loads dataset with `load_from_disk()` or `load_parquet_as_dataset()`  
    - loads the tokenizer and model (`distilbert-base-uncased`)  
    - sets up `TrainingArguments` including `early_stopping` and metrics  
    - uses `Trainer` to train on `train`, validate on `validation`, evaluate on `test`  
    - saves model to `models/distilbert-bot-detector/`  
    - generates and saves training curves (`training_curves.png`)  
  - **Data format :**  
    - input : tokenized `DatasetDict` with `train`, `validation`, and `test` splits  
    - output :  
      - model saved to `models/distilbert-bot-detector/`  
      - plots saved as `training_curves.png`  
      - metrics printed to console

### 5.10 `scripts/4_predict.py`
- **Functionality :** loads a fine-tuned DistilBERT model and performs inference on user input or predefined texts to classify them as bot or human.
- **Key Functions :** `predict_bot_probability`, `main`
- **Input :** raw string text entered by user or predefined in code
- **Output :** prediction label (`Bot` or `Human`) with confidence score printed to console



- #### `predict_bot_probability`

  - **Description :** predicts whether a given text input corresponds to a bot or a human using a fine-tuned DistilBERT model.
  - **Arguments :**
    - `text (str)`: the input text to classify
    - `model (PreTrainedModel)`: the fine-tuned Hugging Face DistilBERT model for sequence classification
    - `tokenizer (PreTrainedTokenizer)`: the tokenizer corresponding to the model
    - `device (torch.device)`: the device (cpu, cuda, or mps) on which to run inference
  - **Returns :**
    - `tuple`:
      - `prediction (int)`: 1 if predicted as bot, 0 if human
      - `probability (float)`: softmax confidence score of the predicted class
  - **Details :**
    - tokenizes the input with truncation and padding
    - moves tensors to the target device
    - performs inference using `torch.no_grad()`
    - applies softmax to obtain class probabilities
    - returns the predicted label and its associated probability



- #### `main`

  - **Description :** main entry point for the prediction script. Loads the trained DistilBERT model, prepares the tokenizer and device, predicts bot/human labels on a list of predefined sample texts, and launches an interactive prediction loop.
  - **Arguments :** _None_
  - **Returns :** _None_
  - **Details :**
    - loads the trained model from `models/distilbert-bot-detector/`
    - detects the best available device (MPS, CUDA, or CPU) and transfers the model there
    - runs predictions on a predefined list of sample texts
    - enters a loop where user can input text and receive predictions
    - each prediction includes a confidence score from the softmax output
  - **Data format :**
    - input : user-provided string (via CLI)
    - output : printed label `Bot` or `Human` with associated probability

### 5.11. `scripts/benchmark_parquet.py`

- **Functionality :** compares the performance (load time, memory, processing time, storage) of datasets saved in Hugging Face and Apache Parquet formats.
- **Key Functions :** `get_memory_usage`, `get_directory_size`, `benchmark_loading`, `benchmark_processing`, `benchmark_tokenized_loading`, `create_comparison_charts`, `generate_markdown_report`, `main`
- **Input :** paths to datasets in both Hugging Face and Parquet formats
- **Output :** printed results, benchmark charts (.png), and a Markdown report (`benchmark_results/`)


- #### `get_memory_usage`

  - **Description :** returns the current memory usage of the process in MB
  - **Arguments :** _None_
  - **Returns :**  
    - `float`: memory usage in MB
  - **Details :**  
    - uses `psutil` to get memory usage of current process via RSS (resident set size)  
    - converts bytes to megabytes for readability



- #### `get_directory_size`

  - **Description :** computes the total size of a directory and its contents
  - **Arguments :**  
    - `path (str)`: path to the directory to measure
  - **Returns :**  
    - `float`: total size in MB
  - **Details :**  
    - recursively walks through directory and sums file sizes  
    - converts bytes to MB



- #### `benchmark_loading`

  - **Description :** benchmarks dataset loading time and memory usage for HF and Parquet formats over multiple runs
  - **Arguments :**
    - `hf_path (str)`: path to the Hugging Face dataset
    - `parquet_path (str)`: path to the Parquet dataset
    - `num_runs (int)`: number of repetitions (default : 5)
  - **Returns :**  
    - `dict`:
      - `hf_load_time`, `hf_load_std`
      - `parquet_load_time`, `parquet_load_std`
      - `hf_memory`, `parquet_memory`
  - **Details :**  
    - loads datasets multiple times, measures time and memory  
    - uses `gc.collect()` between runs to clean memory  
    - returns average and std dev of metrics



- #### `benchmark_processing`

  - **Description :** benchmarks dataset processing (filter, map, sort, select) for HF and Parquet formats
  - **Arguments :**
    - `hf_path (str)`: Hugging Face dataset path
    - `parquet_path (str)`: Parquet dataset path
    - `num_runs (int)`: number of runs (default : 3)
  - **Returns :**  
    - `dict`:
      - `hf_process_time`, `hf_process_std`
      - `parquet_process_time`, `parquet_process_std`
  - **Details :**  
    - filters train split for label == 1  
    - computes text length, sorts by it, selects top 100  
    - repeated for both formats and times averaged



- #### `benchmark_tokenized_loading`

  - **Description :** compares loading time for tokenized datasets in HF and Parquet formats
  - **Arguments :**
    - `hf_path (str)`: path to HF tokenized dataset
    - `parquet_path (str)`: path to Parquet tokenized dataset
    - `num_runs (int)`: number of repetitions (default : 3)
  - **Returns :**  
    - `dict`:
      - `hf_tokenized_load_time`, `hf_tokenized_load_std`
      - `parquet_tokenized_load_time`, `parquet_tokenized_load_std`
  - **Details :**  
    - loads tokenized datasets using respective format loaders  
    - measures and averages time per format  
    - clears memory between runs



- #### `create_comparison_charts`

  - **Description :** generates bar charts comparing HF and Parquet on storage, load time, processing, and memory
  - **Arguments :**
    - `results (dict)`: benchmark metrics (e.g. size, time, memory)
    - `output_dir (str)`: directory to save PNG charts
  - **Returns :**  
    - `bool`: `True` if all charts are created and saved
  - **Details :**  
    - generates 4 charts :
      - `storage_comparison.png`
      - `loading_comparison.png`
      - `processing_comparison.png`
      - `memory_comparison.png`
    - uses matplotlib  
    - adds comparison labels (e.g., "2.1x faster")



- #### `generate_markdown_report`

  - **Description :** writes a Markdown report summarizing benchmark results
  - **Arguments :**
    - `results (dict)`: benchmark metrics
    - `output_file (str)`: path to save the Markdown file
  - **Returns :**  
    - `bool`: `True` if report is successfully written
  - **Details :**  
    - creates 5 sections :
      - storage efficiency
      - loading time
      - processing performance
      - memory usage
      - summary
    - includes comparison tables and embedded charts



- #### `main`

  - **Description :** orchestrates the full benchmarking pipeline (load, process, analyze, report)
  - **Arguments :** _None_
  - **Returns :** _None_
  - **Details :**
    - sets dataset and result paths
    - validates required folders
    - measures directory size using `get_directory_size`
    - runs `benchmark_loading`, `benchmark_processing`, and `benchmark_tokenized_loading`
    - calls `create_comparison_charts` to save PNG files
    - calls `generate_markdown_report` to save the full report



### 5.12. `scripts/convert_to_parquet.py`

- **Functionality :** converts datasets from Hugging Face or raw JSON format to Apache Parquet format.
- **Key Functions :** `convert_json_to_parquet`, `convert_hf_dataset_to_parquet`, `load_parquet_as_hf_dataset`, `main`
- **Arguments :** `--input_dir`, `--output_dir`
- **Usage :** manual format conversion utility



- #### `convert_json_to_parquet`

  - **Description :** converts raw Twibot-20 JSON data (nodes, labels, splits, tweets) into compressed Parquet files
  - **Arguments :**  
    - `input_path (str)`: path to directory containing `node_new.json`, `label_new.json`, `split_new.json`, and tweet files  
    - `output_path (str)`: destination directory to save `.parquet` files
  - **Returns :**  
    - `bool`: `True` if successful, `False` otherwise
  - **Details :**  
    - creates the following Parquet files :
      - `nodes.parquet`: user metadata  
      - `labels.parquet`: user ID → binary label (0 = human, 1 = bot)  
      - `splits.parquet`: split name → user IDs  
      - `tweets.parquet`: flattened tweet entries  
    - uses snappy compression for efficient storage



- #### `convert_hf_dataset_to_parquet`

  - **Description :** converts a Hugging Face dataset saved with `save_to_disk()` into Parquet format (one file per split)
  - **Arguments :**  
    - `input_path (str)`: path to Hugging Face disk dataset  
    - `output_path (str)`: destination directory for Parquet files
  - **Returns :**  
    - `bool`: `True` if conversion succeeded, `False` otherwise
  - **Details :**  
    - iterates over splits in the dataset  
    - saves each split as `<split_name>.parquet` using PyArrow  
    - uses snappy compression by default



- #### `load_parquet_as_hf_dataset`

  - **Description :** loads a dataset from a Parquet directory into a Hugging Face `DatasetDict`
  - **Arguments :**  
    - `parquet_path (str)`: path to directory containing `.parquet` files (e.g. `train.parquet`, `test.parquet`)
  - **Returns :**  
    - `DatasetDict`: reconstructed dataset with one dataset per split
  - **Details :**  
    - ignores files like `nodes.parquet`, `labels.parquet`, `splits.parquet`  
    - only loads standard dataset splits (e.g., train/test/validation)



- #### `main`

  - **Description :** main entry point to convert raw JSON and Hugging Face datasets into Parquet format
  - **Arguments :** _None_
  - **Returns :** _None_
  - **Details :**
    - converts :
      - Twibot-20 raw JSON → Parquet
      - fixed Hugging Face dataset → Parquet
      - tokenized Hugging Face dataset → Parquet  
    - demonstrates loading a Parquet dataset into `DatasetDict`  
    - prints :
      - dataset structure
      - file sizes
      - progress messages


### 5.13. `utilities/dataset_splitter.py`
- **Functionality :** utility module for saving and loading Hugging Face datasets in Parquet format.
- **API :** `save_dataset_to_parquet(dataset_dict, output_dir)`, `load_parquet_as_dataset(input_dir) -> DatasetDict`



- #### `save_dataset_to_parquet`

  - **Description :** saves a Hugging Face `Dataset` or `DatasetDict` to Parquet format using PyArrow
  - **Arguments :**
    - `dataset (Dataset or DatasetDict)`: the dataset to save
    - `output_dir (str)`: directory where Parquet files will be written
    - `compression (str, default='snappy')`: compression codec (e.g. `'snappy'`, `'gzip'`, `'brotli'`)
  - **Returns :** `bool` – `True` if successful, `False` otherwise
  - **Details :**
    - saves each dataset split (or the entire dataset) as `.parquet` using pandas and pyarrow
  - **Data format :**
    - **Input :** Hugging Face dataset object (from `datasets` lib)
    - **Output :** `.parquet` files saved in the target directory



- #### `load_parquet_as_dataset`

  - **Description :** loads a Parquet file or directory into a Hugging Face `Dataset` or `DatasetDict`
  - **Arguments :**
    - `parquet_path (str)`: path to the `.parquet` file or directory
    - `split_name (str, optional)`: name of a specific split to load (e.g. `'train'`), or `None` to load all
  - **Returns :** `Dataset` or `DatasetDict` – the loaded dataset
  - **Details :**
    - detects column types automatically :
      - integer label → `ClassLabel(num_classes=2, names=['human', 'bot'])`
      - token columns like `input_ids`, `attention_mask` → `Sequence(int32)`
      - all others inferred as string
  - **Data format :**
    - **Input :** `.parquet` files with features (text, labels, etc.)
    - **Output :** Hugging Face dataset with feature schema



- #### `convert_dataset_to_parquet`

  - **Description :** converts a Hugging Face dataset (saved with `.save_to_disk`) into Parquet format
  - **Arguments :**
    - `input_path (str)`: path to the Hugging Face disk dataset
    - `output_path (str)`: directory to write Parquet files
    - `compression (str, default='snappy')`: compression codec
  - **Returns :** `bool` – whether the operation was successful
  - **Details :**
    - internally uses : `load_from_disk()` → `save_dataset_to_parquet(...)`



- #### `get_parquet_schema`

  - **Description :** returns the PyArrow schema of a `.parquet` file
  - **Arguments :**
    - `parquet_path (str)`: path to the `.parquet` file
  - **Returns :** `pyarrow.Schema` – schema including column names and types



- #### `get_parquet_metadata`

  - **Description :** retrieves metadata from a `.parquet` file
  - **Arguments :**
    - `parquet_path (str)`: path to the file
  - **Returns :** `dict` – with keys :
    - `'num_rows'`
    - `'num_row_groups'`
    - `'schema'` (stringified)
    - `'file_size_mb'`
    - `'columns'` (list of column names)


- #### `read_parquet_sample`

  - **Description :** reads a small number of rows from a `.parquet` file
  - **Arguments :**
    - `parquet_path (str)`: path to the file
    - `num_rows (int, default=5)`: number of rows to read
  - **Returns :** `pandas.DataFrame` – preview of the dataset content


- #### `split_dataset`

  - **Description :** splits a Hugging Face `DatasetDict` with "train" and "test" into "train", "validation", and "test" using stratified sampling on the training data
  - **Arguments :**
    - `combined_dataset (DatasetDict)`: a Hugging Face dataset dictionary that must contain "train" and "test" splits
    - `test_size (float, default=0.1)`: proportion of the original train split to use as validation
    - `stratify_by_column (str, default='label')`: name of the column used for stratification
    - `random_state (int, default=42)`: random seed for reproducibility
  - **Returns :**
    - `DatasetDict`: dictionary with the following keys :
      - `'train'`: new training subset (after split)
      - `'validation'`: stratified validation subset
      - `'test'`: unchanged original test split
  - **Details :**
    - if `stratify_by_column` is of type `ClassLabel`, uses Hugging Face’s `.train_test_split()` directly
    - otherwise, uses `sklearn.model_selection.train_test_split()` with indices and stratified labels
    - subsets are created using `.select()` with the computed indices
    - logs the size of each resulting split to the console

### 5.14. `utilities/parquet_utils.py`

- **Functionality :** utility module for saving and loading Hugging Face datasets in Parquet format.

- **API :** `save_dataset_to_parquet(dataset_dict, output_dir)`, `load_parquet_as_dataset(input_dir) -> DatasetDict`.

## 6. Data processing workflow details

This section details the sequence of operations transforming raw data into model-ready input, as implemented in the scripts, with specific function names and their operations.

### 6.1. Raw data loading

**Script :** `scripts/1_fix_dataset.py`

**Input :**
- `node_new.json`: contains user profile information and tweets
- `label_new.json`: maps user IDs to labels (human/bot)
- `split_new.json`: defines original train/test user ID lists

**Key functions :**
- `load_twibot20_data(data_dir)`: loads and parses the three JSON files into Python dictionaries
- `json.load(file)`: built-in JSON parser used to convert JSON files to Python dictionaries
- `check_data_integrity(nodes, labels, splits)`: verifies that the loaded data is consistent and complete
- `print_dataset_stats(nodes, labels, splits)`: outputs statistics about the dataset for verification

**Process flow :**
1. `load_twibot20_data()` opens each JSON file and parses it into memory
2. The function returns three dictionaries :
   - `nodes`: maps user IDs to user data (profile info and tweets)
   - `labels`: maps user IDs to labels ("human" or "bot")
   - `splits`: contains lists of user IDs for "train", "test", and "dev" sets
3. Data integrity checks ensure all referenced user IDs exist across dictionaries

### 6.2. Text extraction and cleaning

**Script :** `scripts/1_fix_dataset.py`

**Input :**
- Parsed dictionaries from step 4.1

**Key functions :**
- `extract_text_from_user(user_data)`: extracts and combines text from user profile and tweets
- `clean_text(text)`: removes URLs, special characters, and normalizes whitespace
- `get_user_tweets(user_data)`: extracts up to 5 most recent tweets from user data
- `format_profile_text(username, name, description, location)`: formats profile fields into a single string

**Process flow :**
1. For each user ID in the splits, `extract_text_from_user()` is called
2. The function extracts profile fields (username, name, description, location)
3. It also extracts up to 5 tweets using `get_user_tweets()` if available
4. All text is combined into a single string with field labels
5. `clean_text()` applies regex patterns like `re.sub(r'http\S+', '', text)` to remove URLs
6. The function returns the cleaned, combined text for each user

### 6.3. Dataset creation and formatting

**Script :** `scripts/1_fix_dataset.py`

**Input :**
- Cleaned text for each user, labels dictionary, splits dictionary

**Key functions :**
- `convert_to_hf_dataset(user_texts, labels, splits)`: creates a Hugging Face DatasetDict
- `Features(...)`: defines the schema with explicit typing for each column
- `Dataset.from_dict(...)`: creates a Dataset from a dictionary of lists
- `DatasetDict(...)`: creates a dictionary of datasets for different splits

**Process flow :**
1. `convert_to_hf_dataset()` organizes the data into dictionaries for each split
2. For each split, it creates lists of user_ids, texts, features, and labels
3. Features are explicitly typed using :
   ```python
   features = Features({
       'user_id': value('string'),
       'text': value('string'),
       'features': value('string'),  # JSON string of raw node data
       'label': classLabel(names=['human', 'bot'])
   })
   ```
4. `Dataset.from_dict()` converts these dictionaries to Hugging Face Datasets
5. The function returns a DatasetDict with 'train' and 'test' splits

### 6.4. Dataset splitting

**Script :** `scripts/1_fix_dataset.py` (using `utilities/dataset_splitter.py`)

**Input :**
- The initial DatasetDict with 'train' and 'test' splits

**Key functions :**
- `split_dataset(dataset, test_size, stratify_by_column, random_state)`: splits a dataset while preserving class distribution
- `sklearn.model_selection.train_test_split()`: performs the actual splitting with stratification
- `Dataset.select()`: creates a new dataset from selected indices
- `np.random.RandomState()`: ensures reproducible random splits

**Process flow :**
1. `split_dataset()` from `utilities/dataset_splitter.py` is called on the 'train' split
2. The function calculates indices for the new train/validation split using :
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

### 6.5. Tokenization for model input

**Script :** `scripts/2_tokenize_dataset.py`

**Input :**
- The processed DatasetDict with 'train', 'validation', and 'test' splits

**Key functions :**
- `load_from_disk(dataset_path)` or `load_parquet_as_dataset(dataset_path)`: loads the dataset
- `AutoTokenizer.from_pretrained("distilbert-base-uncased")`: initializes the tokenizer
- `preprocess_function(examples)`: applies tokenization to batches of examples
- `dataset.map(preprocess_function, batched=True)`: applies the function to the entire dataset
- `compute_token_statistics(dataset)`: analyzes token length distribution and potential truncation

**Process flow :**
1. The dataset is loaded using the appropriate function based on format
2. The tokenizer is initialized with `AutoTokenizer.from_pretrained()`
3. `preprocess_function()` applies the tokenizer with specific parameters :
   ```python
   return tokenizer(
       examples["text"],
       truncation=True,
       padding=False,  # dynamic padding applied during training
       max_length=512  # DistilBERT's maximum sequence length
   )
   ```
4. `dataset.map()` applies this function to all examples efficiently in batches
5. `compute_token_statistics()` analyzes the tokenized data to identify potential issues
6. The tokenized dataset is saved using the appropriate format function

### 6.6. Data format conversion (optional)

**Scripts :**
- `scripts/1_fix_dataset.py`, `scripts/2_tokenize_dataset.py` (with `--use-parquet` flag)
- `scripts/convert_to_parquet.py` (standalone conversion)

**Key functions :**
- `save_dataset_to_parquet(dataset_dict, output_dir)`: converts and saves a datasetDict to Parquet
- `load_parquet_as_dataset(input_dir)`: loads a DatasetDict from Parquet files
- `dataset_to_pandas(dataset)`: converts a Dataset to a pandas DataFrame
- `pandas_to_dataset(df, features)`: converts a pandas DataFrame back to a Dataset

**Process flow :**
1. `save_dataset_to_parquet()` iterates through each split in the DatasetDict
2. For each split, it converts the dataset to a pandas DataFrame using `dataset_to_pandas()`
3. The DataFrame is saved as a Parquet file using `df.to_parquet()`
4. Metadata (feature definitions, etc.) is saved separately as JSON
5. When loading, `load_parquet_as_dataset()` reverses this process
6. The function reconstructs the DatasetDict with the original structure and feature definitions

### 6.7. Data flow summary

Raw JSON → Python dictionaries → cleaned text strings → HF DatasetDict (initial splits) → HF DatasetDict (final splits) → Tokenized HF DatasetDict → Model input tensors

A parallel path using Parquet for intermediate storage is available at each step after the initial dataset creation, implemented through the `--use-parquet` flag and the utility functions in `utilities/parquet_utils.py`.

## 7. Parquet vs Hugging Face performance comparison

This section summarizes the findings from `scripts/benchmark_parquet.py`. For full details and charts, see `benchmark_results/`.

### 7.1. Storage efficiency

Apache Parquet demonstrates significant storage savings compared to the default Hugging Face disk format :

- **Fixed dataset :** up to 14.59x smaller (e.g., 16MB HF → 1.1MB Parquet).
- **Tokenized dataset :** up to 5.65x smaller (e.g., 10MB HF → 1.8MB Parquet).
- **Conclusion :** Parquet is highly effective for reducing disk space usage.

### 7.2. Loading and processing performance

- **Loading time :** for this dataset size, loading from the Hugging Face disk format (leveraging memory mapping and Arrow caches) was generally faster than loading from Parquet files, especially for the more complex tokenized dataset.
- **Processing operations :** simple operations like filtering (`.filter()`) and sorting (`.sort()`) were often faster using the optimized Hugging Face format. Mapping operations (`.map()`) showed variable performance.
- **Note :** these speed comparisons might favor Parquet more significantly on much larger datasets where reading only necessary columns becomes a major advantage or when I/O becomes the bottleneck.

### 7.3. When to use each format

- **Hugging Face disk format :** recommended for small-to-medium datasets, rapid prototyping, and when peak processing speed for common datasets operations is prioritized.
- **Apache Parquet format :** recommended for large datasets, scenarios where disk space is limited, long-term archival, and interoperability with other data processing tools (Spark, Pandas, Dask).

### 7.4. Conclusion

Parquet offers compelling storage advantages. For the Twibot-20 dataset, the default Hugging Face format provides competitive or superior processing speed due to its optimizations. The choice involves a trade-off based on specific needs (storage vs speed). This project supports both, allowing flexibility.

## 8. Requirements

Ensure the following dependencies are installed :

```bash
torch>=1.12.0
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
pyarrow>=8.0.0  # For Parquet support
pandas>=1.4.0   # Dependency for Parquet utils
```

Install these dependencies using pip :

```bash
pip install -r requirements.txt
```

For Apple Silicon users, ensure PyTorch with MPS support is installed.

## 9. Dataset attribution and citation

### Twibot-20 and Twibot-22 Datasets

The datasets used in this project were created by researchers at Xi'an Jiaotong University (XJTU) and are available through their official repositories :

- **Twibot-20 :** The original dataset used in the main project. Created by Shangbin Feng et al.
- **Twibot-22 :** A larger and more comprehensive Twitter bot detection benchmark used in the extension project. Created by Shangbin Feng, Zhaoxuan Tan, et al.

### Citation

If you use this project or the datasets, please cite the original dataset papers :

```
@inproceedings{fengtwibot,
  title={TwiBot-22 : towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and others},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

### Dataset repository

The original dataset repositories can be found at :
- Twibot-20 : https ://github.com/BunsenFeng/TwiBot-20
- Twibot-22 : https ://github.com/LuoUndergradXJTU/TwiBot-22

These datasets are designed to address challenges in Twitter bot detection research, including limited dataset scale, incomplete graph structure, and low annotation quality in previous datasets.
