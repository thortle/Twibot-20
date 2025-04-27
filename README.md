# Twitter Bot Detection using Profile Text and DistilBERT

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

- **Objective:** This project aims to build a machine learning model to classify Twitter accounts as either 'human' or 'bot' using the Twibot-20 dataset.
- **Approach:** We fine-tuned a pre-trained DistilBERT model for sequence classification using the Hugging Face Transformers library. The project supports both the standard Hugging Face dataset format and the efficient Apache Parquet format.
- **Data Focus:** Classification is based on user profile text (username, name, description, location) and up to 5 recent tweets when available in the source data.

## 2. Project Structure
/
├── scripts/ # Main pipeline scripts
│ ├── 1_fix_dataset.py # Data extraction and preprocessing
│ ├── 2_tokenize_dataset.py # Text tokenization
│ ├── 3_train_model.py # Model training
│ ├── 4_predict.py # Making predictions
│ ├── convert_to_parquet.py # Convert datasets to Parquet format
│ └── benchmark_parquet.py # Benchmark Parquet vs Hugging Face performance
│
├── utilities/ # Helper modules

│ ├── dataset_splitter.py # Dataset splitting functionality
│ ├── parquet_utils.py # Apache Parquet utilities
│ └── README_PARQUET.md # Documentation for Parquet implementation
│
├── models/ # Trained models
│ └── distilbert-bot-detector/ # Trained DistilBERT model files
│
├── data/ # Datasets (original and generated)
│ ├── Twibot-20/ # Original dataset files (place node_new.json, etc. here)
│ ├── twibot20_fixed_dataset/ # (Generated - HF Format)
│ ├── twibot20_fixed_tokenized/ # (Generated - HF Format)
│ ├── twibot20_llama_tokenized/ # (Generated - Alternative tokenizer for T5 model)
│ ├── twibot20_fixed_parquet/ # (Optional - Parquet Format)
│ ├── twibot20_parquet/ # (Optional - Intermediate Parquet storage)
│ └── twibot20_tokenized_parquet/ # (Optional - Parquet Format)
│
├── benchmark_results/ # Performance comparison results
│ ├── storage_comparison.png # Storage efficiency charts
│ ├── loading_comparison.png # Loading time charts
│ ├── processing_comparison.png # Processing time charts
│ ├── memory_comparison.png # Memory usage charts
│ └── parquet_performance.md # Detailed benchmark report
│
├── README.md # This file



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

### 7.2. Limitations
- **Data Scope:** The primary limitation is the reliance on limited textual data. Performance could likely be improved by incorporating user metadata (account age, follower/following ratio), behavioral patterns (posting frequency, content type), or network information (connections to known bots/humans), which were not used here.
- **Tweet Availability:** The `node_new.json` file did not consistently contain tweet data for all users, limiting the model's exposure to actual user-generated content beyond the profile.
- **Sophisticated Bots:** The model might struggle against advanced bots designed to mimic human profiles closely or those with very sparse profiles.
- **Generalization:** Performance on different Twitter datasets or newer bot types may vary. The Twibot-20 dataset has specific characteristics.
- **Text Cleaning & Tokenization:** Basic cleaning was applied. More advanced NLP techniques (e.g., handling emojis, non-standard characters, language detection) were not implemented. Truncation affects a small percentage (<1%) of very long profiles/tweet combinations.

### 7.3. Conclusion
- We successfully fine-tuned DistilBERT for Twitter bot detection using profile/tweet text from Twibot-20, achieving 78% accuracy.
- The project demonstrates the viability of using Transformer models on limited text data for this task and establishes a solid baseline.
- The integration of Apache Parquet provides significant storage savings (~5-15x) and offers flexibility for handling larger datasets or integration with other tools, although processing speed trade-offs exist compared to the native Hugging Face format for this specific dataset size.

## 8. Usage Instructions

### 8.1. Prerequisites
1.  Clone the repository.
2.  Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the Twibot-20 dataset files (`node_new.json`, `label_new.json`, `split_new.json`) and place them inside the `data/Twibot-20/` directory.

### 8.2. Running the Pipeline (Default - HF Format)
Execute the scripts sequentially:
```bash
python scripts/1_fix_dataset.py
python scripts/2_tokenize_dataset.py
python scripts/3_train_model.py
python scripts/4_predict.py # For interactive prediction
```

This will generate datasets in data/twibot20_fixed_dataset/ and data/twibot20_fixed_tokenized/, and the trained model in models/distilbert-bot-detector/.

### 8.3. Running the Pipeline (Optional - Parquet Format)

Use the --use-parquet flag for the processing and training steps:

```bash
python scripts/1_fix_dataset.py --use-parquet
python scripts/2_tokenize_dataset.py --use-parquet
python scripts/3_train_model.py --use-parquet
python scripts/4_predict.py # Prediction script uses the saved model regardless of training data format
```

This will generate datasets in data/twibot20_fixed_parquet/ and data/twibot20_tokenized_parquet/. The model is saved in the same location (models/distilbert-bot-detector/).

### 8.4. Optional Scripts

Convert existing HF datasets to Parquet:

```bash
python scripts/convert_to_parquet.py --input_dir data/twibot20_fixed_dataset --output_dir data/twibot20_fixed_parquet
python scripts/convert_to_parquet.py --input_dir data/twibot20_fixed_tokenized --output_dir data/twibot20_tokenized_parquet
```

Run performance benchmarks (requires datasets in both formats):

```bash
python scripts/benchmark_parquet.py
```

(Results appear in console and benchmark_results/)

9. API and Module Documentation

(Detailed description of inputs, outputs, and key functions for each script/module)
9.1. scripts/1_fix_dataset.py

    Fonctionnalité: Charge les données brutes JSON, extrait et nettoie le texte des profils/tweets, crée un DatasetDict avec les colonnes user_id, text, features (JSON string du node), label (0/1), effectue la division train/validation et sauvegarde le dataset traité.

    Fonctions Clés: load_twibot20_data, extract_text_from_user, clean_text, convert_to_hf_dataset, main.

    Entrée: Chemin vers le dossier contenant les fichiers JSON originaux (data/Twibot-20/). Argument optionnel --use-parquet.

    Sortie: Dataset (DatasetDict) sauvegardé sur disque (data/twibot20_fixed_dataset/) OU au format Parquet (data/twibot20_fixed_parquet/).

9.2. scripts/2_tokenize_dataset.py

    Fonctionnalité: Charge le dataset traité (HF ou Parquet), applique la tokenisation DistilBERT et sauvegarde le dataset résultant.

    Fonctions Clés: preprocess_function, main.

    Entrée: Chemin vers le dataset traité (data/twibot20_fixed_dataset/ ou data/twibot20_fixed_parquet/). Argument optionnel --use-parquet.

    Sortie: Dataset tokenisé (DatasetDict) sauvegardé sur disque (data/twibot20_fixed_tokenized/) OU au format Parquet (data/twibot20_tokenized_parquet/).

9.3. scripts/3_train_model.py

    Fonctionnalité: Charge le dataset tokenisé (HF ou Parquet), configure et fine-tune le modèle DistilBERT, évalue et sauvegarde le modèle.

    Fonctions Clés: compute_metrics, main.

    Entrée: Chemin vers le dataset tokenisé (data/twibot20_fixed_tokenized/ ou data/twibot20_tokenized_parquet/). Argument optionnel --use-parquet.

    Sortie: Modèle fine-tuné (models/distilbert-bot-detector/), résultats d'évaluation (console), courbes d'entraînement (training_curves.png).

9.4. scripts/4_predict.py

    Fonctionnalité: Charge le modèle fine-tuné pour prédictions interactives.

    Fonctions Clés: predict_bot_probability, main.

    Entrée: Texte à classifier (via invite), chemin vers modèle (models/distilbert-bot-detector/).

    Sortie: Prédiction ('Human'/'Bot') et confiance (console).

9.5. scripts/convert_to_parquet.py

    Fonctionnalité: Convertit un dataset format disque HF en format Parquet.

    Arguments: --input_dir, --output_dir.

    Utilisation: Outil manuel pour la conversion de format.

9.6. scripts/benchmark_parquet.py

    Fonctionnalité: Compare les performances des formats HF disque et Parquet.

    Entrée: Chemins vers les datasets dans les deux formats.

    Sortie: Résultats (console), rapport markdown et graphiques (benchmark_results/).

9.7. utilities/dataset_splitter.py

    Fonctionnalité: Module utilitaire pour diviser un dataset HF de manière stratifiée.

    API: split_dataset(combined_dataset, test_size=0.1, stratify_by_column='label', random_state=42) -> DatasetDict.

9.8. utilities/parquet_utils.py

    Fonctionnalité: Module utilitaire pour la sauvegarde et le chargement de datasets HF au format Parquet.

    API: save_dataset_to_parquet(dataset_dict, output_dir), load_dataset_from_parquet(input_dir) -> DatasetDict.

10. Data Processing Workflow Details

(This section details the sequence of operations transforming raw data into model-ready input, as implemented in the scripts)
10.1. Raw Data Loading

    Script: scripts/1_fix_dataset.py

    Input: node_new.json, label_new.json, split_new.json

    Process: JSON files are parsed into Python dictionaries (nodes, labels, splits).

10.2. Text Extraction and Cleaning

    Script: scripts/1_fix_dataset.py

    Input: nodes dictionary.

    Process: For each user ID present in the splits, the extract_text_from_user function combines profile fields (Username, Name, Description, Location) and up to 5 tweets into a single string. The clean_text function then removes URLs and normalizes whitespace.

10.3. Dataset Creation and Formatting

    Script: scripts/1_fix_dataset.py

    Input: Cleaned text, labels dictionary, splits dictionary.

    Process: The convert_to_hf_dataset function organizes the data into a DatasetDict with 'train' and 'test' splits. Each entry includes user_id, the cleaned text, the raw features (as JSON string), and the integer label (0/1). Features are explicitly typed (Value, ClassLabel).

10.4. Dataset Splitting

    Script: scripts/1_fix_dataset.py (using utilities/dataset_splitter.py)

    Input: The initial DatasetDict created in step 10.3.

    Process: The split_dataset function takes the 'train' split and divides it stratigraphically by 'label' into a new 'train' split (90%) and a 'validation' split (10%). The original 'test' split is retained.

10.5. Tokenization for Model Input

    Script: scripts/2_tokenize_dataset.py

    Input: The final processed DatasetDict (from HF disk or Parquet).

    Process: The preprocess_function applies the distilbert-base-uncased tokenizer to the text column. It generates input_ids and attention_mask for each sample, truncating sequences longer than 512 tokens. This tokenized data is saved.

10.6. Data Format Conversion (Optional)

    Scripts: scripts/1_fix_dataset.py, scripts/2_tokenize_dataset.py (with --use-parquet), scripts/convert_to_parquet.py

    Input/Output: DatasetDict objects.

    Process: Uses functions from utilities/parquet_utils.py to serialize DatasetDict objects into Parquet files or deserialize them back into DatasetDict objects.

10.7. Data Flow Summary

Raw JSON -> Dicts -> Cleaned Text Strings -> HF DatasetDict (Initial Splits) -> HF DatasetDict (Final Splits) -> Tokenized HF DatasetDict -> Model Input Tensors.
(Parallel path using Parquet for intermediate storage is available)
11. Parquet vs Hugging Face Performance Comparison

(This section summarizes the findings from scripts/benchmark_parquet.py. For full details and charts, see benchmark_results/)
11.1. Storage Efficiency

Apache Parquet demonstrates significant storage savings compared to the default Hugging Face disk format:

    Fixed Dataset: Up to 14.59x smaller (e.g., 16MB HF -> 1.1MB Parquet).

    Tokenized Dataset: Up to 5.65x smaller (e.g., 10MB HF -> 1.8MB Parquet).
    Conclusion: Parquet is highly effective for reducing disk space usage.

11.2. Loading and Processing Performance

    Loading Time: For this dataset size, loading from the Hugging Face disk format (leveraging memory mapping and Arrow caches) was generally faster than loading from Parquet files, especially for the more complex tokenized dataset.

    Processing Operations: Simple operations like filtering (.filter()) and sorting (.sort()) were often faster using the optimized Hugging Face format. Mapping operations (.map()) showed variable performance.
    Note: These speed comparisons might favor Parquet more significantly on much larger datasets where reading only necessary columns becomes a major advantage or when I/O becomes the bottleneck.

11.3. When to Use Each Format

    Hugging Face Disk Format: Recommended for small-to-medium datasets, rapid prototyping, and when peak processing speed for common datasets operations is prioritized.

    Apache Parquet Format: Recommended for large datasets, scenarios where disk space is limited, long-term archival, and interoperability with other data processing tools (Spark, Pandas, Dask).

11.4. Conclusion

Parquet offers compelling storage advantages. For the Twibot-20 dataset, the default Hugging Face format provides competitive or superior processing speed due to its optimizations. The choice involves a trade-off based on specific needs (storage vs speed). This project supports both, allowing flexibility.
12. Requirements

Ensure the following dependencies are installed:

```
torch>=1.12.0
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
pyarrow>=8.0.0  # For Parquet support
pandas>=1.4.0   # Dependency for Parquet utils
```

Install these dependencies using pip.

For Apple Silicon users, ensure PyTorch with MPS support is installed.

