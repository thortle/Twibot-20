# Twitter Bot Detection using Profile Text and DistilBERT

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
│   └── 4_predict.py              # Making predictions
│
├── utilities/                    # Helper modules
│   └── dataset_splitter.py       # Dataset splitting functionality
│
├── models/                       # Trained models
│   └── distilbert-bot-detector/  # Trained model files
│
└── README.md                     # This file
```

## 3. Pipeline

The project follows a 4-step pipeline:

1. **Data Extraction** (`scripts/1_fix_dataset.py`): Extracts and preprocesses profile text from the Twibot-20 dataset.
2. **Tokenization** (`scripts/2_tokenize_dataset.py`): Tokenizes the text data using the DistilBERT tokenizer.
3. **Training** (`scripts/3_train_model.py`): Fine-tunes a DistilBERT model on the tokenized data.
4. **Prediction** (`scripts/4_predict.py`): Uses the trained model to make predictions on new data.

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
  - `data/twibot20_fixed_tokenized/`: Contains the tokenized dataset ready for model training
    - `dataset_dict.json`: Stores tokenized dataset configuration
    - Separate folders for each tokenized split (train/validation/test)
- **Preparation Steps:**
  - Raw JSON data is loaded from the three files
  - Text is extracted by combining username, name, description, location, and up to 5 tweets
  - The extracted text is cleaned by removing URLs and extra whitespace
  - The data is converted to Hugging Face `DatasetDict` format
  - The train set is split into train and validation sets (90/10 ratio) with stratification by label
  - Final dataset statistics:
    - Train: 7,450 samples (56.1% bots, 43.9% humans)
    - Validation: 828 samples (56.0% bots, 44.0% humans)
    - Test: 1,183 samples (54.1% bots, 45.9% humans)
    - Average text length: ~150 characters per user

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
  - The tokenized dataset is saved as `twibot20_fixed_tokenized`

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

This script extracts profile text from the Twibot-20 dataset and creates a new dataset with better text content for training.

### 2. Tokenization

```bash
python scripts/2_tokenize_dataset.py
```

This script tokenizes the fixed dataset using the DistilBERT tokenizer.

### 3. Training

```bash
python scripts/3_train_model.py
```

This script fine-tunes a DistilBERT model on the tokenized data.

### 4. Prediction

```bash
python scripts/4_predict.py
```

This script loads the trained model and allows you to make predictions on new text data.

## 9. Script Documentation

Each script in this project serves a specific purpose in the data processing and model training pipeline:

### utilities/dataset_splitter.py
- **Purpose**: Splits datasets into train, validation, and test sets with stratification
- **Input**: A Hugging Face DatasetDict with 'train' and 'test' splits
- **Output**: A DatasetDict with 'train', 'validation', and 'test' splits
- **When to use**: Used by 1_fix_dataset.py to create validation splits

### scripts/1_fix_dataset.py
- **Purpose**: Processes raw Twibot-20 data files and extracts text from user profiles and tweets
- **Input**: Raw JSON files from the Twibot-20 dataset
- **Output**: Creates the `twibot20_fixed_dataset` directory with processed data
- **Key features**:
  - Extracts username, name, description, location and up to 5 tweets when available
  - Cleans text by removing URLs and extra whitespace
  - Uses dataset_splitter.py to create train/validation/test splits
- **Next step**: Run 2_tokenize_dataset.py after this script

### scripts/2_tokenize_dataset.py
- **Purpose**: Tokenizes the processed dataset for input to the DistilBERT model
- **Input**: The fixed dataset from `twibot20_fixed_dataset` directory
- **Output**: Creates the `twibot20_fixed_tokenized` directory with tokenized data
- **Key features**:
  - Uses DistilBERT tokenizer to process the text
  - Provides detailed tokenization statistics
  - Shows examples of tokenized output
- **Next step**: Run 3_train_model.py after this script

### scripts/3_train_model.py
- **Purpose**: Fine-tunes a DistilBERT model on the tokenized data for bot detection
- **Input**: The tokenized dataset from `twibot20_fixed_tokenized` directory
- **Output**: Trained model saved to the `models/distilbert-bot-detector` directory
- **Key features**:
  - Supports MPS for Apple Silicon or CUDA for NVIDIA GPUs
  - Implements early stopping to prevent overfitting
  - Evaluates on test set after training
- **Next step**: Use the trained model for predictions

### scripts/4_predict.py
- **Purpose**: Uses the trained model to make predictions on new text inputs
- **Input**: The trained model from `models/distilbert-bot-detector` directory
- **Output**: Bot/human predictions for provided text inputs
- **Usage**: Provides an interactive interface and sample predictions
- **When to use**: After training the model to make predictions on new data

## 10. Requirements

```
torch>=1.12.0
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
ijson>=3.1.4
```

For Apple Silicon users, ensure you have PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```
