# Twibot-22 data

This directory contains the processed datasets for the Twibot-22 extension project.

## Dataset structure

- `twibot22_balanced_dataset/`: contains the balanced dataset in Hugging Face format
  - Created by the `prepare_dataset.py` script
  - Contains train/validation/test splits

- `twibot22_balanced_tokenized/`: contains the tokenized balanced dataset in Hugging Face format
  - Created by the `2_tokenize_balanced_dataset.py` script
  - Used as input for model training

- `twibot22_balanced_parquet/`: contains the balanced dataset in Parquet format
  - Optional alternative format for better performance

- `twibot22_balanced_tokenized_parquet/`: contains the tokenized balanced dataset in Parquet format
  - Optional alternative format for better performance

## Data flow

1. Raw tweets are extracted from the Twibot-22 dataset using `1_extract_tweets.py`
2. The extracted tweets are processed and split using `prepare_dataset.py`
3. The balanced dataset is tokenized using `2_tokenize_balanced_dataset.py`
4. The tokenized dataset is used for model training with `3_train_model.py`

## Note

The actual data files are not included in the repository due to their large size.
