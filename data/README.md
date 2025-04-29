# Twibot-20 Data Directory

This directory contains the datasets for the Twitter Bot Detection project.

## Directory Structure

- `Twibot-20/`: Original dataset files (node_new.json, label_new.json, split_new.json)
- `twibot20_fixed_dataset/`: Processed dataset in Hugging Face format
- `twibot20_fixed_tokenized/`: Tokenized dataset in Hugging Face format
- `twibot20_llama_tokenized/`: Alternative tokenized dataset for T5 model
- `twibot20_fixed_parquet/`: Processed dataset in Parquet format (optional)
- `twibot20_tokenized_parquet/`: Tokenized dataset in Parquet format (optional)

## Data Flow

1. Raw data from `Twibot-20/` is processed by `scripts/1_fix_dataset.py`
2. Processed data is saved to `twibot20_fixed_dataset/` (or `twibot20_fixed_parquet/` if using Parquet)
3. Processed data is tokenized by `scripts/2_tokenize_dataset.py`
4. Tokenized data is saved to `twibot20_fixed_tokenized/` (or `twibot20_tokenized_parquet/` if using Parquet)
5. Tokenized data is used for model training by `scripts/3_train_model.py`

## Note

The actual data files are not included in the repository due to their large size. Please download the Twibot-20 dataset and place the files in the appropriate directories as described in the main README.
