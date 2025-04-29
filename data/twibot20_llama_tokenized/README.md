# Twibot-20 Llama Tokenized Dataset (Hugging Face Format)

This directory contains the Twibot-20 dataset tokenized with an alternative tokenizer for the T5 model (used as a substitute for Llama).

## Content

When generated, this directory will contain:
- `dataset_dict.json`: Metadata about the dataset structure
- `dataset_info.json`: Information about the dataset features and statistics
- `train/`: Subdirectory containing the tokenized training split
- `validation/`: Subdirectory containing the tokenized validation split
- `test/`: Subdirectory containing the tokenized test split

Each split subdirectory contains:
- `.arrow` files: Binary Arrow format files containing the actual data
- `.idx` files: Index files for efficient access

## Dataset Structure

The dataset contains all columns from the processed dataset, plus tokenization-specific columns for the T5 model.

## Purpose

This dataset is used for training and evaluating the T5 model as an alternative to the DistilBERT model. The T5 model was used as a substitute for Llama in the project due to resource constraints.

## Note

The actual data files are not included in this repository due to their large size.
