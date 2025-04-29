# Twibot-20 Original Dataset

This directory should contain the original Twibot-20 dataset files:

- `node_new.json`: Contains user profile information and potentially tweets
  ```json
  { "u17461978": { "description": "...", "name": "SHAQ", ... }, ... }
  ```

- `label_new.json`: Maps user_id (str) to label ('human' or 'bot')
  ```json
  { "u17461978": "human", "u1297437077403885568": "bot", ... }
  ```

- `split_new.json`: Defines original train/test user ID lists
  ```json
  { "train": ["u17461978", ...], "test": [...], "dev": [...] }
  ```

## Source

The Twibot-20 dataset was created by researchers at Xi'an Jiaotong University (XJTU) and is available through their official repository:
https://github.com/BunsenFeng/TwiBot-20

## Usage

Place the original dataset files in this directory before running the pipeline scripts. The first script (`scripts/1_fix_dataset.py`) will load these files and process them.

## Note

The actual data files are not included in this repository due to their large size. Please download them from the original source.
