# Trained Models

This directory contains the trained models for Twitter bot detection.

## Directory Structure

```
models/
├── distilbert-bot-detector/  # DistilBERT model trained on Twibot-20 dataset
└── README.md                 # This file
```

## Models

### DistilBERT Bot Detector

The `distilbert-bot-detector` directory contains a fine-tuned DistilBERT model for Twitter bot detection. The model was trained on the Twibot-20 dataset and achieves the following performance on the test set:

| Metric          | Score  |
|-----------------|--------|
| Test Accuracy   | 0.78   |
| Test Precision  | 0.77   |
| Test Recall     | 0.78   |
| Test F1-Score   | 0.77   |
| Test Loss       | 0.52   |

**Note**: The large model files (*.safetensors, *.bin) are not included in this repository due to size constraints. Only the configuration files and tokenizer files are included.

## How to Use

To use these models, you need to:

1. Clone this repository
2. Run the training script to generate the model files:
   ```bash
   python scripts/3_train_model.py
   ```
3. Or download the pre-trained model files from the releases section (if available)
4. Use the prediction script to make predictions:
   ```bash
   python scripts/4_predict.py
   ```

## Model Configuration

The model configuration files (config.json, tokenizer_config.json, etc.) are included in this repository. These files contain the model architecture and tokenizer settings needed to recreate the model.
