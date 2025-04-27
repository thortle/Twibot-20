# DistilBERT Bot Detector

This directory contains a fine-tuned DistilBERT model for Twitter bot detection.

## Model Details

- **Base Model**: `distilbert-base-uncased`
- **Task**: Sequence Classification (Binary: Human vs Bot)
- **Dataset**: Twibot-20
- **Training Data**: 7,450 samples (56.1% bots, 43.9% humans)
- **Validation Data**: 828 samples (56.0% bots, 44.0% humans)
- **Test Data**: 1,183 samples (54.1% bots, 45.9% humans)

## Performance

| Metric          | Score  |
|-----------------|--------|
| Test Accuracy   | 0.78   |
| Test Precision  | 0.77   |
| Test Recall     | 0.78   |
| Test F1-Score   | 0.77   |
| Test Loss       | 0.52   |

## Files

This directory contains the following files:

- `config.json`: Model configuration
- `special_tokens_map.json`: Special tokens mapping
- `tokenizer_config.json`: Tokenizer configuration
- `tokenizer.json`: Tokenizer data
- `vocab.txt`: Vocabulary file

**Note**: The large model file (`model.safetensors`, ~268MB) is not included in this repository due to size constraints.

## How to Use

To use this model for predictions:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model_path = "models/distilbert-bot-detector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Prepare input text
text = "Username: bot123 Name: Free Bitcoin Description: Get free Bitcoin now! Click the link in my bio!"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Make prediction
import torch
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
# Get prediction (0 = human, 1 = bot)
predicted_class = torch.argmax(predictions, dim=1).item()
confidence = predictions[0][predicted_class].item()

print(f"Prediction: {'Bot' if predicted_class == 1 else 'Human'}")
print(f"Confidence: {confidence:.4f}")
```

## Training

This model was trained using the following hyperparameters:

- Learning rate: 5e-5
- Batch size: 16 per device
- Maximum epochs: 3 (with early stopping)
- Weight decay: 0.01
- Optimizer: AdamW

To retrain this model, run:

```bash
python scripts/3_train_model.py
```
