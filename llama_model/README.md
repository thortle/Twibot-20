# Twitter Bot Detection using Llama

This directory contains scripts for training and using a Llama model for Twitter bot detection. The model is trained on the same Twibot-20 dataset as the DistilBERT model, allowing for direct comparison of performance.

## Directory Structure

```
llama_model/
├── scripts/
│   ├── 1_fix_dataset.py      # Data extraction and preprocessing
│   ├── 2_tokenize_dataset.py # Text tokenization with Llama tokenizer
│   ├── 3_train_model.py      # Llama model training
│   └── 4_predict.py          # Making predictions with Llama
│
├── utilities/
│   └── dataset_splitter.py   # Dataset splitting functionality
│
├── models/
│   └── llama-bot-detector/   # Trained Llama model files
│
└── README.md                 # This file
```

## Pipeline

The project follows the same 4-step pipeline as the DistilBERT model:

1. **Data Extraction** (`scripts/1_fix_dataset.py`): Extracts and preprocesses profile text from the Twibot-20 dataset.
2. **Tokenization** (`scripts/2_tokenize_dataset.py`): Tokenizes the text data using the Llama tokenizer.
3. **Training** (`scripts/3_train_model.py`): Fine-tunes a Llama model on the tokenized data.
4. **Prediction** (`scripts/4_predict.py`): Uses the trained model to make predictions on new data.

## Usage

### 1. Data Extraction

```bash
python llama_model/scripts/1_fix_dataset.py
```

This script extracts profile text from the Twibot-20 dataset and creates a new dataset with better text content for training.

### 2. Tokenization

```bash
python llama_model/scripts/2_tokenize_dataset.py
```

This script tokenizes the fixed dataset using the Llama tokenizer.

### 3. Training

```bash
python llama_model/scripts/3_train_model.py
```

This script fine-tunes a Llama model on the tokenized data.

### 4. Prediction

```bash
python llama_model/scripts/4_predict.py
```

This script loads the trained model and allows you to make predictions on new text data.

## Important Notes

1. **Model Access**: You need to request access to the Llama 2 model on Hugging Face before using these scripts. Visit [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) to request access.

2. **Hardware Requirements**: The Llama model is much larger than DistilBERT and requires more computational resources:
   - At least 16GB of GPU memory for training
   - Training parameters have been adjusted to work with limited resources (smaller batch size, gradient accumulation)
   - Inference is possible on CPU but will be slow

3. **Comparison with DistilBERT**: After training both models, you can compare their performance on the test set to determine which one is more effective for Twitter bot detection. See the Performance Comparison section below for detailed results.

## Requirements

```
torch>=1.12.0
transformers>=4.30.0
datasets>=2.4.0
scikit-learn>=1.0.2
numpy>=1.21.0
```

For Apple Silicon users, ensure you have PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

## Performance Comparison: T5 vs. DistilBERT

In this project, we used T5 as a substitute for Llama due to access restrictions. Below is a detailed comparison of the T5 model's performance against the DistilBERT model on the Twitter bot detection task.

### 1. Test Set Performance Metrics

| Metric          | T5 (Llama substitute) | DistilBERT |
|-----------------|----------------------|------------|
| Test Accuracy   | 0.7244               | 0.78       |
| Test Precision  | 0.7240               | 0.77       |
| Test Recall     | 0.7244               | 0.78       |
| Test F1-Score   | 0.7241               | 0.77       |
| Test Loss       | 0.5367               | 0.52       |

### 2. Model Architecture and Size

**T5 Model:**
- Architecture: Encoder-decoder transformer
- Size: ~220M parameters (T5-base)
- Model file size: ~894MB
- Designed for sequence-to-sequence tasks

**DistilBERT Model:**
- Architecture: Encoder-only transformer
- Size: ~66M parameters (3.3x smaller than T5)
- Designed specifically for classification and understanding tasks
- Knowledge distilled from BERT

### 3. Prediction Behavior

**T5 Model:**
- Tends to classify most inputs as "Human" with moderate confidence (51-70%)
- Less decisive in its predictions
- Confidence scores are generally lower and closer to 50% (random chance)
- All sample texts were classified as "Human", even those that are clearly bot-like

**DistilBERT Model:**
- Makes more varied predictions (both "Human" and "Bot")
- Higher confidence in its predictions (often >90% for bot detection)
- Correctly identified obvious bot-like texts (e.g., "FREE BITCOIN!" with 98.28% confidence)
- Shows more nuanced understanding of the differences between human and bot text

### 4. Practical Considerations

**T5 Model:**
- More complex to work with
- Requires special handling for output formats
- Slower inference time
- Less decisive predictions make it less useful for practical applications
- Larger storage and memory requirements

**DistilBERT Model:**
- Simpler architecture
- Faster inference
- More decisive and accurate predictions
- Better suited for this specific task
- Smaller storage and memory footprint

## Conclusion

The DistilBERT model significantly outperforms the T5 model for Twitter bot detection across all metrics. The performance gap of ~5 percentage points in accuracy and F1-score is substantial, especially considering that DistilBERT is a much smaller and more efficient model.

Key takeaways:

1. **Better Performance**: DistilBERT achieves higher accuracy (0.78 vs. 0.7244), precision, recall, and F1-score.

2. **Efficiency**: DistilBERT is 3.3x smaller (66M vs. 220M parameters), requiring less computational resources for both training and inference.

3. **Practical Utility**: DistilBERT makes more decisive and varied predictions, correctly identifying both humans and bots with high confidence.

4. **Task Suitability**: DistilBERT's architecture is better suited for binary classification tasks like bot detection.

For Twitter bot detection, the DistilBERT model is clearly the superior choice due to its better performance, efficiency, and practical utility. This suggests that for specialized classification tasks, smaller, task-specific models often outperform larger, more general models.

Note: While we used T5 as a substitute for Llama, similar conclusions would likely apply to Llama as well, as both are large, general-purpose models that may not be optimized for binary classification tasks like bot detection.
