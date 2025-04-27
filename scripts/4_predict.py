"""
Make Predictions with Trained Bot Detection Model

This script loads a fine-tuned DistilBERT model and uses it to make predictio€ns
on new text data to detect whether the account is a bot or human.
"""

import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_bot_probability(text, model, tokenizer, device):
    """
    Predict the probability that the given text is from a bot.

    Args:
        text (str): The text to classify
        model: The trained model
        tokenizer: The tokenizer
        device: The device to run inference on

    Returns:
        tuple: (prediction, probability) where prediction is 0 (human) or 1 (bot)
               and probability is the confidence score
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    prediction = np.argmax(probabilities)

    return prediction, probabilities[prediction]

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "distilbert-bot-detector")

    print("[4/4] Making predictions...")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run scripts/3_train_model.py first to train the model.")
        return

    # 1. Load model and tokenizer
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # 2. Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device for inference")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for inference")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")

    # Move model to device
    model.to(device)

    # 3. Make predictions on sample texts
    print("\nMaking predictions on sample texts...")

    sample_texts = [
        "Just posted a photo https://t.co/UyGsGOblh5",
        "Check out my new blog post! https://t.co/abc123 #marketing #socialmedia",
        "I'm so excited to announce that I'll be speaking at the conference next month! Can't wait to share my insights on AI and machine learning.",
        "FREE BITCOIN! Click here to claim your free Bitcoin now! Limited time offer! https://t.co/scam",
        "Good morning everyone! Hope you all have a wonderful day ahead. I'm planning to go hiking this weekend if the weather permits.",
    ]

    for i, text in enumerate(sample_texts):
        prediction, probability = predict_bot_probability(text, model, tokenizer, device)
        label = "Bot" if prediction == 1 else "Human"
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"Prediction: {label} (confidence: {probability:.4f})")

    # 4. Interactive mode
    print("\n" + "="*50)
    print("Interactive Bot Detection Mode")
    print("Enter text to classify (or 'quit' to exit)")
    print("="*50)

    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        prediction, probability = predict_bot_probability(user_input, model, tokenizer, device)
        label = "Bot" if prediction == 1 else "Human"
        print(f"Prediction: {label} (confidence: {probability:.4f})")

    print("\nThank you for using the Bot Detection model!")

if __name__ == "__main__":
    main()
