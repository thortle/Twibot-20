"""
Make Predictions with Llama Bot Detection Model

This script loads a fine-tuned Llama model for Twitter bot detection and makes predictions on new text.
It provides an interactive interface for testing the model on custom inputs.
"""

import os
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_bot_probability(text, model, tokenizer, device):
    """
    Predict the probability that a given text is from a bot.

    Args:
        text (str): The text to classify
        model: The fine-tuned model
        tokenizer: The tokenizer for the model
        device: The device to run inference on

    Returns:
        tuple: (prediction label, probability)
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length"
    ).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get prediction and probability
    prediction = torch.argmax(probabilities, dim=-1).item()
    probability = probabilities[0][prediction].item()

    return prediction, probability

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "llama-bot-detector")

    print("[4/4] Making predictions with T5 model (as a substitute for Llama)...")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Make predictions with the Llama bot detection model")
    parser.add_argument("--sample", action="store_true", help="Run with sample inputs")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run llama_model/scripts/3_train_model.py first to train the model.")
        print("\nNote: Using Flan-T5 as a substitute for Llama due to access restrictions.")
        print("To use the actual Llama model, you need to request access on Hugging Face.")
        print("Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        return

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device for inference.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for inference.")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference.")

    model.to(device)
    model.eval()

    # Define sample inputs
    sample_inputs = [
        "Just a regular person sharing my thoughts on Twitter. Love hiking and reading!",
        "Follow me for daily updates on crypto prices! 100x gains guaranteed! #crypto #bitcoin #getrich",
        "Official account of John Smith, Professor at University of Technology. Tweets about AI and machine learning.",
        "BUY NOW!!! Limited time offer!!! Click here: https://bit.ly/2X9Y8Z7 #discount #sale #buynow",
        "Sharing my photography and travel experiences. Amateur photographer since 2010."
    ]

    # Make predictions on sample inputs
    if args.sample:
        print("\nMaking predictions on sample inputs:")
        for i, text in enumerate(sample_inputs):
            prediction, probability = predict_bot_probability(text, model, tokenizer, device)
            label = "bot" if prediction == 1 else "human"
            print(f"\nSample {i+1}:")
            print(f"Text: {text}")
            print(f"Prediction: {label.upper()} (confidence: {probability:.2%})")
        return

    # Interactive mode
    print("\nEnter text to classify (or 'q' to quit):")
    while True:
        text = input("\nText: ")
        if text.lower() == 'q':
            break

        if not text.strip():
            print("Please enter some text.")
            continue

        prediction, probability = predict_bot_probability(text, model, tokenizer, device)
        label = "bot" if prediction == 1 else "human"
        print(f"Prediction: {label.upper()} (confidence: {probability:.2%})")

if __name__ == "__main__":
    main()
