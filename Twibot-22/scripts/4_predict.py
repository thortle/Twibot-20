"""
Make Predictions with Trained Bot Detection Model

This script loads the trained bot detection model and allows you to make predictions
on new text inputs. It provides an interactive interface for testing the model.
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_bot_probability(text, model, tokenizer, device):
    """
    Predict the probability that the given text is from a bot.
    
    Args:
        text (str): The text to classify
        model: The trained model
        tokenizer: The tokenizer
        device: The device to use for inference
        
    Returns:
        tuple: (prediction, probability) - 'human' or 'bot', and the probability
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted class and probability
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_class = "human" if predicted_class_idx == 0 else "bot"
    probability = probabilities[0][predicted_class_idx].item()
    
    return predicted_class, probability

def main():
    # Get the Twibot-22 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    twibot22_dir = os.path.dirname(script_dir)
    
    # Set model directory
    model_dir = os.path.join(twibot22_dir, "models", "bot_detection_model", "best_model")
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        print("Please train the model first using 'python Twibot-22/scripts/3_train_model.py'")
        return
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model and tokenizer
    print(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Move model to the appropriate device
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Sample texts for demonstration
    sample_texts = [
        "Just setting up my Twitter account! #excited #newuser",
        "Check out this amazing offer! Click here to get 90% off: https://bit.ly/2X3Y4Z #deal #discount #sale",
        "Had a great day at the park with my family. The weather was perfect!",
        "FOLLOW ME AND I FOLLOW BACK!!! 100% GUARANTEED!!! #TeamFollowBack #Follow4Follow",
        "Just published a new article on my blog about sustainable gardening practices. Would love your feedback!"
    ]
    
    # Make predictions on sample texts
    print("\nSample predictions:")
    for text in sample_texts:
        prediction, probability = predict_bot_probability(text, model, tokenizer, device)
        print(f"\nText: {text}")
        print(f"Prediction: {prediction.upper()} (Confidence: {probability:.2%})")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode. Enter text to classify (or 'q' to quit):")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'q':
            break
        
        prediction, probability = predict_bot_probability(text, model, tokenizer, device)
        print(f"Prediction: {prediction.upper()} (Confidence: {probability:.2%})")
    
    print("\nThank you for using the bot detection model!")

if __name__ == "__main__":
    main()
