"""
Sample Prediction Script for Llama Bot Detection Model

This script demonstrates how to use the trained Llama model for Twitter bot detection.
It loads the model and makes predictions on sample texts.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 
                         "cpu")
    model.to(device)
    
    print(f"Model loaded successfully and moved to {device}")
    
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
    """
    Make a prediction on a single text.
    
    Args:
        text (str): The text to classify
        model: The trained model
        tokenizer: The tokenizer
        device: The device to run inference on
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get prediction and confidence
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def main():
    # Path to the trained model
    model_path = "llama_model/models/llama-bot-detector"
    
    # Sample texts
    sample_texts = [
        "Just posted a photo https://t.co/UyGsGOblh5",
        "Check out my new blog post! https://t.co/abc123 #marketing #socialmedia",
        "I'm so excited to announce that I'll be speaking at the conference next month! Can't wait to share my insights on AI and machine learning.",
        "FREE BITCOIN! Click here to claim your free Bitcoin now! Limited time offer! https://t.co/scam",
        "Good morning everyone! Hope you all have a wonderful day ahead. I'm planning to go hiking this weekend if the weather permits."
    ]
    
    try:
        # Load model
        model, tokenizer, device = load_model(model_path)
        
        # Make predictions on sample texts
        print("\nMaking predictions on sample texts...\n")
        for i, text in enumerate(sample_texts, 1):
            prediction, confidence = predict(text, model, tokenizer, device)
            label = "Bot" if prediction == 1 else "Human"
            print(f"Sample {i}:")
            print(f"Text: {text}")
            print(f"Prediction: {label} (confidence: {confidence:.4f})")
            print()
            
        # Interactive mode
        print("=" * 50)
        print("Interactive Bot Detection Mode")
        print("Enter text to classify (or 'quit' to exit)")
        print("=" * 50)
        print()
        
        while True:
            user_input = input("Enter text: ")
            if user_input.lower() == 'quit':
                break
                
            prediction, confidence = predict(user_input, model, tokenizer, device)
            label = "Bot" if prediction == 1 else "Human"
            print(f"Prediction: {label} (confidence: {confidence:.4f})")
            print()
            
        print("\nThank you for using the Bot Detection model!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This script requires a trained model. Please run the training script first.")
        print("If you don't have access to the Llama model, you can use the DistilBERT model instead:")
        print("python scripts/4_predict.py --sample")

if __name__ == "__main__":
    main()
