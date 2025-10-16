from models.chatbot_model import ChatbotModel
from utils.helpers import load_training_data
import json

def train_initial_model():
    """Train the initial chatbot model"""
    print("Training initial chatbot model...")
    
    # Load training data
    training_data = load_training_data('data/training_data.json')
    
    # Create and train model
    chatbot = ChatbotModel()
    chatbot.train(training_data)
    
    print("Initial model training completed!")
    print(f"Trained on {len(training_data)} examples")

if __name__ == "__main__":
    train_initial_model()
