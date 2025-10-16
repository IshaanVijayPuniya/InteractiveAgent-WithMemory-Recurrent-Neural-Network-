import json
import numpy as np
from models.chatbot_model import ChatbotModel
from models.memory_handler import MemoryHandler
from utils.helpers import preprocess_text, load_training_data
import os

class InteractiveChatbot:
    def __init__(self):
        self.chatbot = ChatbotModel()
        self.memory = MemoryHandler()
        self.conversation_history = []
        
        # Load existing model or train new one
        if os.path.exists('models/chatbot_model.h5'):
            self.chatbot.load_model()
        else:
            print("Training initial model...")
            self.retrain_model()
    
    def get_response(self, user_input, user_id="default_user"):
        # Get context from memory
        context = self.memory.get_user_context(user_id)
        
        # Generate response
        response = self.chatbot.generate_response(user_input, context)
        
        # Store interaction
        self.memory.store_interaction(user_id, user_input, response)
        self.conversation_history.append({"user": user_input, "bot": response})
        
        return response
    
    def ask_for_feedback(self, user_input, bot_response):
        print(f"\nBot: {bot_response}")
        print("\nWas this response helpful?")
        print("1: Excellent")
        print("2: Good")
        print("3: Neutral")
        print("4: Poor")
        print("5: Very Poor")
        
        try:
            rating = input("Please rate (1-5) or press Enter to skip: ")
            if rating.strip():
                rating = int(rating)
                if 1 <= rating <= 5:
                    # Ask for correction if rating is poor
                    correction = None
                    if rating >= 4:  # Poor or Very Poor
                        correction = input("How should I have responded? (or press Enter to skip): ")
                        if correction.strip() == "":
                            correction = None
                    
                    # Store feedback
                    self.store_feedback(user_input, bot_response, rating, correction)
                    print("Thank you for your feedback!")
                    
                    # Check if we should retrain
                    if self.should_retrain():
                        print("Learning from new feedback...")
                        self.retrain_model()
        except ValueError:
            print("Invalid input. Feedback skipped.")
    
    def store_feedback(self, user_input, bot_response, rating, correction=None):
        feedback = {
            "user_input": user_input,
            "bot_response": bot_response,
            "rating": rating,
            "correction": correction,
            "timestamp": np.datetime64('now').astype(str)
        }
        
        # Load existing feedback
        try:
            with open('data/feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        feedback_data.append(feedback)
        
        # Save feedback
        with open('data/feedback_data.json', 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    def should_retrain(self):
        """Check if we have enough new feedback to retrain"""
        try:
            with open('data/feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
            
            # Retrain if we have at least 10 new feedback entries
            return len(feedback_data) % 10 == 0
        except FileNotFoundError:
            return False
    
    def retrain_model(self):
        """Retrain the model with all available data"""
        print("Retraining model with updated data...")
        
        # Load original training data
        training_data = load_training_data('data/training_data.json')
        
        # Load feedback data for additional training
        try:
            with open('data/feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
            
            # Use corrections as new training examples
            for feedback in feedback_data:
                if feedback.get('correction'):
                    training_data.append({
                        "input": feedback['user_input'],
                        "output": feedback['correction']
                    })
        except FileNotFoundError:
            pass
        
        # Train the model
        self.chatbot.train(training_data)
        print("Model retraining completed!")
    
    def save_conversation(self):
        """Save conversation history"""
        with open('data/conversations.json', 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

def main():
    chatbot = InteractiveChatbot()
    user_id = input("Enter your user ID (or press Enter for default): ").strip()
    if not user_id:
        user_id = "default_user"
    
    print("Chatbot with Memory and Learning Capability")
    print("Type 'quit' to exit, 'history' to see your conversation history")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'history':
                history = chatbot.memory.get_conversation_history(user_id)
                print("\nConversation History:")
                for i, msg in enumerate(history[-5:], 1):  # Show last 5 messages
                    print(f"{i}. You: {msg['user_input']}")
                    print(f"   Bot: {msg['bot_response']}")
                continue
            elif user_input == "":
                continue
            
            # Get bot response
            response = chatbot.get_response(user_input, user_id)
            
            # Ask for feedback
            chatbot.ask_for_feedback(user_input, response)
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        chatbot.save_conversation()

if __name__ == "__main__":
    main()
