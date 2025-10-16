import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text for training"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def load_training_data(file_path):
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Preprocess training data
        processed_data = []
        for item in data:
            processed_data.append({
                'input': preprocess_text(item['input']),
                'output': preprocess_text(item['output'])
            })
        
        return processed_data
    except FileNotFoundError:
        print(f"Training data file {file_path} not found!")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return []

def save_feedback(user_input, bot_response, rating, correction=None):
    """Save user feedback to file"""
    feedback = {
        'user_input': user_input,
        'bot_response': bot_response,
        'rating': rating,
        'correction': correction,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Load existing feedback
        try:
            with open('data/feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        # Add new feedback
        feedback_data.append(feedback)
        
        # Save back to file
        with open('data/feedback_data.json', 'w') as f:
            json.dump(feedback_data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving feedback: {e}")
