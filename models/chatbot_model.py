import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import os

class ChatbotModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 50
        self.vocab_size = 10000
        self.embedding_dim = 100
        
    def build_model(self):
        """Build the neural network model"""
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, training_data):
        """Prepare training data for the model"""
        # Extract inputs and outputs
        inputs = [item['input'] for item in training_data]
        outputs = [item['output'] for item in training_data]
        
        # Create and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(inputs + outputs)
        
        # Convert texts to sequences
        input_sequences = self.tokenizer.texts_to_sequences(inputs)
        output_sequences = self.tokenizer.texts_to_sequences(outputs)
        
        # Pad sequences
        X = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='post')
        y = pad_sequences(output_sequences, maxlen=self.max_sequence_length, padding='post')
        
        return X, y
    
    def train(self, training_data):
        """Train the model on provided data"""
        if not training_data:
            print("No training data provided!")
            return
        
        # Prepare data
        X, y = self.prepare_data(training_data)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train the model
        history = self.model.fit(
            X, y,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model and tokenizer
        self.save_model()
        
        return history
    
    def generate_response(self, user_input, context=None):
        """Generate response for user input"""
        if self.model is None or self.tokenizer is None:
            return "I'm still learning. Please train me first!"
        
        # Combine context with current input
        if context:
            full_input = " ".join(context) + " " + user_input
        else:
            full_input = user_input
        
        # Preprocess input
        sequence = self.tokenizer.texts_to_sequences([full_input])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Generate prediction
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(prediction[0])
        
        # Convert back to text
        reverse_word_map = {v: k for k, v in self.tokenizer.word_index.items()}
        response_words = []
        
        for i in range(len(prediction[0])):
            if predicted_index in reverse_word_map:
                word = reverse_word_map[predicted_index]
                if word == '<OOV>':
                    continue
                response_words.append(word)
            # Get next word prediction (simplified)
            if len(response_words) >= 10:  # Limit response length
                break
        
        response = " ".join(response_words) if response_words else "I'm not sure how to respond to that."
        return response.capitalize()
    
    def save_model(self):
        """Save model and tokenizer"""
        if self.model:
            self.model.save('models/chatbot_model.h5')
        if self.tokenizer:
            with open('models/tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def load_model(self):
        """Load pre-trained model and tokenizer"""
        try:
            self.model = tf.keras.models.load_model('models/chatbot_model.h5')
            with open('models/tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
