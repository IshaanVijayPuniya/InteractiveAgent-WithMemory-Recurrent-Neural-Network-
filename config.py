# Model configuration
MODEL_CONFIG = {
    'max_vocab_size': 10000,
    'max_sequence_length': 50,
    'embedding_dim': 100,
    'lstm_units': 128,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2
}

# Memory configuration
MEMORY_CONFIG = {
    'max_conversation_history': 100,
    'context_window': 10
}

# File paths
PATHS = {
    'model': 'models/chatbot_model.h5',
    'tokenizer': 'models/tokenizer.pkl',
    'training_data': 'data/training_data.json',
    'feedback_data': 'data/feedback_data.json',
    'conversations': 'data/conversations.json'
}
