🤖 Chatbot with Memory and Learning Capabilities

An intelligent chatbot that maintains conversation memory, asks for feedback on responses, and retrains itself based on user reviews.

🌟 Features

Conversation Memory — Remembers user interactions and maintains context

Feedback System — Asks users to rate responses (1–5 scale)

Continuous Learning — Retrains the model based on user feedback and corrections

User-specific Context — Maintains separate conversation history for different users

Neural Network Model — Uses an LSTM-based neural network for response generation
```
chatbot-with-memory/
├── app.py                   # Main application
├── train_model.py           # Initial model training script
├── requirements.txt         # Python dependencies
├── config.py                # Configuration settings
│
├── models/                  # Model implementations
│   ├── chatbot_model.py     # Neural network model
│   └── memory_handler.py    # Conversation memory management
│
├── utils/                   # Utility functions
│   └── helpers.py           # Text processing helpers
│
├── data/                    # Data files
│   ├── training_data.json   # Initial training data
│   ├── feedback_data.json   # User feedback storage
│   └── conversations.json   # Conversation history
│
└── README.md
```

