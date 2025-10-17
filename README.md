🤖 Chatbot with Memory and Learning Capabilities

An intelligent chatbot that maintains conversation memory, asks for feedback on responses, and retrains itself based on user reviews.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9b9f9c5a-2b0a-49df-b697-56d97e721ded" />


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

How It Works

The chatbot maintains conversation history per user session for context-aware replies.

Each response is followed by a feedback prompt (1–5 rating scale).

Feedback data is stored in data/feedback_data.json.

The chatbot continuously retrains using both the initial training data and user feedback to improve over time.


| File                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `training_data.json` | Contains the initial training dataset for the chatbot.       |
| `feedback_data.json` | Stores user ratings and corrections for continuous learning. |
| `conversations.json` | Keeps track of ongoing and past conversations per user.      |




🧩 Technologies Used

Python 3.9+

TensorFlow / Keras — For the LSTM-based neural network

Flask / FastAPI — To serve the chatbot as a web API

JSON — For storing training data, feedback, and conversations



🚀 Future Enhancements

Integration with large language models (LLMs) for hybrid responses

Real-time visualization of chatbot learning metrics

Web-based dashboard for conversation and feedback management

Support for multi-turn, multi-user conversations with authentication


🧑‍💻 Contributing

Contributions are welcome!
To contribute:

Fork the repository

Create a new branch (feature/new-feature)

Commit your changes

Submit a pull request


