import json
import os
from datetime import datetime
from collections import defaultdict

class MemoryHandler:
    def __init__(self):
        self.user_conversations = defaultdict(list)
        self.load_conversations()
    
    def store_interaction(self, user_id, user_input, bot_response):
        """Store user-bot interaction in memory"""
        interaction = {
            'user_input': user_input,
            'bot_response': bot_response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.user_conversations[user_id].append(interaction)
        
        # Keep only recent history (last 100 messages)
        if len(self.user_conversations[user_id]) > 100:
            self.user_conversations[user_id] = self.user_conversations[user_id][-100:]
        
        self.save_conversations()
    
    def get_user_context(self, user_id, window_size=5):
        """Get recent conversation context for a user"""
        user_history = self.user_conversations.get(user_id, [])
        recent_interactions = user_history[-window_size:]
        
        context = []
        for interaction in recent_interactions:
            context.append(interaction['user_input'])
            context.append(interaction['bot_response'])
        
        return context
    
    def get_conversation_history(self, user_id, limit=None):
        """Get full conversation history for a user"""
        history = self.user_conversations.get(user_id, [])
        if limit:
            return history[-limit:]
        return history
    
    def save_conversations(self):
        """Save all conversations to file"""
        try:
            with open('data/conversations.json', 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                json.dump(dict(self.user_conversations), f, indent=2)
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def load_conversations(self):
        """Load conversations from file"""
        try:
            if os.path.exists('data/conversations.json'):
                with open('data/conversations.json', 'r') as f:
                    conversations = json.load(f)
                    self.user_conversations.update(conversations)
        except Exception as e:
            print(f"Error loading conversations: {e}")
