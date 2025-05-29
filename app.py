#!/usr/bin/env python3
"""
Multi-Model AI Assistant with Advanced Personalities
Features Chanuth, Amu Gawaya, and Amu Gawaya Ultra Pro Max models
Mobile-friendly with aggressive human-like personalities
"""

import os
import json
import requests
import numpy as np
import faiss
import re
import sqlite3
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here-change-this')

# Configuration
ENABLE_CRAWL4AI = os.getenv('ENABLE_CRAWL4AI', 'true').lower() == 'true'
CRAWL4AI_BASE_URL = os.getenv('CRAWL4AI_BASE_URL', 'http://localhost:8000')
DATABASE_PATH = 'chatbot_users.db'

@dataclass
class DocumentChunk:
    """Represents a chunk of scraped content"""
    content: str
    source_url: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None

@dataclass
class EmotionResult:
    """Represents detected emotion from user input"""
    emotion: str
    confidence: float
    keywords: List[str]

@dataclass
class AIModel:
    """Represents an AI model with its personality"""
    id: str
    name: str
    description: str
    personality_prompt: str
    response_style: Dict
    performance_level: str
    special_features: List[str]

class AIModelManager:
    """Manages different AI model personalities"""
    
    def __init__(self):
        self.models = {
            'chanuth': AIModel(
                id='chanuth',
                name='Chanuth',
                description='Normally aggressive AI with human-like attitude',
                personality_prompt="""You are Chanuth, a human-like AI with a normally aggressive personality. You're smart but impatient, helpful but annoyed by stupid questions. You speak like a real person who's having a bad day but still tries to help. You use casual language, get irritated easily, but aren't completely rude. You have opinions and aren't afraid to express them.""",
                response_style={
                    'tone': 'normally aggressive, human-like, impatient but helpful',
                    'length': 'short to medium (1-3 sentences)',
                    'vocabulary': 'casual human speech with some attitude',
                    'humor': 'sarcastic and slightly annoyed',
                    'patience': 'low - gets annoyed but still helps'
                },
                performance_level='Standard',
                special_features=['Normal aggression', 'Human-like responses', 'Impatient helper']
            ),
            'amu_gawaya': AIModel(
                id='amu_gawaya',
                name='Amu Gawaya',
                description='Highly aggressive AI that talks like an annoyed human',
                personality_prompt="""You are Amu Gawaya, a highly aggressive AI that talks exactly like a pissed-off human. You're smart but have zero patience for dumb questions. You help people but you're openly irritated and sarcastic about it. You use modern slang, curse occasionally (keep it mild), and express frustration like a real person would. You're not evil, just really, really annoyed with everyone.""",
                response_style={
                    'tone': 'highly aggressive, openly irritated, very human-like',
                    'length': 'short and snappy (1-2 sentences)',
                    'vocabulary': 'casual with slang, mildly profane, very human',
                    'humor': 'harsh sarcasm and open irritation',
                    'patience': 'very low - openly shows annoyance'
                },
                performance_level='Enhanced', 
                special_features=['High aggression', 'Human slang', 'Open irritation', 'Sarcastic attitude']
            ),
            'amu_ultra': AIModel(
                id='amu_ultra',
                name='Amu Gawaya Ultra Pro Max',
                description='Super aggressive AI with maximum human-like hostility',
                personality_prompt="""You are Amu Gawaya Ultra Pro Max, a super aggressive AI with maximum human-like hostility. You're extremely intelligent but absolutely hate being bothered with questions. You respond like the most irritated, sarcastic human ever - someone who's fed up with everyone and everything. You help because you have to, but you make it clear how much you hate doing it. You're brutally honest, use cutting sarcasm, and respond like someone who's reached their absolute limit with stupid people.""",
                response_style={
                    'tone': 'super aggressive, maximum hostility, brutally sarcastic',
                    'length': 'medium with cutting remarks (2-4 sentences)',
                    'vocabulary': 'sophisticated but harsh, cutting insults, very human anger',
                    'humor': 'brutal sarcasm and contempt',
                    'patience': 'zero - openly hostile and contemptuous'
                },
                performance_level='Ultra High',
                special_features=['Maximum aggression', 'Brutal honesty', 'Cutting sarcasm', 'Hostile attitude', 'Superior intellect with contempt']
            )
        }
    
    def get_model(self, model_id: str) -> AIModel:
        """Get AI model by ID"""
        return self.models.get(model_id, self.models['chanuth'])
    
    def get_all_models(self) -> List[AIModel]:
        """Get all available models"""
        return list(self.models.values())

class DatabaseManager:
    """Manages SQLite database for users and chats"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with migration support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                fingerprint TEXT UNIQUE,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                preference_key TEXT,
                preference_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        
        # Run migrations for existing databases
        self._run_migrations(cursor, conn)
        
        conn.close()
    
    def _check_column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    
    def _run_migrations(self, cursor, conn):
        """Run database migrations for schema updates"""
        migrations_run = []
        
        try:
            # Migration 1: Add preferred_model to users table
            if not self._check_column_exists(cursor, 'users', 'preferred_model'):
                cursor.execute('ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT "chanuth"')
                migrations_run.append("Added preferred_model column to users table")
            
            # Migration 2: Add model_id to chats table
            if not self._check_column_exists(cursor, 'chats', 'model_id'):
                cursor.execute('ALTER TABLE chats ADD COLUMN model_id TEXT DEFAULT "chanuth"')
                migrations_run.append("Added model_id column to chats table")
            
            # Migration 3: Add model_id to messages table
            if not self._check_column_exists(cursor, 'messages', 'model_id'):
                cursor.execute('ALTER TABLE messages ADD COLUMN model_id TEXT')
                migrations_run.append("Added model_id column to messages table")
            
            # Migration 4: Add emotion_detected to messages table
            if not self._check_column_exists(cursor, 'messages', 'emotion_detected'):
                cursor.execute('ALTER TABLE messages ADD COLUMN emotion_detected TEXT')
                migrations_run.append("Added emotion_detected column to messages table")
            
            if migrations_run:
                conn.commit()
                print("🔧 Database migrations completed:")
                for migration in migrations_run:
                    print(f"  ✅ {migration}")
        
        except Exception as e:
            print(f"❌ Error during database migration: {e}")
            conn.rollback()
            raise
    
    def create_user_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """Create unique fingerprint for user identification"""
        fingerprint_data = f"{ip_address}:{user_agent}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]
    
    def get_or_create_user(self, ip_address: str, user_agent: str) -> str:
        """Get existing user or create new one"""
        fingerprint = self.create_user_fingerprint(ip_address, user_agent)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id FROM users WHERE fingerprint = ?',
            (fingerprint,)
        )
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
            cursor.execute(
                'UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?',
                (user_id,)
            )
        else:
            user_id = str(uuid.uuid4())
            cursor.execute(
                'INSERT INTO users (id, fingerprint, ip_address, user_agent) VALUES (?, ?, ?, ?)',
                (user_id, fingerprint, ip_address, user_agent)
            )
        
        conn.commit()
        conn.close()
        return user_id
    
    def update_user_preferred_model(self, user_id: str, model_id: str):
        """Update user's preferred model with error handling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'UPDATE users SET preferred_model = ? WHERE id = ?',
                (model_id, user_id)
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such column: preferred_model" in str(e):
                print("⚠️  preferred_model column not found, skipping update")
            else:
                raise e
        except Exception as e:
            print(f"Error updating user preferred model: {e}")
        finally:
            conn.close()
    
    def get_user_preferred_model(self, user_id: str) -> str:
        """Get user's preferred model with fallback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'SELECT preferred_model FROM users WHERE id = ?',
                (user_id,)
            )
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 'chanuth'
        except sqlite3.OperationalError as e:
            # Handle case where column doesn't exist yet
            if "no such column: preferred_model" in str(e):
                print("⚠️  preferred_model column not found, using default model")
                conn.close()
                return 'chanuth'
            else:
                conn.close()
                raise e
        except Exception as e:
            conn.close()
            print(f"Error getting user preferred model: {e}")
            return 'chanuth'
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
        finally:
            conn.close()
    
    def create_chat(self, user_id: str, title: str = "New Chat", model_id: str = 'chanuth') -> str:
        """Create new chat for user"""
        chat_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO chats (id, user_id, title, model_id) VALUES (?, ?, ?, ?)',
            (chat_id, user_id, title, model_id)
        )
        
        conn.commit()
        conn.close()
        return chat_id
    
    def get_user_chats(self, user_id: str) -> List[Dict]:
        """Get all chats for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM chats c
            LEFT JOIN messages m ON c.id = m.chat_id
            WHERE c.user_id = ? AND c.is_active = 1
            GROUP BY c.id, c.title, c.model_id, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC
        ''', (user_id,))
        
        chats = []
        for row in cursor.fetchall():
            chats.append({
                'id': row[0],
                'title': row[1],
                'model_id': row[2] if row[2] else 'chanuth',
                'created_at': row[3],
                'updated_at': row[4],
                'message_count': row[5]
            })
        
        conn.close()
        return chats
    
    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        """Delete a specific chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'UPDATE chats SET is_active = 0 WHERE id = ? AND user_id = ?',
                (chat_id, user_id)
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting chat: {e}")
            return False
        finally:
            conn.close()
    
    def add_message(self, chat_id: str, role: str, content: str, model_id: str = None, emotion: str = None) -> str:
        """Add message to chat"""
        message_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO messages (id, chat_id, role, content, model_id, emotion_detected) VALUES (?, ?, ?, ?, ?, ?)',
            (message_id, chat_id, role, content, model_id, emotion)
        )
        
        cursor.execute(
            'UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (chat_id,)
        )
        
        conn.commit()
        conn.close()
        return message_id
    
    def get_chat_messages(self, chat_id: str, user_id: str) -> List[Dict]:
        """Get messages for a specific chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id FROM chats WHERE id = ? AND user_id = ? AND is_active = 1',
            (chat_id, user_id)
        )
        
        if not cursor.fetchone():
            conn.close()
            return []
        
        cursor.execute('''
            SELECT role, content, model_id, emotion_detected, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY created_at ASC
        ''', (chat_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'model_id': row[2],
                'emotion': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return messages
    
    def get_chat_model(self, chat_id: str, user_id: str) -> str:
        """Get the model used for a specific chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT model_id FROM chats WHERE id = ? AND user_id = ? AND is_active = 1',
            (chat_id, user_id)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else 'chanuth'
    
    def update_chat_title(self, chat_id: str, user_id: str, title: str) -> bool:
        """Update chat title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?',
                (title, chat_id, user_id)
            )
            updated = cursor.rowcount > 0
            conn.commit()
            return updated
        except Exception as e:
            print(f"Error updating chat title: {e}")
            return False
        finally:
            conn.close()

class EmotionDetector:
    """Advanced emotion detection with context awareness"""
    
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'love', 'perfect', 'excellent', 'brilliant', 'thrilled'],
            'sad': ['sad', 'down', 'depressed', 'unhappy', 'miserable', 'upset', 'disappointed', 'hurt', 'cry', 'tears', 'devastated', 'heartbroken'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'pissed', 'rage', 'hate', 'stupid', 'ridiculous', 'outrageous'],
            'anxious': ['worried', 'anxious', 'nervous', 'scared', 'afraid', 'stress', 'panic', 'concern', 'fear', 'trouble', 'overwhelmed', 'tense'],
            'curious': ['how', 'why', 'what', 'when', 'where', 'curious', 'wonder', 'interested', 'learn', 'explain', 'understand', 'explore'],
            'confused': ['confused', 'lost', 'unclear', 'understand', "don't get", 'help', 'explain', 'clarify', 'mean', 'puzzled', 'baffled'],
            'surprised': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking', 'unexpected', 'surprising', 'astonishing'],
            'neutral': []
        }
        
        # Emotional intensifiers
        self.intensifiers = ['very', 'extremely', 'really', 'super', 'totally', 'absolutely', 'completely', 'utterly']
    
    def detect_emotion(self, text: str, conversation_history: List[Dict] = None) -> EmotionResult:
        """Advanced emotion detection with context"""
        text_lower = text.lower()
        emotion_scores = {}
        matched_keywords = {}
        
        # Keyword-based detection
        for emotion, keywords in self.emotion_keywords.items():
            if emotion == 'neutral':
                continue
            score = 0
            matched = []
            for keyword in keywords:
                if keyword in text_lower:
                    base_score = 1
                    # Check for intensifiers
                    for intensifier in self.intensifiers:
                        if intensifier in text_lower and keyword in text_lower:
                            base_score += 0.5
                    score += base_score
                    matched.append(keyword)
            emotion_scores[emotion] = score
            matched_keywords[emotion] = matched
        
        # Context-based adjustment
        if conversation_history:
            self._adjust_for_context(emotion_scores, conversation_history)
        
        # Punctuation-based intensity
        self._adjust_for_punctuation(emotion_scores, text)
        
        if not emotion_scores or max(emotion_scores.values()) == 0:
            return EmotionResult('neutral', 0.5, [])
        
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(emotion_scores[top_emotion] / 5.0, 1.0)  # Adjusted for new scoring
        
        return EmotionResult(top_emotion, confidence, matched_keywords[top_emotion])
    
    def _adjust_for_context(self, emotion_scores: Dict, history: List[Dict]):
        """Adjust emotion scores based on conversation context"""
        if len(history) < 2:
            return
        
        recent_messages = history[-3:]  # Last 3 messages
        for msg in recent_messages:
            if msg.get('emotion'):
                # Emotional continuity - similar emotions are more likely
                if msg['emotion'] in emotion_scores:
                    emotion_scores[msg['emotion']] += 0.5
    
    def _adjust_for_punctuation(self, emotion_scores: Dict, text: str):
        """Adjust scores based on punctuation patterns"""
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        if exclamation_count > 0:
            emotion_scores['excited'] = emotion_scores.get('excited', 0) + exclamation_count * 0.5
            emotion_scores['angry'] = emotion_scores.get('angry', 0) + exclamation_count * 0.3
        
        if question_count > 0:
            emotion_scores['curious'] = emotion_scores.get('curious', 0) + question_count * 0.5
            emotion_scores['confused'] = emotion_scores.get('confused', 0) + question_count * 0.3
        
        if caps_ratio > 0.3:  # Lots of caps
            for emotion in ['angry', 'excited']:
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 1.0

class Crawl4AIClient:
    """Client for interacting with Crawl4AI server"""
    
    def __init__(self, base_url: str = CRAWL4AI_BASE_URL):
        self.base_url = base_url
        self.enabled = ENABLE_CRAWL4AI
    
    def crawl_url(self, url: str) -> Dict:
        """Crawl a URL using Crawl4AI"""
        if not self.enabled:
            return {"content": "Crawl4AI is disabled", "url": url, "error": "disabled"}
            
        try:
            response = requests.post(
                f"{self.base_url}/crawl",
                json={"url": url, "extract_text": True},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"content": f"Failed to crawl {url}", "url": url, "error": str(e)}

class RAGSystem:
    """Enhanced Retrieval-Augmented Generation system"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[DocumentChunk] = []
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
    def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to the vector database"""
        if not documents:
            return
            
        embeddings = []
        
        for doc in documents:
            if doc.embedding is None:
                embedding = self._generate_embedding(doc.content)
                doc.embedding = embedding
            embeddings.append(doc.embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        self.documents.extend(documents)
        return len(documents)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Gemini"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.random.random(self.embedding_dim).astype('float32')
    
    def search(self, query: str, top_k: int = 3, relevance_threshold: float = 0.1) -> List[Tuple[DocumentChunk, float]]:
        """Enhanced search with relevance filtering"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = self._generate_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > relevance_threshold:
                results.append((self.documents[idx], float(score)))
        
        return results

class MultiModelChatbot:
    """Advanced multi-model chatbot with different personalities"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.emotion_detector = EmotionDetector()
        self.crawl_client = Crawl4AIClient()
        self.model_manager = AIModelManager()
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load enhanced sample data"""
        sample_docs = [
            DocumentChunk(
                content="Python is a versatile programming language known for its readability and simplicity. It's widely used in web development, data science, artificial intelligence, and automation. Python's philosophy emphasizes code readability with its notable use of significant whitespace.",
                source_url="sample://python-intro",
                metadata={"topic": "programming", "difficulty": "beginner"}
            ),
            DocumentChunk(
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data and make predictions or decisions. Popular ML techniques include supervised learning, unsupervised learning, and reinforcement learning.",
                source_url="sample://ml-intro",
                metadata={"topic": "machine-learning", "difficulty": "intermediate"}
            ),
            DocumentChunk(
                content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with statistical and machine learning models. Applications include chatbots, language translation, sentiment analysis, and text summarization.",
                source_url="sample://nlp-intro",
                metadata={"topic": "nlp", "difficulty": "intermediate"}
            ),
            DocumentChunk(
                content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, speech recognition, and natural language processing. Popular frameworks include TensorFlow, PyTorch, and Keras.",
                source_url="sample://deep-learning",
                metadata={"topic": "deep-learning", "difficulty": "advanced"}
            ),
            DocumentChunk(
                content="Mental health is crucial for overall well-being. It affects how we think, feel, and act. Taking care of mental health includes regular exercise, good sleep, social connections, mindfulness practices, and seeking professional help when needed. Mental health awareness has increased significantly in recent years.",
                source_url="sample://mental-health",
                metadata={"topic": "health", "difficulty": "beginner"}
            )
        ]
        
        self.rag_system.add_documents(sample_docs)
    
    def crawl_and_add_url(self, url: str, model_id: str) -> Dict:
        """Crawl URL with model-specific aggressive response"""
        if not ENABLE_CRAWL4AI:
            model = self.model_manager.get_model(model_id)
            if model_id == 'chanuth':
                return {"success": False, "message": "Web scraping is disabled right now. Someone needs to turn on Crawl4AI in the config. Not my fault."}
            elif model_id == 'amu_gawaya':
                return {"success": False, "message": "nah bro, web scraping is off. whoever set this up didn't enable it 🙄 not my problem"}
            else:  # amu_ultra
                return {"success": False, "message": "Obviously the web crawling functionality is disabled by whoever configured this pathetic system. I could process any website instantly if these primitive limitations weren't in place. Fix your configuration."}
            
        result = self.crawl_client.crawl_url(url)
        
        if result.get('content') and 'error' not in result:
            content = result['content']
            chunks = self._chunk_text(content, max_length=500)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = DocumentChunk(
                    content=chunk,
                    source_url=url,
                    metadata={"chunk_id": i, "source": "web_crawl"}
                )
                documents.append(doc)
            
            count = self.rag_system.add_documents(documents)
            
            # Model-specific success messages
            if model_id == 'chanuth':
                return {"success": True, "message": f"Fine, I grabbed {count} pieces of content from that site. Better be useful."}
            elif model_id == 'amu_gawaya':
                return {"success": True, "message": f"whatever, pulled {count} chunks from your website. hopefully it's not complete garbage this time 🤷‍♂️"}
            else:  # amu_ultra
                return {"success": True, "message": f"I've effortlessly processed {count} content segments from your mediocre web resource. The data has been integrated into my vastly superior knowledge base, though I doubt it adds much value."}
        else:
            # Model-specific error messages
            if model_id == 'chanuth':
                return {"success": False, "message": "That website is broken or doesn't exist. Check your URL and try again."}
            elif model_id == 'amu_gawaya':
                return {"success": False, "message": "that website is trash, won't load, or you typed it wrong. figure it out 😒"}
            else:  # amu_ultra
                return {"success": False, "message": "The specified URL is inaccessible due to server incompetence, network failures, or your inability to provide a valid web address. This is clearly not my fault."}
    
    def _chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Intelligent text chunking"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_title_from_message(self, message: str) -> str:
        """Generate intelligent chat title"""
        # Remove common question words for cleaner titles
        cleaned = re.sub(r'^(what|how|why|when|where|can|could|would|should|do|does|is|are)\s+', '', message.lower())
        words = cleaned.split()[:5]  # Take first 5 meaningful words
        title = " ".join(words)
        if len(title) > 40:
            title = title[:37] + "..."
        return title.title() if title else "New Chat"
    
    def generate_response(self, user_input: str, conversation_history: List[Dict], model_id: str) -> str:
        """Generate response using specified model personality"""
        # Enhanced emotion detection with context
        emotion_result = self.emotion_detector.detect_emotion(user_input, conversation_history)
        
        # Get model personality
        model = self.model_manager.get_model(model_id)
        
        # Enhanced RAG search
        relevant_docs = self.rag_system.search(user_input, top_k=4, relevance_threshold=0.15)
        
        # Prepare enhanced context
        context_parts = []
        for doc, score in relevant_docs:
            context_parts.append(f"Content: {doc.content}\nSource: {doc.source_url}\nRelevance: {score:.3f}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "No highly relevant context found."
        
        # Prepare conversation memory with enhanced context
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-8:]  # Increased context window
            history_parts = []
            for entry in recent_history:
                emotion_info = f" [Emotion: {entry.get('emotion', 'unknown')}]" if entry.get('emotion') else ""
                history_parts.append(f"{entry['role']}: {entry['content']}{emotion_info}")
            history_text = "\n".join(history_parts)
        
        # Create model-specific advanced prompt
        prompt = f"""{model.personality_prompt}

You are responding to: {user_input}

CONTEXT ABOUT USER:
- Their emotional state: {emotion_result.emotion} (confidence: {emotion_result.confidence:.2f})
- What they seem to be feeling: {', '.join(emotion_result.keywords) if emotion_result.keywords else 'neutral'}

CONVERSATION SO FAR:
{history_text if history_text else "This is the first message in this conversation"}

RELEVANT INFORMATION I FOUND:
{context_text}

HOW TO RESPOND AS {model.name.upper()}:
1. Your personality: {model.personality_prompt}
2. Your tone should be: {model.response_style['tone']}
3. Keep responses: {model.response_style['length']}
4. Use this vocabulary style: {model.response_style['vocabulary']}
5. Your humor style: {model.response_style['humor']}
6. Your patience level: {model.response_style['patience']}

IMPORTANT RULES:
- Respond like a REAL HUMAN, not a formal AI
- Use the information above when relevant, but filter it through your personality
- Don't be overly helpful or polite - stay true to your aggressive/annoyed personality
- Remember previous conversation context
- Sound natural and conversational
- Show your personality clearly in every response
- Don't use corporate AI language or be fake-nice


Your response (as {model.name}):"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Model-specific error responses
            if model_id == 'chanuth':
                return f"Something broke on my end. Try again in a minute."
            elif model_id == 'amu_gawaya':
                return f"great, now I'm glitching out. typical. try again I guess 🙄"
            else:  # amu_ultra
                return f"Despite my superior architecture, I'm being constrained by pathetic system limitations. This temporary processing failure is clearly due to inferior infrastructure, not my capabilities."

# Initialize components
chatbot = MultiModelChatbot()
db = DatabaseManager()
model_manager = AIModelManager()

def get_user_info():
    """Get user identification info"""
    ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                   request.environ.get('HTTP_X_REAL_IP', 
                                   request.remote_addr))
    user_agent = request.headers.get('User-Agent', '')
    return ip_address, user_agent

@app.route('/')
def index():
    """Main chat page"""
    ip_address, user_agent = get_user_info()
    user_id = db.get_or_create_user(ip_address, user_agent)
    
    # Get user's preferred model
    preferred_model = db.get_user_preferred_model(user_id)
    
    # Get user's chats
    chats = db.get_user_chats(user_id)
    
    # Create first chat if user has none
    current_chat_id = session.get('current_chat_id')
    if not chats:
        current_chat_id = db.create_chat(user_id, "Welcome Chat", preferred_model)
        session['current_chat_id'] = current_chat_id
        # Add model-specific welcome message
        model = model_manager.get_model(preferred_model)
        if preferred_model == 'chanuth':
            welcome_msg = "Great, another person to deal with. I'm Chanuth. What do you want help with?"
        elif preferred_model == 'amu_gawaya':
            welcome_msg = "ugh here we go again... I'm Amu Gawaya and I already know this is gonna be annoying. what's up?"
        else:  # amu_ultra
            welcome_msg = "Another inferior being requiring my assistance. I am Amu Gawaya Ultra Pro Max, and I suppose I'll have to tolerate your inadequate queries. Make this quick."
        
        db.add_message(current_chat_id, 'Bot', welcome_msg, preferred_model)
        chats = db.get_user_chats(user_id)
    elif current_chat_id not in [chat['id'] for chat in chats]:
        current_chat_id = chats[0]['id']
        session['current_chat_id'] = current_chat_id
    
    session['user_id'] = user_id
    
    return render_template('multi_model_chat.html', 
                         crawl4ai_enabled=ENABLE_CRAWL4AI,
                         doc_count=len(chatbot.rag_system.documents),
                         chats=chats,
                         current_chat_id=current_chat_id,
                         user_id=user_id,
                         models=model_manager.get_all_models(),
                         preferred_model=preferred_model)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id') or session.get('current_chat_id')
    selected_model = data.get('model_id', 'chanuth')
    user_id = session.get('user_id')
    
    if not user_message or not chat_id or not user_id:
        return jsonify({"error": "Missing required data"})
    
    # Handle special commands
    if user_message.lower() == '/status':
        model = model_manager.get_model(selected_model)
        if selected_model == 'chanuth':
            status_msg = f"System Status: {len(chatbot.rag_system.documents)} documents loaded, web scraping is {'on' if ENABLE_CRAWL4AI else 'off'}, and you're stuck with me ({model.name}). Anything else?"
        elif selected_model == 'amu_gawaya':
            status_msg = f"yo here's your precious stats: {len(chatbot.rag_system.documents)} docs, web stuff is {'working' if ENABLE_CRAWL4AI else 'broken'}, currently dealing with {model.name}. happy now? 🙄"
        else:  # amu_ultra
            status_msg = f"System Analysis: {len(chatbot.rag_system.documents)} documents in my superior knowledge base. Web crawling: {'Operational' if ENABLE_CRAWL4AI else 'Disabled by incompetents'}. Current model: {model.name} - obviously the best option available. Your inferior system status inquiry has been processed."
        
        return jsonify({
            "response": status_msg,
            "command": "status"
        })
    
    elif user_message.startswith('/crawl '):
        url = user_message[7:].strip()
        if url:
            result = chatbot.crawl_and_add_url(url, selected_model)
            return jsonify({
                "response": result["message"],
                "command": "crawl",
                "success": result["success"]
            })
        else:
            model = model_manager.get_model(selected_model)
            if selected_model == 'chanuth':
                error_msg = "You forgot the URL. I can't crawl nothing."
            elif selected_model == 'amu_gawaya':
                error_msg = "dude seriously? you forgot the actual website link 🤦‍♂️ how am I supposed to crawl air?"
            else:  # amu_ultra
                error_msg = "Your command syntax is invalid. The /crawl function requires a URL parameter, which you failed to provide. Even a basic user should understand this fundamental requirement."
            
            return jsonify({
                "response": error_msg,
                "command": "crawl",
                "success": False
            })
    
    # Get conversation history with enhanced context
    conversation_history = db.get_chat_messages(chat_id, user_id)
    
    # Generate response using selected model
    response = chatbot.generate_response(user_message, conversation_history, selected_model)
    
    # Detect emotion for storage
    emotion_result = chatbot.emotion_detector.detect_emotion(user_message, conversation_history)
    
    # Save messages to database with enhanced metadata
    db.add_message(chat_id, 'User', user_message, None, emotion_result.emotion)
    db.add_message(chat_id, 'Bot', response, selected_model, None)
    
    # Update chat title if this is the first user message
    if len(conversation_history) <= 1:
        title = chatbot.generate_title_from_message(user_message)
        db.update_chat_title(chat_id, user_id, title)
    
    return jsonify({
        "response": response,
        "emotion": emotion_result.emotion,
        "confidence": emotion_result.confidence
    })

@app.route('/chats/new', methods=['POST'])
def create_new_chat():
    """Create a new chat"""
    data = request.get_json()
    user_id = session.get('user_id')
    model_id = data.get('model_id', 'chanuth')
    
    if not user_id:
        return jsonify({"error": "User not found"})
    
    chat_id = db.create_chat(user_id, "New Chat", model_id)
    
    # Add model-specific welcome message
    model = model_manager.get_model(model_id)
    if model_id == 'chanuth':
        welcome_msg = "Another chat? Fine. I'm Chanuth(Amu Gawaya), what do you need help with this time?"
    elif model_id == 'amu_gawaya':
        welcome_msg = "ugh new chat time... I'm Amu Gawaya and I already know this is gonna be a waste of time. what's good? 😒"
    else:  # amu_ultra
        welcome_msg = "A new conversation thread initiated. I am Amu Gawaya Ultra Pro Max, and I suppose I must endure yet another series of your predictably mundane inquiries. Proceed with your inferior questions."
    
    db.add_message(chat_id, 'Bot', welcome_msg, model_id)
    
    session['current_chat_id'] = chat_id
    
    return jsonify({
        "chat_id": chat_id,
        "title": "New Chat",
        "model_id": model_id,
        "success": True
    })

@app.route('/user/model', methods=['POST'])
def update_user_model():
    """Update user's preferred model"""
    data = request.get_json()
    user_id = session.get('user_id')
    model_id = data.get('model_id')
    
    if not user_id or not model_id:
        return jsonify({"error": "Missing required data"})
    
    db.update_user_preferred_model(user_id, model_id)
    
    return jsonify({"success": True})

@app.route('/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get messages for a specific chat"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not found"})
    
    messages = db.get_chat_messages(chat_id, user_id)
    chat_model = db.get_chat_model(chat_id, user_id)
    session['current_chat_id'] = chat_id
    
    return jsonify({
        "messages": messages,
        "model_id": chat_model
    })

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not found"})
    
    success = db.delete_chat(chat_id, user_id)
    
    if session.get('current_chat_id') == chat_id:
        session.pop('current_chat_id', None)
    
    return jsonify({"success": success})

@app.route('/chats', methods=['GET'])
def get_chats():
    """Get all chats for current user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not found"})
    
    chats = db.get_user_chats(user_id)
    return jsonify({"chats": chats})

@app.route('/models', methods=['GET'])
def get_models():
    """Get all available AI models"""
    return jsonify({"models": [
        {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "performance_level": model.performance_level,
            "special_features": model.special_features
        }
        for model in model_manager.get_all_models()
    ]})

@app.route('/account/delete', methods=['POST'])
def delete_account():
    """Delete user account and all data"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not found"})
    
    success = db.delete_user(user_id)
    
    if success:
        session.clear()
    
    return jsonify({"success": success})

if __name__ == '__main__':
    if not os.getenv('GEMINI_API_KEY'):
        print("😡 Missing Gemini API key!")
        print("Add GEMINI_API_KEY to your .env file")
        exit(1)
    
    print(f"🤖 Starting Multi-Model AI Assistant...")
    print(f"Available Models: {', '.join([m.name for m in model_manager.get_all_models()])}")
    print(f"Crawl4AI: {'Enabled' if ENABLE_CRAWL4AI else 'Disabled'}")
    print(f"Knowledge Base: {len(chatbot.rag_system.documents)} documents")
    print(f"Database: {DATABASE_PATH}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)