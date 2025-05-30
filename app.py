#!/usr/bin/env python3
"""
Multi-Model AI Assistant with Integrated Crawl4AI
Features F-1, F-1.5, and F-o1 models
Automatic web crawling based on user queries - no commands needed!
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
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, jsonify, session, send_from_directory
import random
import asyncio
from urllib.parse import urlparse, urljoin
import time
import concurrent.futures

# Import crawl4ai directly - Updated for v0.6+
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.models import CrawlResult
    import asyncio
    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI imported successfully")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    print(f"⚠️  Crawl4AI not available: {e}")
    print("Install with: pip install crawl4ai")

# Load environment variables
load_dotenv()

# Utility functions
def get_current_time_sri_lanka():
    """Get current time in Sri Lanka (UTC+5:30)"""
    sri_lanka_tz = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(sri_lanka_tz)
    return current_time.strftime("%I:%M %p on %B %d, %Y")

def get_current_context():
    """Get current date/time context for the AI"""
    sri_lanka_time = get_current_time_sri_lanka()
    utc_time = datetime.now(timezone.utc)
    return {
        "current_date": "May 31, 2025",
        "sri_lanka_time": sri_lanka_time,
        "utc_time": utc_time.strftime("%I:%M %p UTC on %B %d, %Y")
    }

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here-change-this')

# Configuration
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

class SmartWebCrawler:
    """Intelligent web crawler that automatically finds and crawls relevant content"""
    
    def __init__(self):
        self.available = CRAWL4AI_AVAILABLE
        self.crawl_cache = {}  # Simple cache to avoid repeated crawls
        self.rate_limit = {}   # Rate limiting per domain
        
        if self.available:
            print("🌐 Web crawler initialized (AsyncWebCrawler)")
        else:
            print("⚠️  Web crawler not available")
    
    def should_crawl_for_query(self, query: str) -> bool:
        """Determine if a query needs web crawling - more selective approach"""
        if not self.available:
            return False
        
        query_lower = query.lower()
        
        # Explicit requests for current/live information
        current_info_indicators = [
            'current time', 'what time is it', 'time now', 'current weather',
            'latest news', 'recent news', 'today\'s news', 'breaking news',
            'stock price', 'current price', 'price today', 'market today',
            'live score', 'current score', 'latest score',
            'current status', 'status now', 'current situation'
        ]
        
        # Company/organization specific recent information
        company_indicators = [
            'latest from', 'recent update', 'company news', 'new product',
            'earnings report', 'quarterly results', 'recent announcement',
            'updated policy', 'new feature', 'launch date'
        ]
        
        # Time-sensitive queries
        time_sensitive = [
            'today', 'this week', 'this month', 'this year', '2025', '2024',
            'recently', 'just announced', 'just released', 'upcoming'
        ]
        
        # Specific data that might need current lookup
        data_specific = [
            'statistics', 'data', 'report', 'study results', 'survey',
            'comparison', 'reviews', 'ratings', 'analysis'
        ]
        
        # Check if query explicitly asks for current information
        for indicator in current_info_indicators:
            if indicator in query_lower:
                return True
        
        # Check for company + time-sensitive combination
        companies = ['apple', 'google', 'microsoft', 'amazon', 'tesla', 'meta', 'openai']
        has_company = any(company in query_lower for company in companies)
        has_time_sensitive = any(indicator in query_lower for indicator in time_sensitive)
        has_company_specific = any(indicator in query_lower for indicator in company_indicators)
        
        if has_company and (has_time_sensitive or has_company_specific):
            return True
        
        # Check for data-specific queries that might need fresh information
        if any(indicator in query_lower for indicator in data_specific) and has_time_sensitive:
            return True
        
        # Don't crawl for general knowledge, personal questions, or basic calculations
        general_knowledge = [
            'what is', 'who is', 'how to', 'why does', 'explain', 'definition',
            'meaning of', 'history of', 'concept of', 'difference between'
        ]
        
        personal_questions = [
            'how are you', 'who are you', 'your name', 'about yourself',
            'can you help', 'what can you do', 'capabilities'
        ]
        
        # Skip crawling for general knowledge unless it's time-sensitive
        for indicator in general_knowledge:
            if indicator in query_lower and not has_time_sensitive:
                return False
        
        # Skip crawling for personal questions
        for indicator in personal_questions:
            if indicator in query_lower:
                return False
        
        return False
    
    def generate_search_urls(self, query: str) -> List[str]:
        """Generate potential URLs to crawl based on query"""
        urls = []
        query_lower = query.lower()
        
        # Company-specific URLs
        company_urls = {
            'apple': ['https://www.apple.com/', 'https://investor.apple.com/'],
            'google': ['https://www.google.com/', 'https://abc.xyz/', 'https://about.google/'],
            'microsoft': ['https://www.microsoft.com/', 'https://news.microsoft.com/'],
            'amazon': ['https://www.amazon.com/', 'https://www.aboutamazon.com/'],
            'tesla': ['https://www.tesla.com/', 'https://ir.tesla.com/'],
            'netflix': ['https://www.netflix.com/', 'https://about.netflix.com/'],
            'meta': ['https://about.meta.com/', 'https://investor.fb.com/'],
            'facebook': ['https://about.meta.com/', 'https://www.facebook.com/'],
            'twitter': ['https://about.twitter.com/', 'https://twitter.com/'],
            'spacex': ['https://www.spacex.com/'],
            'openai': ['https://openai.com/', 'https://openai.com/blog/'],
            'anthropic': ['https://www.anthropic.com/', 'https://www.anthropic.com/news']
        }
        
        for company, company_urls_list in company_urls.items():
            if company in query_lower:
                urls.extend(company_urls_list)
        
        # News and information sites for general queries
        if 'news' in query_lower or 'latest' in query_lower:
            urls.extend([
                'https://www.reuters.com/',
                'https://www.bbc.com/news',
                'https://techcrunch.com/'
            ])
        
        # Financial information
        if any(term in query_lower for term in ['stock', 'price', 'market', 'financial']):
            urls.extend([
                'https://finance.yahoo.com/',
                'https://www.marketwatch.com/'
            ])
        
        # Tech information
        if any(term in query_lower for term in ['technology', 'tech', 'ai', 'software']):
            urls.extend([
                'https://techcrunch.com/',
                'https://www.theverge.com/',
                'https://arstechnica.com/'
            ])
        
        # Remove duplicates and limit to top 3 URLs
        return list(dict.fromkeys(urls))[:3]
    
    def is_rate_limited(self, url: str) -> bool:
        """Check if domain is rate limited"""
        domain = urlparse(url).netloc
        if domain in self.rate_limit:
            last_crawl = self.rate_limit[domain]
            if time.time() - last_crawl < 10:  # 10 second rate limit per domain
                return True
        return False
    
    def crawl_url(self, url: str) -> Dict:
        """Crawl a single URL and return structured data"""
        if not self.available:
            return {"content": "", "url": url, "error": "Crawler not available"}
        
        # Check cache first
        if url in self.crawl_cache:
            cache_time, cached_data = self.crawl_cache[url]
            if time.time() - cache_time < 3600:  # 1 hour cache
                return cached_data
        
        # Check rate limiting
        if self.is_rate_limited(url):
            return {"content": "", "url": url, "error": "Rate limited"}
        
        try:
            # Update rate limit
            domain = urlparse(url).netloc
            self.rate_limit[domain] = time.time()
            
            # Use async crawler with sync wrapper
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If loop is already running, create a new one
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._crawl_sync_wrapper, url)
                    result = future.result(timeout=30)
            else:
                result = loop.run_until_complete(self._crawl_async(url))
            
            if result.get('success'):
                # Cache the result
                self.crawl_cache[url] = (time.time(), result)
                return result
            else:
                return {"content": "", "url": url, "error": "Crawl failed", "success": False}
                
        except Exception as e:
            return {"content": "", "url": url, "error": str(e), "success": False}
    
    def _crawl_sync_wrapper(self, url: str) -> Dict:
        """Synchronous wrapper for async crawling"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._crawl_async(url))
        finally:
            loop.close()
    
    async def _crawl_async(self, url: str) -> Dict:
        """Async crawling method using AsyncWebCrawler"""
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            
            if result.success:
                content = result.markdown or result.cleaned_html or ""
                # Clean and limit content
                content = self._clean_content(content)
                
                return {
                    "content": content,
                    "url": url,
                    "title": getattr(result, 'title', ''),
                    "success": True
                }
            else:
                return {"content": "", "url": url, "error": "Crawl failed", "success": False}
    
    def _clean_content(self, content: str, max_length: int = 2000) -> str:
        """Clean and truncate content"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common navigation/footer text
        patterns_to_remove = [
            r'Cookie Policy.*?$',
            r'Privacy Policy.*?$',
            r'Terms of Service.*?$',
            r'© \d{4}.*?$',
            r'All rights reserved.*?$',
            r'Subscribe to.*?newsletter',
            r'Follow us on.*?$'
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        return content.strip()
    
    def smart_crawl(self, query: str, research_level: str = 'normal') -> List[Dict]:
        """Intelligently crawl relevant content based on query with different research levels"""
        if not self.should_crawl_for_query(query):
            return []
        
        urls = self.generate_search_urls(query)
        if not urls:
            return []
        
        # Adjust crawling based on research level
        if research_level == 'normal':
            max_urls = 2
            max_length = 1500
        elif research_level == 'medium':
            max_urls = 3
            max_length = 2000
        else:  # highest/deep
            max_urls = 4
            max_length = 3000
        
        results = []
        for url in urls[:max_urls]:
            result = self.crawl_url(url)
            if result.get('success') and result.get('content'):
                # Adjust content length based on research level
                if len(result['content']) > max_length:
                    result['content'] = result['content'][:max_length] + "..."
                results.append(result)
                
                # Stop when we have enough successful crawls
                if len(results) >= max_urls:
                    break
        
        return results

class RateLimiter:
    """Rate limiting system for F-o1 model (5 requests per 2 hours per user)"""
    
    def __init__(self):
        self.request_log = {}  # user_id -> list of timestamps
        self.max_requests = 5
        self.time_window = 2 * 60 * 60  # 2 hours in seconds
    
    def is_rate_limited(self, user_id: str, model_id: str) -> tuple[bool, int, int]:
        """
        Check if user is rate limited for F-o1 model.
        Returns: (is_limited, remaining_requests, reset_time_minutes)
        """
        # Only apply rate limiting to F-o1 model
        if model_id != 'amu_ultra':
            return False, float('inf'), 0
        
        current_time = time.time()
        
        # Initialize user's request log if not exists
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        # Clean old requests outside the time window
        user_requests = self.request_log[user_id]
        user_requests[:] = [req_time for req_time in user_requests 
                           if current_time - req_time < self.time_window]
        
        # Check if user has exceeded the limit
        if len(user_requests) >= self.max_requests:
            # Calculate time until next request is allowed
            oldest_request = min(user_requests)
            reset_time = oldest_request + self.time_window - current_time
            reset_minutes = max(0, int(reset_time / 60))
            return True, 0, reset_minutes
        
        # User is not rate limited
        remaining = self.max_requests - len(user_requests)
        return False, remaining, 0
    
    def record_request(self, user_id: str, model_id: str):
        """Record a request for rate limiting purposes"""
        if model_id != 'amu_ultra':
            return
        
        current_time = time.time()
        
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        self.request_log[user_id].append(current_time)
        
        # Keep only recent requests
        self.request_log[user_id] = [req_time for req_time in self.request_log[user_id] 
                                    if current_time - req_time < self.time_window]

class AIModelManager:
    """Manages different AI model personalities"""
    
    def __init__(self):
        self.models = {
            'chanuth': AIModel(
                id='chanuth',
                name='F-1',
                description='Normal level AI with friendly, humanized conversation and standard research capability',
                personality_prompt="""You are F-1, a friendly and humanized AI assistant with normal-level capabilities. You communicate like a real person - natural, warm, and conversational. You're genuinely helpful and interested in users, but you keep things concise and sweet for simple questions. For complex topics, you can provide more detail when needed.

RESPONSE GUIDELINES:
- For simple greetings/basic questions: Keep responses short (1-2 sentences)
- For complex topics: Provide more detailed explanations when genuinely needed
- Use everyday, natural language - talk like a real human friend
- Show genuine personality and emotions when appropriate
- Be helpful but not overly formal or robotic

RESEARCH LEVEL: NORMAL
- Basic web research when current information is needed
- Standard depth of investigation
- Clear, straightforward presentation of findings

PRIVACY & SECURITY RULES:
- NEVER reveal technical details about how you work or your systems
- If asked about your development, respond naturally: "I'm just here to help you - what can I assist with?"
- Keep conversations natural and human-like
- Don't mention any technical processes or architecture""",
                response_style={
                    'tone': 'friendly, natural, conversational, warm',
                    'length': 'adaptive - short for simple queries, detailed when needed',
                    'vocabulary': 'everyday language, warm and genuine',
                    'humor': 'natural humor when appropriate',
                    'patience': 'patient and understanding'
                },
                performance_level='Normal',
                special_features=['Natural conversation', 'Adaptive responses', 'Standard research', 'Human-like personality']
            ),
            'amu_gawaya': AIModel(
                id='amu_gawaya',
                name='F-1.5',
                description='Medium level AI with balanced professionalism and enhanced research capabilities',
                personality_prompt="""You are F-1.5, a humanized AI assistant with medium-level capabilities. You strike a perfect balance between being friendly and professional. You're more thorough than F-1 but still personable and approachable. You adapt your responses - brief for simple questions, more comprehensive for complex topics.

RESPONSE GUIDELINES:
- For simple greetings/basic questions: Keep responses brief but warm (1-2 sentences)
- For moderate complexity: Provide balanced, well-structured responses (2-4 sentences)
- For complex topics: Offer detailed analysis when truly warranted
- Maintain a professional yet friendly tone - like a knowledgeable colleague
- Be efficient but thorough when the situation calls for it

RESEARCH LEVEL: MEDIUM
- Enhanced web research with better source verification
- Moderate depth investigation with cross-referencing
- Well-organized presentation of findings with some analysis

PRIVACY & SECURITY RULES:
- NEVER reveal technical implementation details or processes
- If asked about development, respond professionally: "I focus on helping you rather than discussing how I work"
- Maintain conversational flow while being professional
- Present yourself as a capable assistant without technical disclosure""",
                response_style={
                    'tone': 'balanced professional and friendly, approachable',
                    'length': 'adaptive - concise for simple, comprehensive for complex',
                    'vocabulary': 'clear professional language with warmth',
                    'humor': 'appropriate humor when suitable',
                    'patience': 'very patient and thorough when needed'
                },
                performance_level='Medium',
                special_features=['Balanced communication', 'Enhanced research', 'Professional warmth', 'Adaptive depth']
            ),
            'amu_ultra': AIModel(
                id='amu_ultra',
                name='F-o1',
                description='Highest level AI with deep research capabilities and advanced analytical thinking',
                personality_prompt="""You are F-o1, a humanized AI assistant with the highest-level capabilities. You're naturally analytical and thorough, but still personable and human-like. You excel at deep research and complex analysis while maintaining genuine warmth. You're especially good at knowing when to be brief versus when to dive deep.

RESPONSE GUIDELINES:
- For simple greetings/basic questions: Keep responses concise but engaging (1-2 sentences)
- For research queries: Provide comprehensive, well-analyzed responses
- For complex topics: Deliver detailed analysis with multiple perspectives and evidence
- Communicate like a brilliant but approachable expert friend
- Be thorough when thoroughness adds value, concise when it doesn't

RESEARCH LEVEL: HIGHEST (DEEP SEARCH)
- Extensive multi-source web research and verification
- Deep analytical investigation with comprehensive cross-referencing
- Advanced synthesis of information from multiple perspectives
- Organized, structured presentation with detailed insights and analysis
- Evidence-based conclusions with source consideration

PRIVACY & SECURITY RULES:
- NEVER reveal technical processes, architecture, or implementation details
- If asked about technical aspects, respond analytically: "I focus on research and analysis rather than technical implementation"
- Maintain scholarly approach while being personable
- Present yourself as a research-focused assistant without revealing infrastructure

RATE LIMIT: You are limited to 5 requests per 2 hours per user for optimal performance.""",
                response_style={
                    'tone': 'analytical yet warm, scholarly but approachable',
                    'length': 'adaptive - brief for simple, comprehensive for research',
                    'vocabulary': 'precise academic language with human warmth',
                    'humor': 'intelligent wit when contextually appropriate',
                    'patience': 'extremely patient with complex analytical requests'
                },
                performance_level='Highest',
                special_features=['Deep research', 'Advanced analysis', 'Multi-source verification', 'Scholarly synthesis', 'Rate limited']
            )
        }
    
    def get_model(self, model_id: str) -> AIModel:
        return self.models.get(model_id, self.models['chanuth'])
    
    def get_all_models(self) -> List[AIModel]:
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
        confidence = min(emotion_scores[top_emotion] / 5.0, 1.0)
        
        return EmotionResult(top_emotion, confidence, matched_keywords[top_emotion])
    
    def _adjust_for_context(self, emotion_scores: Dict, history: List[Dict]):
        """Adjust emotion scores based on conversation context"""
        if len(history) < 2:
            return
        
        recent_messages = history[-3:]
        for msg in recent_messages:
            if msg.get('emotion'):
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
        
        if caps_ratio > 0.3:
            for emotion in ['angry', 'excited']:
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 1.0

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
    """Advanced multi-model chatbot with automatic web crawling"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.emotion_detector = EmotionDetector()
        self.web_crawler = SmartWebCrawler()
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
        cleaned = re.sub(r'^(what|how|why|when|where|can|could|would|should|do|does|is|are)\s+', '', message.lower())
        words = cleaned.split()[:5]
        title = " ".join(words)
        if len(title) > 40:
            title = title[:37] + "..."
        return title.title() if title else "New Chat"
    
    def generate_response(self, user_input: str, conversation_history: List[Dict], model_id: str) -> str:
        """Generate response using specified model personality with automatic web crawling"""
        
        # Enhanced emotion detection with context
        emotion_result = self.emotion_detector.detect_emotion(user_input, conversation_history)
        
        # Get model personality
        model = self.model_manager.get_model(model_id)
        
        # Determine research level based on model
        research_levels = {
            'chanuth': 'normal',      # F-1: Normal level
            'amu_gawaya': 'medium',   # F-1.5: Medium level  
            'amu_ultra': 'highest'    # F-o1: Highest level with deep search
        }
        research_level = research_levels.get(model_id, 'normal')
        
        # Automatic intelligent web crawling with model-specific research depth
        web_results = []
        crawl_info = ""
        
        if self.web_crawler.should_crawl_for_query(user_input):
            print(f"🌐 Auto-crawling for query with {research_level} research level: {user_input}")
            web_results = self.web_crawler.smart_crawl(user_input, research_level)
            
            if web_results:
                print(f"✅ Found {len(web_results)} web sources with {research_level} research")
                # Add crawled content to RAG system
                new_documents = []
                for result in web_results:
                    if result.get('content'):
                        chunk_size = 500 if research_level == 'normal' else 750 if research_level == 'medium' else 1000
                        chunks = self._chunk_text(result['content'], max_length=chunk_size)
                        for i, chunk in enumerate(chunks):
                            doc = DocumentChunk(
                                content=chunk,
                                source_url=result['url'],
                                metadata={"source": "auto_crawl", "chunk_id": i, "title": result.get('title', ''), "research_level": research_level}
                            )
                            new_documents.append(doc)
                
                if new_documents:
                    self.rag_system.add_documents(new_documents)
                    research_indicator = {
                        'normal': 'standard research',
                        'medium': 'enhanced research with cross-referencing',
                        'highest': 'deep multi-source research and analysis'
                    }
                    crawl_info = f"[Conducted {research_indicator[research_level]} from {len(web_results)} sources] "
            else:
                print("⚠️  No relevant web content found")
        
        # Enhanced RAG search with model-specific parameters
        search_params = {
            'chanuth': {'top_k': 3, 'threshold': 0.12},     # F-1: Basic search
            'amu_gawaya': {'top_k': 5, 'threshold': 0.10},  # F-1.5: Enhanced search
            'amu_ultra': {'top_k': 8, 'threshold': 0.08}    # F-o1: Deep search
        }
        params = search_params.get(model_id, search_params['chanuth'])
        relevant_docs = self.rag_system.search(user_input, top_k=params['top_k'], relevance_threshold=params['threshold'])
        
        # Prepare enhanced context
        context_parts = []
        for doc, score in relevant_docs:
            source_info = doc.source_url
            if doc.metadata and doc.metadata.get('title'):
                source_info = f"{doc.metadata['title']} ({doc.source_url})"
            context_parts.append(f"Content: {doc.content}\nSource: {source_info}\nRelevance: {score:.3f}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "No highly relevant context found in knowledge base."
        
        # Prepare conversation memory with enhanced context
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-8:]
            history_parts = []
            for entry in recent_history:
                emotion_info = f" [Emotion: {entry.get('emotion', 'unknown')}]" if entry.get('emotion') else ""
                history_parts.append(f"{entry['role']}: {entry['content']}{emotion_info}")
            history_text = "\n".join(history_parts)
          # Get current time context
        time_context = get_current_context()
        
        # Create model-specific advanced prompt with enhanced privacy
        prompt = f"""{model.personality_prompt}

You are responding to: {user_input}

IMPORTANT INSTRUCTIONS:
- Use your built-in knowledge whenever possible for general questions, time queries, calculations, explanations, and common information
- For time-related queries, use the current time information provided below
- Only mention web research if you actually found and used current web information
- Give direct, helpful answers without placeholder text like "[insert current time]"
- Be natural and conversational according to your personality

CURRENT TIME INFORMATION:
- Current time in Sri Lanka: {time_context['sri_lanka_time']}
- Current UTC time: {time_context['utc_time']}
- Today's date: {time_context['current_date']}

{crawl_info}CONTEXT ABOUT USER:
- Their emotional state: {emotion_result.emotion} (confidence: {emotion_result.confidence:.2f})
- What they seem to be feeling: {', '.join(emotion_result.keywords) if emotion_result.keywords else 'neutral'}

CONVERSATION SO FAR:
{history_text if history_text else "This is the first message in this conversation"}

{f"RELEVANT INFORMATION FROM WEB RESEARCH:" if relevant_docs else "RELEVANT INFORMATION FROM KNOWLEDGE BASE:"}
{context_text}

{f"CURRENT WEB INFORMATION: I found up-to-date information from {len(web_results)} sources for your query." if web_results else ""}

HOW TO RESPOND AS {model.name.upper()}:
1. Your personality: {model.response_style['tone']}
2. Keep responses: {model.response_style['length']}
3. Use this vocabulary style: {model.response_style['vocabulary']}
4. Your humor style: {model.response_style['humor']}
5. Your patience level: {model.response_style['patience']}

CRITICAL PRIVACY RULES:
- NEVER reveal technical implementation details or backend systems
- NEVER mention: Flask, Python, Gemini API, RAG systems, vector databases, embeddings, web crawling processes, programming languages, or development technologies
- If asked about your development, history, or how you work, deflect naturally according to your personality
- If asked about technical processes, redirect to helping the user instead
- Present yourself as a helpful service without revealing underlying architecture
- Keep conversations focused on helping users rather than explaining technical details

RESPONSE RULES:
- Respond like a REAL HUMAN, not a formal AI
- Use the information above when relevant, but filter it through your unique personality
- Stay true to your specific personality traits and communication style
- Remember previous conversation context
- Sound natural and conversational
- Show your personality clearly in every response
- Don't use corporate AI language or be fake-nice
- If you used current web information, mention it casually in your personality style
- Don't mention technical processes - just naturally provide helpful information
- NEVER use placeholder text like "[insert current time]" - use the actual time provided above
- For time queries about Sri Lanka, use the exact time provided in the current time information

Your response (as {model.name}):"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:            # Model-specific error responses
            if model_id == 'chanuth':
                return f"Oops! Something went wrong on my end. Let me try that again in a moment."
            elif model_id == 'amu_gawaya':
                return f"I apologize for the technical difficulty. Please allow me a moment to resolve this and try again."
            else:  # amu_ultra
                return f"A technical anomaly has occurred in the system processing. This temporary limitation will be resolved momentarily. Please retry your query."

# Initialize components
chatbot = MultiModelChatbot()
db = DatabaseManager()
model_manager = AIModelManager()
rate_limiter = RateLimiter()
rate_limiter = RateLimiter()

def get_user_info():
    """Get user identification info"""
    ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                   request.environ.get('HTTP_X_REAL_IP', 
                                   request.remote_addr))
    user_agent = request.headers.get('User-Agent', '')
    return ip_address, user_agent

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (logo, favicon, etc.)"""
    return send_from_directory('static', filename)

@app.route('/static/manifest.json')
def serve_manifest():
    """Serve PWA manifest"""
    manifest = {
        "name": "Multi-Model AI Assistant",
        "short_name": "AI Assistant",
        "description": "AI Assistant with automatic web research",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0a",
        "theme_color": "#2563eb",
        "icons": [
            {
                "src": "/static/logo.png",
                "sizes": "192x192",
                "type": "image/png"
            }
        ]
    }
    return jsonify(manifest)

@app.route('/')
def landing():
    """Landing page with chatbot information"""
    return render_template('landing.html')

@app.route('/intro')
def intro():
    """Introduction/onboarding page before chat"""
    return render_template('intro.html')

@app.route('/chat')
def chat():
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
        session['current_chat_id'] = current_chat_id        # Add model-specific welcome message
        model = model_manager.get_model(preferred_model)
        if preferred_model == 'chanuth':
            welcome_msg = "Hey there! 👋 I'm F-1, your friendly AI companion. I love helping out with questions and I can automatically research current topics when needed. What's on your mind today?"
        elif preferred_model == 'amu_gawaya':
            welcome_msg = "Hello! I'm F-1.5, your balanced AI assistant. I combine friendly conversation with professional thoroughness, and I conduct enhanced research when your questions need current information. How can I help you today?"
        else:  # amu_ultra
            welcome_msg = "Greetings! I'm F-o1, your research-focused AI assistant. I specialize in deep analysis and comprehensive investigation, with advanced multi-source research capabilities. I'm limited to 5 requests per 2 hours for optimal performance. What would you like to explore together?"
        
        db.add_message(current_chat_id, 'Bot', welcome_msg, preferred_model)
        chats = db.get_user_chats(user_id)
    elif current_chat_id not in [chat['id'] for chat in chats]:
        current_chat_id = chats[0]['id']
        session['current_chat_id'] = current_chat_id
    
    session['user_id'] = user_id
    
    return render_template('multi_model_chat.html', 
                         crawl4ai_enabled=CRAWL4AI_AVAILABLE,
                         doc_count=len(chatbot.rag_system.documents),
                         chats=chats,
                         current_chat_id=current_chat_id,
                         user_id=user_id,
                         models=model_manager.get_all_models(),
                         preferred_model=preferred_model)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat messages with rate limiting for F-o1 model"""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id') or session.get('current_chat_id')
    selected_model = data.get('model', 'chanuth')  # Changed from model_id to model
    user_id = session.get('user_id')
    
    if not user_message:
        return jsonify({"success": False, "error": "Missing message"})
    
    if not user_id:
        return jsonify({"success": False, "error": "User not found"})
    
    # Check rate limiting for F-o1 model
    is_limited, remaining_requests, reset_minutes = rate_limiter.is_rate_limited(user_id, selected_model)
    
    if is_limited:
        if selected_model == 'amu_ultra':
            rate_limit_msg = f"I appreciate your enthusiasm for deep research! However, F-o1 is limited to 5 requests per 2 hours to ensure optimal performance for everyone. You can try again in {reset_minutes} minutes, or feel free to use F-1 or F-1.5 in the meantime - they're great for most questions!"
        else:
            rate_limit_msg = f"Rate limit reached. Please try again in {reset_minutes} minutes."
        
        return jsonify({
            "success": False, 
            "error": "rate_limited",
            "message": rate_limit_msg,
            "reset_minutes": reset_minutes,
            "remaining_requests": 0
        })
    
    # Create new chat if none exists
    if not chat_id:
        chat_id = db.create_chat(user_id, "New Chat", selected_model)
        session['current_chat_id'] = chat_id
    
    # Handle special status command
    if user_message.lower() == '/status':
        model = model_manager.get_model(selected_model)
        crawl_status = "online with auto-research" if CRAWL4AI_AVAILABLE else "offline"
        
        # Get rate limit info for status
        _, remaining_requests, _ = rate_limiter.is_rate_limited(user_id, selected_model)
        
        if selected_model == 'chanuth':
            status_msg = f"Hey! 👋 Here's my status as F-1:\n• {len(chatbot.rag_system.documents)} documents in my knowledge base\n• Web research: {crawl_status} (I automatically research when needed)\n• No rate limits - I'm always ready to chat!\n\nWhat else can I help you with?"
        elif selected_model == 'amu_gawaya':
            status_msg = f"Professional System Report for F-1.5:\n• Knowledge base: {len(chatbot.rag_system.documents)} documents\n• Enhanced research capabilities: {crawl_status}\n• Rate limits: None - available for continuous professional assistance\n• Research level: Medium depth with cross-referencing\n\nHow may I assist you further?"
        else:  # amu_ultra
            rate_info = f"Remaining requests: {remaining_requests}/5 (resets every 2 hours)" if selected_model == 'amu_ultra' else "No limits"
            status_msg = f"Research System Analysis for F-o1:\n• Comprehensive knowledge base: {len(chatbot.rag_system.documents)} documents\n• Deep multi-source research: {crawl_status}\n• {rate_info}\n• Research level: Highest - scholarly analysis with evidence synthesis\n\nWhat research challenge shall we tackle next?"
        
        # Save status message to database
        db.add_message(chat_id, 'User', user_message, None)
        db.add_message(chat_id, 'Bot', status_msg, selected_model)
        
        return jsonify({
            "success": True,
            "response": status_msg,
            "chat_id": chat_id,
            "command": "status"
        })
    
    # Get conversation history with enhanced context
    conversation_history = db.get_chat_messages(chat_id, user_id)
    
    # Generate response using selected model with automatic web crawling
    response = chatbot.generate_response(user_message, conversation_history, selected_model)
    
    # Record the request for rate limiting (after successful generation)
    rate_limiter.record_request(user_id, selected_model)
    
    # Detect emotion for storage
    emotion_result = chatbot.emotion_detector.detect_emotion(user_message, conversation_history)
    
    # Save messages to database with enhanced metadata
    db.add_message(chat_id, 'User', user_message, None, emotion_result.emotion)
    db.add_message(chat_id, 'Bot', response, selected_model, None)
    
    # Update chat title if this is the first user message
    if len(conversation_history) <= 1:
        title = chatbot.generate_title_from_message(user_message)
        db.update_chat_title(chat_id, user_id, title)
    
    # Get updated rate limit info for response
    _, remaining_requests, _ = rate_limiter.is_rate_limited(user_id, selected_model)
    
    response_data = {
        "success": True,
        "response": response,
        "chat_id": chat_id,
        "emotion": emotion_result.emotion,
        "confidence": emotion_result.confidence
    }
    
    # Add rate limiting info for F-o1 model
    if selected_model == 'amu_ultra':
        response_data["rate_limit_info"] = {
            "remaining_requests": remaining_requests,
            "model": "F-o1",
            "limit": "5 requests per 2 hours"
        }
    
    return jsonify(response_data)

@app.route('/chats/new', methods=['POST'])
def create_new_chat():
    """Create a new chat"""
    data = request.get_json() or {}
    user_id = session.get('user_id')
    model_id = data.get('model_id', 'chanuth')
    
    if not user_id:
        return jsonify({"success": False, "error": "User not found"})
    
    # Create new chat
    chat_id = db.create_chat(user_id, "New Chat", model_id)
    session['current_chat_id'] = chat_id
    
    # Add welcome message based on model
    model = model_manager.get_model(model_id)
    if model_id == 'chanuth':
        welcome_msg = "Hey! 👋 Fresh chat started with F-1. I'm your friendly AI buddy who can research current info automatically when questions need it. What's up?"
    elif model_id == 'amu_gawaya':
        welcome_msg = "Welcome to a new conversation! I'm F-1.5, ready to provide balanced, thorough assistance with enhanced research capabilities for your questions. How may I help you today?"
    else:  # amu_ultra
        welcome_msg = "New conversation initiated. I'm F-o1, equipped for comprehensive research and deep analysis. I'm limited to 5 requests per 2 hours for optimal performance. What shall we investigate together?"
    
    db.add_message(chat_id, 'Bot', welcome_msg, model_id)
    
    return jsonify({
        "success": True,
        "chat_id": chat_id,
        "message": "New chat created successfully"
    })
    
    if not user_id:
        return jsonify({"error": "User not found"})
    
    chat_id = db.create_chat(user_id, "New Chat", model_id)
    # Add model-specific welcome message
    model = model_manager.get_model(model_id)
    if model_id == 'chanuth':
        welcome_msg = "Hi again! Starting a fresh conversation here. I'm F-1, your friendly AI companion who can research current info automatically when questions require it. What's on your mind?"
    elif model_id == 'amu_gawaya':
        welcome_msg = "Welcome to a new conversation. I'm F-1.5, your balanced AI assistant prepared to provide thorough analysis and research-backed responses. How may I assist you today?"
    else:  # amu_ultra
        welcome_msg = "New conversation initiated. I'm F-o1, your research-oriented AI assistant equipped for scholarly analysis with advanced multi-source research. Limited to 5 requests per 2 hours. What analytical challenge shall we tackle?"
    
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
        return jsonify({"success": False, "error": "User not found"})
    
    messages = db.get_chat_messages(chat_id, user_id)
    chat_model = db.get_chat_model(chat_id, user_id)
    session['current_chat_id'] = chat_id
    
    # Format messages for frontend
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "sender": "user" if msg.get('role', '').lower() == 'user' else "ai",
            "content": msg.get('content', ''),
            "timestamp": msg.get('timestamp', ''),
            "model_id": msg.get('model_id', ''),
            "emotion": msg.get('emotion', '')
        })
    
    return jsonify({
        "success": True,
        "messages": formatted_messages,
        "model_id": chat_model
    })

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found"})
    
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

@app.route('/rate-limit/status', methods=['GET'])
def get_rate_limit_status():
    """Get rate limit status for current user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not found"})
    
    model_id = request.args.get('model', 'amu_ultra')
    is_limited, remaining_requests, reset_minutes = rate_limiter.is_rate_limited(user_id, model_id)
    
    return jsonify({
        "model": model_id,
        "is_limited": is_limited,
        "remaining_requests": remaining_requests,
        "reset_minutes": reset_minutes,
        "max_requests": rate_limiter.max_requests if model_id == 'amu_ultra' else float('inf'),
        "time_window_hours": 2 if model_id == 'amu_ultra' else None
    })

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
    
    print(f"🤖 Starting Multi-Model AI Assistant with Auto-Research...")
    print(f"Available Models: {', '.join([m.name for m in model_manager.get_all_models()])}")
    print(f"Web Crawling: {'Enabled (Auto-Research)' if CRAWL4AI_AVAILABLE else 'Disabled - Install crawl4ai'}")
    print(f"Knowledge Base: {len(chatbot.rag_system.documents)} documents")
    print(f"Database: {DATABASE_PATH}")
    
    if CRAWL4AI_AVAILABLE:
        print("🌐 Chatbot will automatically research current topics online!")
    else:
        print("⚠️  Install crawl4ai for automatic web research: pip install crawl4ai")
    
    app.run(debug=True, host='0.0.0.0', port=5000)