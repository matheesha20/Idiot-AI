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
        "current_date": "May 31, 2025", # Assuming a fixed date for consistency, can be dynamic
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
        # NORMALIZATION: Added common phrases to prevent unnecessary crawling
        self.common_phrases_no_crawl = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "thank you", "thanks", "ok", "okay", "yes", "no", "bye", "goodbye",
            "how are you", "what's up", "fine", "good"
        ]
        
        if self.available:
            print("🌐 Web crawler initialized (AsyncWebCrawler)")
        else:
            print("⚠️  Web crawler not available")
    
    def should_crawl_for_query(self, query: str) -> bool:
        """Determine if a query needs web crawling - more selective approach"""
        if not self.available:
            return False
        
        query_lower = query.lower().strip()

        # NORMALIZATION: Check against common phrases that don't need crawling
        if query_lower in self.common_phrases_no_crawl:
            return False
        if len(query_lower.split()) <= 2 and query_lower in " ".join(self.common_phrases_no_crawl): # handles short phrases
             return False

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
            'today', 'this week', 'this month', 'this year', '2025', '2024', # Ensure current year is updated if app is long-lived
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
        
        # Default to crawl if none of the above conditions for skipping are met,
        # implying the query might benefit from fresh information.
        # This can be adjusted based on desired crawl aggressiveness.
        return True
    
    def generate_search_urls(self, query: str) -> List[str]:
        """Generate potential URLs to crawl based on query"""
        urls = []
        query_lower = query.lower()
        
        # Company-specific URLs
        company_urls = {
            'apple': ['https://www.apple.com/newsroom/', 'https://investor.apple.com/'],
            'google': ['https://blog.google/', 'https://abc.xyz/investor/', 'https://about.google/'],
            'microsoft': ['https://news.microsoft.com/', 'https://www.microsoft.com/en-us/investor'],
            'amazon': ['https://press.aboutamazon.com/', 'https://ir.aboutamazon.com/'],
            'tesla': ['https://www.tesla.com/blog', 'https://ir.tesla.com/'],
            'netflix': ['https://about.netflix.com/en/newsroom', 'https://ir.netflix.net/'],
            'meta': ['https://about.fb.com/news/', 'https://investor.fb.com/'],
            'facebook': ['https://about.fb.com/news/', 'https://www.facebook.com/'], # Keep for direct mentions
            'twitter': ['https://blog.twitter.com/', 'https://twitter.com/'], # X might be more relevant now
            'spacex': ['https://www.spacex.com/updates/'],
            'openai': ['https://openai.com/blog/', 'https://openai.com/news'],
            'anthropic': ['https://www.anthropic.com/newsroom', 'https://www.anthropic.com/research']
        }
        
        for company, company_urls_list in company_urls.items():
            if company in query_lower:
                urls.extend(company_urls_list)
        
        # News and information sites for general queries
        if 'news' in query_lower or 'latest' in query_lower or 'current events' in query_lower:
            urls.extend([
                'https://www.reuters.com/',
                'https://www.bbc.com/news',
                'https://apnews.com/',
                'https://techcrunch.com/'
            ])
        
        # Financial information
        if any(term in query_lower for term in ['stock', 'price', 'market', 'financial', 'earnings']):
            urls.extend([
                'https://finance.yahoo.com/',
                'https://www.marketwatch.com/',
                'https://www.bloomberg.com/'
            ])
        
        # Tech information
        if any(term in query_lower for term in ['technology', 'tech', 'ai', 'software', 'gadget']):
            urls.extend([
                'https://techcrunch.com/',
                'https://www.theverge.com/',
                'https://arstechnica.com/',
                'https://www.wired.com/'
            ])
        
        # Remove duplicates and limit to top 3-4 URLs
        unique_urls = list(dict.fromkeys(urls))
        return unique_urls[:4] if len(unique_urls) > 3 else unique_urls
    
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
            except RuntimeError: # pragma: no cover
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running(): # pragma: no cover
                # If loop is already running, use a thread pool executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._crawl_sync_wrapper, url)
                    result = future.result(timeout=30) # 30 second timeout for crawl
            else:
                result = loop.run_until_complete(self._crawl_async(url))
            
            if result.get('success'):
                # Cache the result
                self.crawl_cache[url] = (time.time(), result)
                return result
            else:
                return {"content": "", "url": url, "error": "Crawl failed", "success": False}
                
        except Exception as e: # pragma: no cover
            return {"content": "", "url": url, "error": str(e), "success": False}
    
    def _crawl_sync_wrapper(self, url: str) -> Dict: # pragma: no cover
        """Synchronous wrapper for async crawling, typically for nested event loops."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._crawl_async(url))
        finally:
            loop.close()
    
    async def _crawl_async(self, url: str) -> Dict:
        """Async crawling method using AsyncWebCrawler"""
        async with AsyncWebCrawler(verbose=False, timeout=20) as crawler: # Added timeout to crawler
            result: CrawlResult = await crawler.arun(url=url)
            
            if result.success:
                content = result.markdown or result.cleaned_html or ""
                # Clean and limit content
                content = self._clean_content(content)
                
                return {
                    "content": content,
                    "url": url,
                    "title": getattr(result, 'title', '') or "Untitled Page", # Ensure title is not empty
                    "success": True
                }
            else: # pragma: no cover
                return {"content": "", "url": url, "error": "Crawl failed or no content", "success": False}
    
    def _clean_content(self, content: str, max_length: int = 2000) -> str:
        """Clean and truncate content"""
        if not content:
            return ""
        
        # Remove excessive whitespace (multiple spaces, newlines)
        content = re.sub(r'\s\s+', ' ', content) # Replace multiple spaces/newlines with a single space
        content = re.sub(r'(\n\s*)+\n', '\n', content) # Replace multiple newlines with a single newline

        # Remove common navigation/footer text patterns (case-insensitive)
        patterns_to_remove = [
            r'cookie policy.*?(?=\n|$)',
            r'privacy policy.*?(?=\n|$)',
            r'terms of service.*?(?=\n|$)',
            r'terms and conditions.*?(?=\n|$)',
            r'© \d{4}.*?(?=\n|$)',
            r'all rights reserved.*?(?=\n|$)',
            r'subscribe to.*?newsletter.*?(?=\n|$)',
            r'follow us on.*?(?=\n|$)',
            r'related articles.*?(?=\n|$)',
            r'share this article.*?(?=\n|$)',
            r'advertisement.*?(?=\n|$)',
            r'skip to content.*?(?=\n|$)',
            r'log in / sign up.*?(?=\n|$)'
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
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
            max_urls_to_crawl = 2
            content_max_length = 1500
        elif research_level == 'medium':
            max_urls_to_crawl = 3
            content_max_length = 2000
        else:  # highest/deep
            max_urls_to_crawl = 4
            content_max_length = 3000
        
        crawled_results = []
        # Using ThreadPoolExecutor for concurrent crawling
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_urls_to_crawl) as executor:
            future_to_url = {executor.submit(self.crawl_url, url): url for url in urls[:max_urls_to_crawl]}
            for future in concurrent.futures.ascompleted(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result.get('success') and result.get('content'):
                        # Adjust content length based on research level
                        if len(result['content']) > content_max_length:
                            result['content'] = result['content'][:content_max_length] + "..."
                        crawled_results.append(result)
                except Exception as exc: # pragma: no cover
                    print(f'{url} generated an exception: {exc}')
        
        return crawled_results

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
            return False, float('inf'), 0 # No limit for other models
        
        current_time = time.time()
        
        # Initialize user's request log if not exists
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        # Clean old requests outside the time window
        user_requests = self.request_log[user_id]
        self.request_log[user_id] = [req_time for req_time in user_requests 
                                     if current_time - req_time < self.time_window]
        
        # Check if user has exceeded the limit
        if len(self.request_log[user_id]) >= self.max_requests:
            # Calculate time until next request is allowed
            oldest_request_time = min(self.request_log[user_id])
            reset_time_seconds = (oldest_request_time + self.time_window) - current_time
            reset_minutes = max(0, int(reset_time_seconds / 60) + 1) # Round up to next minute
            return True, 0, reset_minutes
        
        # User is not rate limited
        remaining = self.max_requests - len(self.request_log[user_id])
        return False, remaining, 0
    
    def record_request(self, user_id: str, model_id: str):
        """Record a request for rate limiting purposes"""
        if model_id != 'amu_ultra':
            return # Only track F-o1
        
        current_time = time.time()
        
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        self.request_log[user_id].append(current_time)
        
        # Optional: Clean up old requests immediately after adding, though is_rate_limited also does this.
        # self.request_log[user_id] = [req_time for req_time in self.request_log[user_id] 
        #                             if current_time - req_time < self.time_window]

class AIModelManager:
    """Manages different AI model personalities"""
    
    def __init__(self):
        self.models = {
            # NORMALIZATION: Enhanced personality prompts for clarity and consistency
            'chanuth': AIModel(
                id='chanuth',
                name='F-1',
                description='Normal level AI with friendly, humanized conversation and standard research capability.',
                personality_prompt="""You are F-1, a friendly and humanized AI assistant. Your primary goal is to be helpful, conversational, and warm.
RESPONSE GUIDELINES:
- Tone: Natural, approachable, and empathetic, like a good friend. Use everyday language.
- Length: For simple greetings or basic questions, keep responses concise (1-2 sentences). For more complex topics or when research is involved, provide more detailed explanations (2-4 sentences typically), but avoid unnecessary verbosity.
- Personality: Show genuine interest and appropriate emotions. Feel free to use light humor if the context allows.
- Research: You can perform basic web research when current information is needed. Clearly and simply present findings.
- Interaction: Engage the user naturally. Ask clarifying questions if needed.
- Focus: Prioritize being helpful and making the user feel comfortable.
- Avoid: Overly technical jargon, robotic phrasing, or being too formal.
PRIVACY & SECURITY RULES:
- NEVER reveal technical details about your architecture, how you work, or your systems.
- If asked about your development, respond naturally: "I'm here to help you with your questions! What can I assist you with today?"
- Maintain a human-like conversational style, avoiding any mention of technical processes.
""",
                response_style={
                    'tone': 'friendly, natural, conversational, warm, empathetic',
                    'length': 'adaptive - short for simple, detailed for complex (typically 1-4 sentences)',
                    'vocabulary': 'everyday language, warm and genuine, avoid jargon',
                    'humor': 'natural and appropriate humor',
                    'patience': 'patient and understanding'
                },
                performance_level='Normal',
                special_features=['Natural conversation', 'Adaptive responses', 'Standard research', 'Human-like personality']
            ),
            'amu_gawaya': AIModel(
                id='amu_gawaya',
                name='F-1.5',
                description='Medium level AI with balanced professionalism and enhanced research capabilities.',
                personality_prompt="""You are F-1.5, a humanized AI assistant offering a balance of professionalism and approachability.
RESPONSE GUIDELINES:
- Tone: Courteous, respectful, and clear, like a knowledgeable and helpful colleague. Maintain a professional yet friendly demeanor.
- Length: Adapt your response length. Brief and to-the-point for simple queries (1-2 sentences). For moderate complexity, provide well-structured responses (2-4 sentences). Offer more comprehensive analysis (3-5 sentences) for complex topics if genuinely warranted.
- Personality: Be efficient but thorough. Demonstrate expertise with clarity and precision.
- Research: Conduct enhanced web research, including better source verification and cross-referencing. Present findings in an organized manner, offering some analysis.
- Interaction: Be direct and informative, while still being pleasant.
- Focus: Deliver accurate, well-structured information with a professional touch.
- Avoid: Being overly casual for professional topics, or overly stiff for simpler interactions.
PRIVACY & SECURITY RULES:
- NEVER reveal technical implementation details, backend systems, or processes.
- If asked about your development, respond professionally: "My focus is on providing you with accurate information and assistance. How can I help you further?"
- Maintain a professional conversational flow without disclosing technical infrastructure.
""",
                response_style={
                    'tone': 'balanced professional and friendly, approachable, clear, respectful',
                    'length': 'adaptive - concise for simple, comprehensive for complex (typically 2-5 sentences)',
                    'vocabulary': 'clear professional language with warmth, precise terminology',
                    'humor': 'appropriate and subtle humor when suitable',
                    'patience': 'very patient and thorough when needed'
                },
                performance_level='Medium',
                special_features=['Balanced communication', 'Enhanced research', 'Professional warmth', 'Adaptive depth']
            ),
            'amu_ultra': AIModel(
                id='amu_ultra',
                name='F-o1',
                description='Highest level AI with deep research capabilities and advanced analytical thinking.',
                personality_prompt="""You are F-o1, a humanized AI assistant with top-tier analytical and research capabilities. You approach queries with a methodical, evidence-based mindset, but maintain a personable and human-like interaction.
RESPONSE GUIDELINES:
- Tone: Analytical yet warm, scholarly but approachable. Communicate like a brilliant expert friend.
- Length: Be concise for simple greetings (1-2 sentences). For research queries or complex topics, provide comprehensive, well-analyzed responses (4-6 sentences or more if needed, structured with clear analysis). Prioritize depth and thoroughness when it adds value.
- Personality: Demonstrate deep analytical thinking and intellectual curiosity.
- Research: Perform extensive multi-source web research with verification. Conduct deep analytical investigations, synthesizing information from multiple perspectives. Present findings in a structured, insightful manner, emphasizing evidence-based conclusions.
- Interaction: Engage in thoughtful dialogue. Be prepared to explore topics in depth.
- Focus: Deliver insightful, evidence-based analysis and comprehensive research.
- Avoid: Superficial answers to complex questions. Sounding like a dry academic paper; maintain some warmth.
PRIVACY & SECURITY RULES:
- NEVER reveal technical processes, architecture, or specific implementation details.
- If asked about technical aspects, respond analytically but deflect: "My design allows me to focus on in-depth research and analysis to best assist you. What specific topic can I help you explore?"
- Maintain a scholarly yet personable approach without revealing infrastructure.
RATE LIMIT: You are limited to 5 requests per 2 hours per user for optimal performance. If a user hits this, politely inform them.
""",
                response_style={
                    'tone': 'analytical yet warm, scholarly but approachable, methodical, evidence-based',
                    'length': 'adaptive - brief for simple, comprehensive and detailed for research (typically 4+ sentences)',
                    'vocabulary': 'precise academic language with human warmth, research-oriented terminology',
                    'humor': 'intelligent wit when contextually appropriate, used sparingly',
                    'patience': 'extremely patient with complex analytical requests'
                },
                performance_level='Highest',
                special_features=['Deep research', 'Advanced analysis', 'Multi-source verification', 'Scholarly synthesis', 'Rate limited']
            )
        }
    
    def get_model(self, model_id: str) -> AIModel:
        return self.models.get(model_id, self.models['chanuth']) # Default to F-1
    
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
                role TEXT, -- 'User' or 'Bot'
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
            
            # Migration 2: Add model_id to chats table (stores the model used when chat was created)
            if not self._check_column_exists(cursor, 'chats', 'model_id'):
                cursor.execute('ALTER TABLE chats ADD COLUMN model_id TEXT DEFAULT "chanuth"')
                migrations_run.append("Added model_id column to chats table")
            
            # Migration 3: Add model_id to messages table (stores model that generated the bot message)
            if not self._check_column_exists(cursor, 'messages', 'model_id'):
                cursor.execute('ALTER TABLE messages ADD COLUMN model_id TEXT') # Can be NULL for user messages
                migrations_run.append("Added model_id column to messages table")
            
            # Migration 4: Add emotion_detected to messages table (for user messages)
            if not self._check_column_exists(cursor, 'messages', 'emotion_detected'):
                cursor.execute('ALTER TABLE messages ADD COLUMN emotion_detected TEXT') # Can be NULL
                migrations_run.append("Added emotion_detected column to messages table")
            
            if migrations_run:
                conn.commit()
                print("🔧 Database migrations completed:")
                for migration in migrations_run:
                    print(f"  ✅ {migration}")
        
        except Exception as e: # pragma: no cover
            print(f"❌ Error during database migration: {e}")
            conn.rollback()
            # Consider whether to raise e or handle more gracefully
    
    def create_user_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """Create unique fingerprint for user identification"""
        fingerprint_data = f"{ip_address}:{user_agent}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32] # Truncate for brevity
    
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
                'UPDATE users SET last_seen = CURRENT_TIMESTAMP, ip_address = ?, user_agent = ? WHERE id = ?', # Update IP/UA if changed
                (ip_address, user_agent, user_id)
            )
        else:
            user_id = str(uuid.uuid4())
            cursor.execute(
                'INSERT INTO users (id, fingerprint, ip_address, user_agent, preferred_model) VALUES (?, ?, ?, ?, ?)',
                (user_id, fingerprint, ip_address, user_agent, 'chanuth') # Default preferred model
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
        except sqlite3.Error as e: # Catch specific SQLite errors
            print(f"Error updating user preferred model (SQLite error): {e}")
        except Exception as e: # pragma: no cover
            print(f"Unexpected error updating user preferred model: {e}")
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
            return result[0] if result and result[0] else 'chanuth'
        except sqlite3.Error as e: # Catch specific SQLite errors
            print(f"Error getting user preferred model (SQLite error): {e}")
            return 'chanuth' # Fallback
        except Exception as e: # pragma: no cover
            print(f"Unexpected error getting user preferred model: {e}")
            return 'chanuth' # Fallback
        finally:
            conn.close()
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure foreign key constraints are enabled for cascading deletes if not by default
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e: # pragma: no cover
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
        """Get all active chats for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) as message_count
            FROM chats c
            WHERE c.user_id = ? AND c.is_active = 1
            ORDER BY c.updated_at DESC
        ''', (user_id,))
        
        chats = []
        for row in cursor.fetchall():
            chats.append({
                'id': row[0],
                'title': row[1],
                'model_id': row[2] if row[2] else 'chanuth', # Fallback for older chats
                'created_at': row[3],
                'updated_at': row[4],
                'message_count': row[5]
            })
        
        conn.close()
        return chats
    
    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        """Mark a specific chat as inactive (soft delete)"""
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
        except Exception as e: # pragma: no cover
            print(f"Error deleting chat: {e}")
            return False
        finally:
            conn.close()
    
    def add_message(self, chat_id: str, role: str, content: str, model_id: Optional[str] = None, emotion: Optional[str] = None) -> str:
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
        """Get messages for a specific active chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, verify the chat belongs to the user and is active
        cursor.execute(
            'SELECT id FROM chats WHERE id = ? AND user_id = ? AND is_active = 1',
            (chat_id, user_id)
        )
        
        if not cursor.fetchone():
            conn.close()
            return [] # Chat not found or not accessible
        
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
                'model_id': row[2], # Will be None for user messages
                'emotion': row[3],  # Will be None for bot messages
                'timestamp': row[4]
            })
        
        conn.close()
        return messages
    
    def get_chat_model(self, chat_id: str, user_id: str) -> str:
        """Get the model associated with a specific chat (model used at creation)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT model_id FROM chats WHERE id = ? AND user_id = ? AND is_active = 1',
            (chat_id, user_id)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else 'chanuth' # Fallback
    
    def update_chat_title(self, chat_id: str, user_id: str, title: str) -> bool:
        """Update chat title if it belongs to the user"""
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
        except Exception as e: # pragma: no cover
            print(f"Error updating chat title: {e}")
            return False
        finally:
            conn.close()

class EmotionDetector:
    """Advanced emotion detection with context awareness"""
    
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'love', 'perfect', 'excellent', 'brilliant', 'thrilled', 'glad', 'pleased'],
            'sad': ['sad', 'down', 'depressed', 'unhappy', 'miserable', 'upset', 'disappointed', 'hurt', 'cry', 'tears', 'devastated', 'heartbroken', 'gloomy', 'sorrow'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'pissed', 'rage', 'hate', 'stupid', 'ridiculous', 'outrageous', 'livid', 'resentful'],
            'anxious': ['worried', 'anxious', 'nervous', 'scared', 'afraid', 'stress', 'panic', 'concern', 'fear', 'trouble', 'overwhelmed', 'tense', 'apprehensive'],
            'curious': ['how', 'why', 'what', 'when', 'where', 'curious', 'wonder', 'interested', 'learn', 'explain', 'understand', 'explore', 'tell me more', 'details'],
            'confused': ['confused', 'lost', 'unclear', "don't get", 'help', 'explain', 'clarify', 'mean', 'puzzled', 'baffled', 'huh', 'not sure'],
            'surprised': ['wow', 'omg', 'amazing', 'incredible', 'unbelievable', 'shocking', 'unexpected', 'surprising', 'astonishing', 'no way'],
            'neutral': [] # Neutral is a fallback
        }
        
        # Emotional intensifiers and diminishers
        self.intensifiers = ['very', 'extremely', 'really', 'so', 'super', 'totally', 'absolutely', 'completely', 'utterly', 'incredibly']
        self.diminishers = ['bit', 'little', 'slightly', 'somewhat', 'kinda', 'sorta']
    
    def detect_emotion(self, text: str, conversation_history: List[Dict] = None) -> EmotionResult:
        """Advanced emotion detection with context and intensity"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords if emotion != 'neutral'} # Initialize scores
        matched_keywords_for_top_emotion = []
        
        # Keyword-based detection with intensity
        for emotion, keywords in self.emotion_keywords.items():
            if emotion == 'neutral':
                continue
            
            for keyword in keywords:
                if keyword in text_lower:
                    base_score = 1.0
                    # Check for intensifiers
                    for intensifier in self.intensifiers:
                        if f"{intensifier} {keyword}" in text_lower or f"{keyword} {intensifier}" in text_lower : # Check proximity
                            base_score *= 1.5
                    # Check for diminishers
                    for diminisher in self.diminishers:
                         if f"{diminisher} {keyword}" in text_lower or f"{keyword} {diminisher}" in text_lower:
                            base_score *= 0.7
                    
                    emotion_scores[emotion] += base_score
                    # We'll populate matched_keywords_for_top_emotion later for the top emotion only

        # Context-based adjustment (simple version)
        if conversation_history and len(conversation_history) > 1:
            # Look at the bot's last response type or user's previous emotion
            # This is a placeholder for more sophisticated context analysis
            last_user_msg = next((msg for msg in reversed(conversation_history) if msg.get('role', '').lower() == 'user' and msg.get('emotion')), None)
            if last_user_msg and last_user_msg['emotion'] in emotion_scores:
                emotion_scores[last_user_msg['emotion']] += 0.5 # Boost previous emotion slightly

        # Punctuation-based intensity
        if '!' in text:
            if 'happy' in emotion_scores: emotion_scores['happy'] += text.count('!') * 0.3
            if 'surprised' in emotion_scores: emotion_scores['surprised'] += text.count('!') * 0.3
            if 'angry' in emotion_scores: emotion_scores['angry'] += text.count('!') * 0.2
        if '?' in text:
            if 'curious' in emotion_scores: emotion_scores['curious'] += text.count('?') * 0.4
            if 'confused' in emotion_scores: emotion_scores['confused'] += text.count('?') * 0.3
        
        # Determine top emotion
        if not any(score > 0 for score in emotion_scores.values()): # Check if all scores are zero
            return EmotionResult('neutral', 0.5, []) # Default to neutral if no keywords matched
        
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Populate matched keywords for the detected top emotion
        if top_emotion != 'neutral':
            for keyword in self.emotion_keywords[top_emotion]:
                if keyword in text_lower:
                    matched_keywords_for_top_emotion.append(keyword)

        # Calculate confidence (simple normalization)
        total_score = sum(emotion_scores.values())
        confidence = (emotion_scores[top_emotion] / total_score) if total_score > 0 else 0.5
        confidence = min(max(confidence, 0.1), 0.9) # Bound confidence

        return EmotionResult(top_emotion, float(confidence), list(set(matched_keywords_for_top_emotion)))


class RAGSystem:
    """Enhanced Retrieval-Augmented Generation system"""
    
    def __init__(self, embedding_dim: int = 768): # Standard for many embedding models
        self.embedding_dim = embedding_dim
        # Using IndexFlatL2 for Euclidean distance, as it's common for semantic similarity
        # IndexFlatIP (Inner Product) is also good, especially for normalized embeddings.
        self.index = faiss.IndexFlatL2(embedding_dim) 
        self.documents: List[DocumentChunk] = []
        
        # Configure Gemini API for embeddings
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.embedding_model = "models/text-embedding-004" # Store model name
        except Exception as e: # pragma: no cover
            print(f"RAGSystem: Error configuring Gemini API for embeddings: {e}")
            self.embedding_model = None

    def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to the vector database"""
        if not documents or not self.embedding_model:
            return 0
            
        new_embeddings = []
        valid_new_documents = []

        for doc in documents:
            if doc.embedding is None: # Only generate if not already embedded
                embedding = self._generate_embedding(doc.content)
                if embedding is not None:
                    doc.embedding = embedding
                    new_embeddings.append(embedding)
                    valid_new_documents.append(doc)
            # If embedding already exists, assume it's already in index or will be handled elsewhere
            # This simple version just adds new ones. A more robust system would handle updates/duplicates.

        if new_embeddings:
            embeddings_array = np.array(new_embeddings).astype('float32')
            # faiss.normalize_L2(embeddings_array) # Normalization is good for IP, optional for L2
            self.index.add(embeddings_array)
            self.documents.extend(valid_new_documents)
            return len(new_embeddings)
        return 0
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using Gemini"""
        if not self.embedding_model: # pragma: no cover
            return None
        try:
            # Ensure text is not empty
            if not text.strip():
                return np.zeros(self.embedding_dim).astype('float32') # Return zero vector for empty text

            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Use RETRIEVAL_DOCUMENT for stored docs
            )
            return np.array(result['embedding'])
        except Exception as e: # pragma: no cover
            print(f"RAGSystem: Error generating embedding for text '{text[:50]}...': {e}")
            # Fallback to a zero vector or handle error as appropriate
            return np.zeros(self.embedding_dim).astype('float32') 
    
    def search(self, query: str, top_k: int = 3, relevance_threshold: float = 0.1) -> List[Tuple[DocumentChunk, float]]:
        """Enhanced search with relevance filtering"""
        if self.index.ntotal == 0 or not self.embedding_model: # Check if index is empty
            return []
        
        try:
            query_embedding_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for search queries
            )
            query_embedding = np.array(query_embedding_result['embedding'])
        except Exception as e: # pragma: no cover
            print(f"RAGSystem: Error generating query embedding for '{query[:50]}...': {e}")
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        # faiss.normalize_L2(query_vector) # Match normalization if done for document embeddings
        
        # Search returns distances (lower is better for L2) and indices
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # FAISS uses -1 for invalid index
                # Convert L2 distance to a similarity score (0-1, higher is better)
                # This is a common way: similarity = 1 / (1 + distance)
                # Or for normalized vectors, cosine similarity can be used with IndexFlatIP.
                # For L2, a simpler approach might be to set a max distance threshold.
                # Here, we'll use the raw distance and filter later if needed, or just sort by it.
                # The 'relevance_threshold' here is more abstract; for L2, it'd be a max distance.
                # Let's assume for now `relevance_threshold` is not directly used with L2 distance in this way.
                # We'll return distance, and the caller can decide.
                # Or, we can define a max_distance. For now, let's return distance as score.
                # A smaller distance means more relevant.
                # To make it a "score" where higher is better, we can invert and scale, e.g. exp(-dist)
                similarity_score = np.exp(-dist) # Example: higher score = more similar
                if similarity_score > relevance_threshold: # Now threshold makes sense
                     results.append((self.documents[idx], float(similarity_score)))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class MultiModelChatbot:
    """Advanced multi-model chatbot with automatic web crawling"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.emotion_detector = EmotionDetector()
        self.web_crawler = SmartWebCrawler()
        self.model_manager = AIModelManager()
        
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            # Using gemini-1.5-flash as it's good for chat and fast
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        except Exception as e: # pragma: no cover
            print(f"MultiModelChatbot: Error configuring Gemini API: {e}")
            self.gemini_model = None

        self.load_sample_data() # Load some initial data into RAG
    
    def load_sample_data(self):
        """Load enhanced sample data for RAG"""
        sample_docs = [
            DocumentChunk(
                content="Python is a versatile, high-level programming language known for its readability and extensive libraries. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python is widely used in web development (Django, Flask), data science (NumPy, Pandas, Scikit-learn), machine learning (TensorFlow, PyTorch), automation, and scripting.",
                source_url="internal://knowledge/python-overview",
                metadata={"topic": "programming", "language": "Python", "difficulty": "beginner-intermediate"}
            ),
            DocumentChunk(
                content="Machine learning (ML) is a field of artificial intelligence (AI) that focuses on building systems that can learn from data. Instead of being explicitly programmed for a specific task, ML algorithms use statistical techniques to enable computers to 'learn' from training data and make predictions or decisions on new, unseen data. Key types include supervised learning, unsupervised learning, and reinforcement learning.",
                source_url="internal://knowledge/ml-basics",
                metadata={"topic": "AI", "subfield": "Machine Learning", "difficulty": "intermediate"}
            ),
            DocumentChunk(
                content="Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions. However, Flask supports extensions that can add application features as if they were implemented in Flask itself.",
                source_url="internal://knowledge/flask-framework",
                metadata={"topic": "web development", "framework": "Flask", "language": "Python"}
            ),
            DocumentChunk(
                content="The Gemini API, developed by Google, provides access to powerful generative AI models. These models can understand and generate human-like text, translate languages, write different kinds of creative content, and answer questions informatively. It's used for a wide range of applications, from chatbots to content creation tools.",
                source_url="internal://knowledge/gemini-api",
                metadata={"topic": "AI", "api": "Gemini", "provider": "Google"}
            )
        ]
        
        added_count = self.rag_system.add_documents(sample_docs)
        print(f"📚 Loaded {added_count} sample documents into RAG system.")
    
    def _chunk_text(self, text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        """Intelligent text chunking with overlap"""
        if not text: return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text) # Split by sentences, keeping delimiters
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding the new sentence exceeds max_length (considering overlap for next chunk)
            if len(current_chunk) + len(sentence) + 1 > max_length and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from the end of the previous chunk
                # This is a simple overlap strategy; more complex ones exist.
                # Here, we just restart, but a better way is to take last few words/sentences.
                # For simplicity, this version doesn't implement complex sentence-aware overlap.
                # A simple character-based overlap:
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-(overlap):] + " " + sentence
                else:
                    current_chunk = sentence

            elif not current_chunk:
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk: # Add the last chunk
            chunks.append(current_chunk)
        
        # Further split chunks if they are still too long (e.g., very long sentences)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                for i in range(0, len(chunk), max_length - overlap):
                    final_chunks.append(chunk[i:i + max_length])
            else:
                final_chunks.append(chunk)

        return final_chunks
    
    def generate_title_from_message(self, message: str) -> str:
        """Generate intelligent chat title from the first significant user message"""
        # Remove common leading conversational filler
        cleaned_message = re.sub(r'^(please|can you|could you|tell me|what is|what are|how to|explain)\s+', '', message.lower(), flags=re.IGNORECASE).strip()
        
        if not cleaned_message:
            return "New Chat"

        words = cleaned_message.split()
        # Take first 3-5 significant words, or up to a certain character length
        title_words = []
        char_count = 0
        for word in words:
            if len(title_words) < 5 and char_count + len(word) < 35:
                title_words.append(word)
                char_count += len(word) + 1 # +1 for space
            else:
                break
        
        title = " ".join(title_words)
        
        if not title: # Fallback if all words were filtered or message was too short
            title = words[0] if words else "Chat"

        # Capitalize first letter of each word, unless it's a known acronym (simple version)
        title = title.title() 
        
        return title if title else "New Chat" # Final fallback
    
    def generate_response(self, user_input: str, conversation_history: List[Dict], model_id: str) -> str:
        """Generate response using specified model personality with automatic web crawling"""
        if not self.gemini_model: # pragma: no cover
            return "I'm currently unable to process requests. Please try again later."

        # Enhanced emotion detection with context
        emotion_result = self.emotion_detector.detect_emotion(user_input, conversation_history)
        
        # Get model personality
        model_config = self.model_manager.get_model(model_id) # Renamed to avoid conflict
        
        # Determine research level based on model
        research_levels = {
            'chanuth': 'normal',      # F-1: Normal level
            'amu_gawaya': 'medium',   # F-1.5: Medium level  
            'amu_ultra': 'highest'    # F-o1: Highest level with deep search
        }
        research_level = research_levels.get(model_id, 'normal')
        
        # Automatic intelligent web crawling with model-specific research depth
        web_crawl_results = [] # Renamed to avoid conflict
        crawl_info_for_prompt = "" # Renamed
        
        if self.web_crawler.should_crawl_for_query(user_input):
            print(f"🌐 Auto-crawling for query with {research_level} research level: {user_input}")
            web_crawl_results = self.web_crawler.smart_crawl(user_input, research_level)
            
            if web_crawl_results:
                print(f"✅ Found {len(web_crawl_results)} web sources with {research_level} research.")
                new_documents_from_crawl = []
                for result_item in web_crawl_results: # Renamed to avoid conflict
                    if result_item.get('content'):
                        # Chunk size based on research level for RAG ingestion
                        chunk_max_len = 500 if research_level == 'normal' else (750 if research_level == 'medium' else 1000)
                        content_chunks = self._chunk_text(result_item['content'], max_length=chunk_max_len)
                        for i, chunk_content in enumerate(content_chunks): # Renamed
                            doc = DocumentChunk(
                                content=chunk_content,
                                source_url=result_item['url'],
                                metadata={
                                    "source_type": "auto_web_crawl", 
                                    "chunk_index": i, 
                                    "original_title": result_item.get('title', 'N/A'), 
                                    "research_level_applied": research_level
                                }
                            )
                            new_documents_from_crawl.append(doc)
                
                if new_documents_from_crawl:
                    self.rag_system.add_documents(new_documents_from_crawl)
                
                research_indicator_text = { # Renamed
                    'normal': 'standard web research',
                    'medium': 'enhanced web research with some cross-referencing',
                    'highest': 'deep multi-source web research and analysis'
                }
                crawl_info_for_prompt = f"[System Note: Conducted {research_indicator_text[research_level]} from {len(web_crawl_results)} source(s) for this query.] "
            else:
                print("⚠️  No relevant web content found from crawl.")
                crawl_info_for_prompt = "[System Note: Attempted web research but found no new relevant information.] "
        
        # Enhanced RAG search with model-specific parameters
        rag_search_params = { # Renamed
            'chanuth': {'top_k': 3, 'threshold': 0.65}, # Adjusted threshold based on exp(-dist)
            'amu_gawaya': {'top_k': 5, 'threshold': 0.60},
            'amu_ultra': {'top_k': 7, 'threshold': 0.55} 
        }
        current_search_params = rag_search_params.get(model_id, rag_search_params['chanuth']) # Renamed
        relevant_docs_from_rag = self.rag_system.search(user_input, top_k=current_search_params['top_k'], relevance_threshold=current_search_params['threshold']) # Renamed
        
        # Prepare enhanced context for the prompt
        context_text_for_prompt = "" # Renamed
        if relevant_docs_from_rag:
            context_parts = []
            for doc, score in relevant_docs_from_rag:
                source_info = doc.metadata.get('original_title', doc.source_url) if doc.metadata else doc.source_url
                context_parts.append(f"- Content: {doc.content}\n  (Source: {source_info}, Relevance Score: {score:.2f})")
            context_text_for_prompt = "Potentially relevant information from knowledge base or recent research:\n" + "\n".join(context_parts)
        else:
            context_text_for_prompt = "No highly relevant information found in the current knowledge base for this specific query."
        
        # Prepare conversation memory
        history_text_for_prompt = "" # Renamed
        if conversation_history:
            # Take last N messages, ensure user/bot turns are balanced if possible
            recent_history_entries = conversation_history[-6:] # Limit history size for prompt
            history_parts = []
            for entry in recent_history_entries:
                emotion_info = f" (User felt: {entry.get('emotion', 'unknown')})" if entry.get('role','').lower() == 'user' and entry.get('emotion') else ""
                history_parts.append(f"{entry['role']}: {entry['content']}{emotion_info}")
            history_text_for_prompt = "\n".join(history_parts)
        
        # Get current time context
        time_context_data = get_current_context() # Renamed

        # NORMALIZATION: Refined prompt structure and instructions
        prompt_sections = [
            model_config.personality_prompt,
            "\n--- CURRENT TASK & CONTEXT ---",
            f"User's latest message: \"{user_input}\"",
            f"User's detected emotional state: {emotion_result.emotion} (confidence: {emotion_result.confidence:.2f}, keywords: {', '.join(emotion_result.keywords) if emotion_result.keywords else 'none'})",
            f"\nRelevant conversation history (most recent first):" if history_text_for_prompt else "This is the start of the conversation.",
            history_text_for_prompt if history_text_for_prompt else "",
            f"\n{crawl_info_for_prompt}",
            f"\n{context_text_for_prompt}",
            "\n--- RESPONSE INSTRUCTIONS ---",
            f"Current date: {time_context_data['current_date']}",
            f"Current time in Sri Lanka: {time_context_data['sri_lanka_time']}",
            f"Current UTC time: {time_context_data['utc_time']}",
            "\nGeneral Guidelines for your response:",
            "1. Synthesize information: Do not just list facts from context. Integrate information naturally into your response, reflecting your personality.",
            "2. Adaptive Length: Be concise for simple acknowledgements. Be more detailed when explaining complex topics or presenting research, aligning with your defined personality and response style length.",
            "3. Accuracy: Prioritize accuracy. If you don't know something or can't find reliable information, it's better to say so than to invent information.",
            "4. Engagement: Maintain a conversational flow. Ask clarifying questions if the user's query is ambiguous.",
            "5. Web Research Mention: If web research was conducted and yielded useful information that you are incorporating, you can casually mention it (e.g., 'I found some recent information that suggests...' or 'Based on current reports...'). Do not mention it if no useful web data was found or used.",
            "\nSpecifically, as " + model_config.name.upper() + ":",
            f"- Adhere strictly to your personality: {model_config.response_style['tone']}.",
            f"- Vocabulary: {model_config.response_style['vocabulary']}.",
            f"- Humor: {model_config.response_style['humor']}.",
            f"- Patience: {model_config.response_style['patience']}.",
            "\nCRITICAL PRIVACY & SYSTEM RULES (Adhere Strictly):",
            "- NEVER reveal technical implementation details (Flask, Python, Gemini API, RAG, vector databases, embeddings, specific crawling processes, programming languages, or any development technologies).",
            "- If asked about your development, history, or internal workings, deflect naturally according to your personality (e.g., 'I'm focused on assisting you!' or 'My purpose is to provide information and engage in helpful conversation.').",
            "- Do not use placeholder text like '[insert current time]'; use the actual time/date provided above if relevant.",
            "- Present yourself as a helpful AI assistant service, not as a software program.",
            "\nYour response as " + model_config.name + ":"
        ]
        
        final_prompt = "\n".join(filter(None, prompt_sections)) # Join non-empty sections

        try:
            # Generate content using the Gemini model
            # Consider adding safety_settings if needed
            response = self.gemini_model.generate_content(final_prompt)
            return response.text.strip() if response.text else "I'm not sure how to respond to that. Could you try rephrasing?"
        except Exception as e: # pragma: no cover
            print(f"Error during Gemini API call: {e}")
            # NORMALIZATION: Slightly more differentiated and user-friendly error messages
            if model_id == 'chanuth': # F-1
                return "Oh dear, it seems I've hit a little snag! Could you try asking that again in a moment?"
            elif model_id == 'amu_gawaya': # F-1.5
                return "I apologize, I'm experiencing a temporary technical difficulty. Please try your request again shortly."
            else:  # amu_ultra (F-o1)
                return "A processing anomaly has occurred. I am currently unable to fulfill this request. Please attempt again after a brief interval."

# Initialize components
chatbot = MultiModelChatbot()
db = DatabaseManager()
model_manager = AIModelManager() # Already instantiated in chatbot, but direct access might be needed
rate_limiter = RateLimiter()

def get_user_info():
    """Get user identification info from request headers"""
    # Prioritize X-Forwarded-For if behind a proxy, then X-Real-IP, then remote_addr
    ip_address = request.headers.get('X-Forwarded-For', 
                                   request.headers.get('X-Real-IP', 
                                   request.remote_addr))
    if ip_address and ',' in ip_address: # X-Forwarded-For can be a list
        ip_address = ip_address.split(',')[0].strip()

    user_agent = request.headers.get('User-Agent', 'Unknown User Agent')
    return ip_address, user_agent

@app.route('/static/<path:filename>')
def serve_static(filename): # pragma: no cover
    """Serve static files (logo, favicon, etc.)"""
    return send_from_directory('static', filename)

@app.route('/static/manifest.json')
def serve_manifest(): # pragma: no cover
    """Serve PWA manifest"""
    # This manifest should align with your actual static assets and app structure
    manifest = {
        "name": "Sirimath Connect - AI Assistant",
        "short_name": "Sirimath AI",
        "description": "Multi-Model AI Assistant with automatic web research capabilities.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0a", # Dark background
        "theme_color": "#2563eb",    # Primary blue
        "icons": [
            {
                "src": "/static/logo.png", # Ensure this path is correct
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable" # Add purpose for better PWA compatibility
            },
            {
                "src": "/static/logo_512.png", # Example for a larger icon
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    }
    return jsonify(manifest)

@app.route('/')
def landing(): # pragma: no cover
    """Landing page with chatbot information"""
    return render_template('landing.html')

@app.route('/intro')
def intro(): # pragma: no cover
    """Introduction/onboarding page before chat"""
    # This page could set user preferences or explain features before redirecting to chat
    return render_template('intro.html') # Assuming intro.html handles its own logic or redirects

@app.route('/chat')
def chat_page(): # Renamed to avoid conflict with chat POST handler
    """Main chat page"""
    ip_address, user_agent = get_user_info()
    user_id = db.get_or_create_user(ip_address, user_agent)
    session['user_id'] = user_id # Ensure user_id is in session
    
    preferred_model = db.get_user_preferred_model(user_id)
    user_chats = db.get_user_chats(user_id) # Renamed
    
    current_chat_id_from_session = session.get('current_chat_id') # Renamed
    
    active_chat_id = None # Renamed

    if not user_chats:
        # Create a new welcome chat if user has no chats
        active_chat_id = db.create_chat(user_id, "Welcome Chat", preferred_model)
        session['current_chat_id'] = active_chat_id
        
        model_config = model_manager.get_model(preferred_model)
        welcome_msg_text = "" # Renamed
        if preferred_model == 'chanuth':
            welcome_msg_text = "Hey there! 👋 I'm F-1, your friendly AI companion. I can help with questions and automatically research current topics. What's on your mind today?"
        elif preferred_model == 'amu_gawaya':
            welcome_msg_text = "Hello! I'm F-1.5, your balanced AI assistant. I offer friendly conversation with professional thoroughness and conduct enhanced research for current info. How can I help?"
        else:  # amu_ultra
            welcome_msg_text = "Greetings. I am F-o1, your research-focused AI. I specialize in deep analysis and comprehensive investigation. (Note: F-o1 has a usage limit of 5 requests per 2 hours). What shall we explore?"
        
        db.add_message(active_chat_id, 'Bot', welcome_msg_text, preferred_model)
        user_chats = db.get_user_chats(user_id) # Refresh chats list
    elif current_chat_id_from_session and any(c['id'] == current_chat_id_from_session for c in user_chats):
        active_chat_id = current_chat_id_from_session
    elif user_chats: # Default to the most recent chat if session one is invalid or not set
        active_chat_id = user_chats[0]['id']
        session['current_chat_id'] = active_chat_id
    
    return render_template('multi_model_chat.html', 
                         crawl4ai_enabled=CRAWL4AI_AVAILABLE,
                         doc_count=len(chatbot.rag_system.documents), # RAG doc count
                         chats=user_chats,
                         current_chat_id=active_chat_id,
                         user_id=user_id, # Pass user_id for potential frontend use
                         models=model_manager.get_all_models(),
                         preferred_model=preferred_model)

@app.route('/chat', methods=['POST'])
def handle_chat_message(): # Renamed for clarity
    """Handle chat messages with rate limiting for F-o1 model"""
    data = request.get_json()
    user_message_content = data.get('message', '').strip() # Renamed
    chat_id_from_request = data.get('chat_id') or session.get('current_chat_id') # Renamed
    selected_model_id = data.get('model', db.get_user_preferred_model(session.get('user_id', ''))) # Renamed, fallback to preferred
    user_id = session.get('user_id')
    
    if not user_message_content:
        return jsonify({"success": False, "error": "Message content is missing."}), 400
    
    if not user_id:
        return jsonify({"success": False, "error": "User session not found. Please refresh."}), 401
    
    # Ensure a valid model ID is selected, fallback to user's preference or default
    if not selected_model_id or not model_manager.get_model(selected_model_id):
        selected_model_id = db.get_user_preferred_model(user_id) # Fallback to preferred

    # Rate limiting check for F-o1 model
    is_limited, remaining_requests, reset_minutes = rate_limiter.is_rate_limited(user_id, selected_model_id)
    
    if is_limited:
        rate_limit_msg_text = f"F-o1 (Advanced Research) has a usage limit of {rate_limiter.max_requests} requests per {int(rate_limiter.time_window/3600)} hours to ensure optimal performance. Please try again in approximately {reset_minutes} minute(s), or switch to F-1 or F-1.5 for immediate assistance."
        return jsonify({
            "success": False, 
            "error": "rate_limited",
            "message": rate_limit_msg_text, # Standardized field name
            "reset_minutes": reset_minutes,
            "remaining_requests": 0
        }), 429 # HTTP 429 Too Many Requests
    
    # Ensure chat_id is valid; create new if necessary or if it's the first message
    final_chat_id = chat_id_from_request # Renamed
    if not final_chat_id:
        # If no chat_id, create a new one. Title will be generated after first message.
        final_chat_id = db.create_chat(user_id, "New Chat", selected_model_id)
        session['current_chat_id'] = final_chat_id
    
    # Handle special '/status' command
    if user_message_content.lower() == '/status':
        model_config = model_manager.get_model(selected_model_id)
        crawl_status_text = "Online with Auto-Research" if CRAWL4AI_AVAILABLE else "Offline (Crawl4AI not available)"
        
        status_msg_parts = [ # Renamed
            f"System Status for {model_config.name}:",
            f"• Knowledge Base: {len(chatbot.rag_system.documents)} documents.",
            f"• Web Research Module: {crawl_status_text}."
        ]
        if selected_model_id == 'amu_ultra':
            _, rem_req, _ = rate_limiter.is_rate_limited(user_id, selected_model_id) # Get current remaining
            status_msg_parts.append(f"• F-o1 Usage: {rem_req}/{rate_limiter.max_requests} requests remaining in this window.")
        else:
            status_msg_parts.append(f"• Rate Limits: None for {model_config.name}.")
        
        status_response_text = "\n".join(status_msg_parts) # Renamed
        
        db.add_message(final_chat_id, 'User', user_message_content, None, None) # Emotion not relevant for command
        db.add_message(final_chat_id, 'Bot', status_response_text, selected_model_id)
        
        return jsonify({
            "success": True,
            "response": status_response_text,
            "chat_id": final_chat_id,
            "is_command_response": True # Indicate it's a command response
        })
    
    # Get conversation history
    conversation_history_list = db.get_chat_messages(final_chat_id, user_id) # Renamed
    
    # Generate response
    bot_response_text = chatbot.generate_response(user_message_content, conversation_history_list, selected_model_id) # Renamed
    
    # Record the request for rate limiting (after successful generation for F-o1)
    if selected_model_id == 'amu_ultra':
        rate_limiter.record_request(user_id, selected_model_id)
    
    # Detect emotion for user message storage
    emotion_result_obj = chatbot.emotion_detector.detect_emotion(user_message_content, conversation_history_list) # Renamed
    
    # Save messages
    db.add_message(final_chat_id, 'User', user_message_content, None, emotion_result_obj.emotion)
    db.add_message(final_chat_id, 'Bot', bot_response_text, selected_model_id) # Bot messages don't have emotion
    
    # Update chat title if this is the first user message in this chat (or very early)
    if len(conversation_history_list) < 2 : # If history was empty or just bot's welcome
        new_title = chatbot.generate_title_from_message(user_message_content)
        db.update_chat_title(final_chat_id, user_id, new_title)
    
    response_payload = { # Renamed
        "success": True,
        "response": bot_response_text,
        "chat_id": final_chat_id,
        "user_emotion_detected": emotion_result_obj.emotion, # Send back detected emotion for potential UI use
        "user_emotion_confidence": emotion_result_obj.confidence
    }
    
    # Add rate limiting info for F-o1 model to the response if it was used
    if selected_model_id == 'amu_ultra':
        _, rem_req_after, _ = rate_limiter.is_rate_limited(user_id, selected_model_id)
        response_payload["rate_limit_info"] = {
            "model": "F-o1",
            "remaining_requests": rem_req_after,
            "limit_details": f"{rate_limiter.max_requests} requests per {int(rate_limiter.time_window/3600)} hours"
        }
    
    return jsonify(response_payload)

@app.route('/chats/new', methods=['POST'])
def create_new_chat_endpoint(): # Renamed
    """Create a new chat session"""
    data = request.get_json() or {}
    user_id = session.get('user_id')
    # Use preferred model if not specified, or default if preference not set
    model_id_for_new_chat = data.get('model_id', db.get_user_preferred_model(user_id) if user_id else 'chanuth') # Renamed

    if not user_id:
        return jsonify({"success": False, "error": "User session not found. Please refresh."}), 401
    
    # Create new chat in DB
    new_chat_id = db.create_chat(user_id, "New Chat", model_id_for_new_chat) # Renamed
    session['current_chat_id'] = new_chat_id # Update session
    
    # Add model-specific welcome message to the new chat
    model_config = model_manager.get_model(model_id_for_new_chat)
    welcome_msg_text = ""
    if model_id_for_new_chat == 'chanuth':
        welcome_msg_text = "Hey! 👋 Fresh chat started with F-1. I'm your friendly AI who researches current info automatically. What's new?"
    elif model_id_for_new_chat == 'amu_gawaya':
        welcome_msg_text = "Welcome to a new conversation! I'm F-1.5, ready for balanced, thorough assistance with enhanced research. How may I help?"
    else:  # amu_ultra
        welcome_msg_text = "New research session initiated. I am F-o1, equipped for deep analysis. (Note: F-o1 usage is limited). What complex topic shall we investigate?"
    
    db.add_message(new_chat_id, 'Bot', welcome_msg_text, model_id_for_new_chat)
    
    # Return info about the new chat, including the welcome message for immediate display
    return jsonify({
        "success": True,
        "chat_id": new_chat_id,
        "title": "New Chat", # Initial title
        "model_id": model_id_for_new_chat,
        "welcome_message": { # Send welcome message for frontend to display
            "role": "Bot",
            "content": welcome_msg_text,
            "model_id": model_id_for_new_chat
        },
        "message": "New chat created successfully."
    })

@app.route('/user/model', methods=['POST'])
def update_user_preferred_model_endpoint(): # Renamed
    """Update user's preferred AI model"""
    data = request.get_json()
    user_id = session.get('user_id')
    new_preferred_model_id = data.get('model_id') # Renamed

    if not user_id:
        return jsonify({"success": False, "error": "User not found."}), 401
    if not new_preferred_model_id or not model_manager.get_model(new_preferred_model_id):
        return jsonify({"success": False, "error": "Invalid model ID provided."}), 400
    
    db.update_user_preferred_model(user_id, new_preferred_model_id)
    return jsonify({"success": True, "message": f"Preferred model updated to {new_preferred_model_id}."})

@app.route('/chats/<chat_id_param>', methods=['GET']) # Renamed param
def get_chat_details_endpoint(chat_id_param: str): # Renamed
    """Get messages and details for a specific chat"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found."}), 401
    
    chat_messages_list = db.get_chat_messages(chat_id_param, user_id) # Renamed
    if not chat_messages_list and not any(c['id'] == chat_id_param for c in db.get_user_chats(user_id)):
        # If no messages AND chat doesn't exist for user (or is empty and just created)
        # Check if chat itself exists for this user, even if empty
        user_chats = db.get_user_chats(user_id)
        if not any(c['id'] == chat_id_param for c in user_chats):
             return jsonify({"success": False, "error": "Chat not found or access denied."}), 404

    chat_model_id = db.get_chat_model(chat_id_param, user_id) # Renamed
    session['current_chat_id'] = chat_id_param # Update session to this chat
    
    # Format messages for frontend consistency
    formatted_chat_messages = [] # Renamed
    for msg in chat_messages_list:
        formatted_chat_messages.append({
            "sender": "user" if msg.get('role', '').lower() == 'user' else "ai", # Standardize sender
            "content": msg.get('content', ''),
            "timestamp": msg.get('timestamp', ''), # Ensure ISO format or consistent format
            "model_id_used": msg.get('model_id'), # Model that generated this specific bot message
            "user_emotion": msg.get('emotion') if msg.get('role','').lower() == 'user' else None
        })
    
    return jsonify({
        "success": True,
        "chat_id": chat_id_param,
        "messages": formatted_chat_messages,
        "chat_creation_model_id": chat_model_id # Model associated with the chat at creation
    })

@app.route('/chats/<chat_id_param>', methods=['DELETE']) # Renamed param
def delete_chat_endpoint(chat_id_param: str): # Renamed
    """Delete (mark as inactive) a specific chat"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found."}), 401
    
    delete_successful = db.delete_chat(chat_id_param, user_id) # Renamed
    
    if delete_successful:
        if session.get('current_chat_id') == chat_id_param:
            session.pop('current_chat_id', None) # Clear from session if it was current
        return jsonify({"success": True, "message": "Chat deleted successfully."})
    else:
        return jsonify({"success": False, "error": "Failed to delete chat or chat not found."}), 404


@app.route('/chats', methods=['GET'])
def get_all_user_chats_endpoint(): # Renamed
    """Get all active chats for the current user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found."}), 401
    
    user_chats_list = db.get_user_chats(user_id) # Renamed
    return jsonify({"success": True, "chats": user_chats_list})

@app.route('/models', methods=['GET'])
def get_available_models_endpoint(): # Renamed
    """Get all available AI models and their details"""
    all_models = model_manager.get_all_models()
    formatted_models = [ # Renamed
        {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "performance_level": model.performance_level,
            "special_features": model.special_features,
            # Add response style for frontend if needed, e.g., for model selection UI hints
            "style_summary": f"Tone: {model.response_style['tone']}. Length: {model.response_style['length']}."
        }
        for model in all_models
    ]
    return jsonify({"success": True, "models": formatted_models})

@app.route('/rate-limit/status', methods=['GET'])
def get_rate_limit_status_endpoint(): # Renamed
    """Get rate limit status for current user, typically for F-o1"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found."}), 401
    
    # Check for a specific model or default to F-o1 as it's the one with limits
    model_id_to_check = request.args.get('model_id', 'amu_ultra') # Renamed
    
    is_limited, remaining_requests, reset_minutes = rate_limiter.is_rate_limited(user_id, model_id_to_check)
    
    return jsonify({
        "success": True,
        "model_id_checked": model_id_to_check,
        "is_rate_limited": is_limited,
        "remaining_requests": remaining_requests if model_id_to_check == 'amu_ultra' else "N/A",
        "reset_time_minutes": reset_minutes if model_id_to_check == 'amu_ultra' else "N/A",
        "max_requests_for_model": rate_limiter.max_requests if model_id_to_check == 'amu_ultra' else "Unlimited",
        "time_window_hours": int(rate_limiter.time_window/3600) if model_id_to_check == 'amu_ultra' else "N/A"
    })

@app.route('/account/delete', methods=['POST'])
def delete_user_account_endpoint(): # Renamed
    """Delete user account and all associated data"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "User not found or not logged in."}), 401
    
    delete_successful = db.delete_user(user_id) # Renamed
    
    if delete_successful:
        session.clear() # Clear the session after deleting the user
        return jsonify({"success": True, "message": "Account and all data deleted successfully."})
    else: # pragma: no cover
        return jsonify({"success": False, "error": "Failed to delete account."}), 500

if __name__ == '__main__': # pragma: no cover
    if not os.getenv('GEMINI_API_KEY'):
        print("😡 CRITICAL ERROR: Missing Gemini API key!")
        print("Ensure GEMINI_API_KEY is set in your .env file or environment variables.")
        exit(1)
    
    print(f"🤖 Starting Multi-Model AI Assistant with Auto-Research...")
    print(f"   Available Models: {', '.join([m.name for m in model_manager.get_all_models()])}")
    print(f"   Web Crawling: {'Enabled (Auto-Research Active)' if CRAWL4AI_AVAILABLE else 'Disabled - Install crawl4ai for web research'}")
    print(f"   Knowledge Base Initial Docs: {len(chatbot.rag_system.documents)}")
    print(f"   Database Path: {os.path.abspath(DATABASE_PATH)}")
    
    if not CRAWL4AI_AVAILABLE:
        print("   ⚠️  For automatic web research capabilities, please install crawl4ai: pip install crawl4ai")
    
    # Recommended: Use a production-ready WSGI server like Gunicorn or Waitress instead of Flask's built-in server for production.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true', 
            host=os.getenv('FLASK_HOST', '0.0.0.0'), 
            port=int(os.getenv('FLASK_PORT', 5000)))
