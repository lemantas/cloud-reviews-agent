import chromadb
import sqlite3
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Configuration
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
CHROMA_COLLECTION = "reviews"
SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "sqlite.db"

# Global clients (lazy loading)
_client = None
_embeddings = None
_vector_store = None
_llm = None
_callbacks = []

def set_callbacks(callbacks):
    """Set callbacks for LLM token tracking."""
    global _callbacks
    _callbacks = callbacks

def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            max_retries=3
        )
    return _embeddings

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            client=get_chroma_client(),
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
        )
    return _vector_store

def get_llm():
    global _llm, _callbacks
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            max_retries=3,
            temperature=0.2,
            streaming=True,
            callbacks=_callbacks
        )
    return _llm

def get_review_stats():
    """Query SQLite database to get review counts grouped by vendor."""
    try:
        if not SQLITE_PATH.exists():
            return {}

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT vendor, COUNT(*) as count
                FROM reviews
                GROUP BY vendor
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            return {vendor: count for vendor, count in results}
    except Exception as e:
        print(f"Error fetching review stats: {e}")
        return {}