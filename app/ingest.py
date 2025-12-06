from pathlib import Path
import sqlite3
import pandas as pd
import chromadb
import nltk
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWS_DIR = DATA_DIR / "reviews"
SQLITE_PATH = DATA_DIR / "sqlite.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# 1) Load reviews
def load_reviews(folder):
    """Read all CSVs from folder and add a 'vendor' column from filename."""
    files = []
    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)
        df["vendor"] = file.stem
        files.append(df)
    return pd.concat(files, ignore_index=True).fillna("")

df = load_reviews(REVIEWS_DIR)

# 2) Build SQLite schema
with sqlite3.connect(SQLITE_PATH) as con:
    con.execute("DROP TABLE IF EXISTS reviews;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
          id INTEGER PRIMARY KEY,
          name TEXT,
          country TEXT,
          date TEXT,
          review_score INTEGER,
          review_header TEXT,
          review_body TEXT,
          vendor TEXT
        );
        """
    )
    df.to_sql("reviews", con, if_exists="replace", index=False)

# 3) Build LangChain documents with hybrid chunking
def create_hybrid_documents(df):
    """Create both full review documents and sentence-level chunks."""
    all_docs = []

    for idx, row in df.iterrows():
        review_id = f"{row.get('vendor')}_{idx}"
        header = row.get('review_header', '').strip()
        body = row.get('review_body', '').strip()

        # Base metadata for all chunks
        base_meta = {
            "name": row.get("name"),
            "country": row.get("country"),
            "date": row.get("date"),
            "score": row.get("review_score"),
            "vendor": row.get("vendor"),
            "review_id": review_id,
        }

        # 1) Full review document
        full_content = f"{header}\n\n{body}" if body else header
        full_doc = Document(
            page_content=full_content,
            metadata={**base_meta, "chunk_type": "review", "chunk_level": "full"}
        )
        all_docs.append(full_doc)

        # 2) Sentence-level documents from body (if exists and non-trivial)
        if body and len(body.split()) > 5:  # Only chunk non-trivial bodies
            try:
                sentences = nltk.sent_tokenize(body)
                for i, sentence in enumerate(sentences):
                    if len(sentence.strip()) > 3:  # Skip very short sentences
                        sentence_doc = Document(
                            page_content=sentence.strip(),
                            metadata={
                                **base_meta,
                                "chunk_type": "sentence",
                                "chunk_level": "sentence",
                                "parent_id": review_id,
                                "sentence_idx": i,
                                "review_header": header
                            }
                        )
                        all_docs.append(sentence_doc)
            except Exception as e:
                print(f"Error tokenizing review {review_id}: {e}")
                # Fallback: treat as single chunk
                continue

    return all_docs

docs = create_hybrid_documents(df)

# 4) Load to ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
# Reset collection
try:
    client.delete_collection("reviews")
except Exception:
    pass

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    client=client,
    collection_name="reviews",
    embedding_function=embeddings,
)

BATCH = 1000
# Count chunk types efficiently
review_count = sum(1 for d in docs if d.metadata.get('chunk_type') == 'review')
sentence_count = len(docs) - review_count
print(f"Ingesting {len(docs)} documents ({review_count} reviews + {sentence_count} sentences)")

for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    vectordb.add_documents(batch)
    print(f"Processed batch {start//BATCH + 1}/{(len(docs) + BATCH - 1)//BATCH}")

print("Ingestion complete!")