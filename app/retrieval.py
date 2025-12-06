from typing import Optional, Dict, List, Any
from langchain_core.retrievers import BaseRetriever
from clients import get_vector_store


TOP_K = 12  # number of results to return
FETCH_K = 30  # Consider FETCH_K results, then diversify to TOP_K
LAMBDA_MMR = 0.5  # 0 - 1 (0 = most diverse, 1 = most relevant)


def create_retriever(
    query: str,
    chunk_type: str = "sentence",
    vendor: Optional[str] = None,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None
) -> BaseRetriever:
    """Dynamically create retrieval query.

    Args:
        query: Search query for vector similarity
        chunk_type: Type of chunks to retrieve ("sentence" or "review")
        vendor: Optional vendor filter
        top_k: Number of results to return (defaults to TOP_K)
        fetch_k: Number of candidates for MMR (defaults to FETCH_K)

    Returns:
        Configured retriever with MMR search and metadata filtering
    """
    vector_store = get_vector_store()
    filters = {"chunk_type": chunk_type}
    if vendor:
        filters["vendor"] = vendor
    else:
        filters.pop("vendor", None)

    if len(filters) > 1:
        filter_query = {"$and": [{k: v} for k, v in filters.items()]}
    else:
        filter_query = filters

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k or TOP_K,
            "fetch_k": fetch_k or FETCH_K,
            "lambda_mult": LAMBDA_MMR,
            "filter": filter_query
        }
    )

def retrieve_documents(
    question: str,
    chunk_type: str = "sentence",
    vendor: Optional[str] = None,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K
) -> List[Dict[str, Any]]:
    """Retrieve documents as structured objects.

    Args:
        question: Search query for retrieval
        chunk_type: Type of chunks to retrieve ("sentence" or "review")
        vendor: Optional vendor filter
        top_k: Number of results to return (defaults to TOP_K constant)
        fetch_k: Number of candidates for MMR (defaults to FETCH_K constant)

    Returns:
        List of snippet dictionaries with keys: text, rating, date, source, vendor, review_header
    """
    retriever = create_retriever(question, chunk_type, vendor, top_k, fetch_k)
    docs = retriever.invoke(question)
    if not docs:
        return []

    # Deduplicate by review_id to avoid multiple chunks from the same review
    seen_review_ids = set()
    unique_docs = []
    for doc in docs:
        review_id = doc.metadata.get('review_id')
        if review_id not in seen_review_ids:
            seen_review_ids.add(review_id)
            unique_docs.append(doc)

    processed_docs = []
    for doc in unique_docs:
        metadata = doc.metadata
        processed_doc = {
            "text": doc.page_content,
            "rating": metadata.get("score", 0),
            "date": metadata.get("date", ""),
            "source": metadata.get("name", "Anonymous"),
            "vendor": metadata.get("vendor", ""),
            "review_header": metadata.get("review_header", "")
        }
        processed_docs.append(processed_doc)

    return processed_docs

def format_snippets_to_text(snippets: List[Dict[str, Any]]) -> str:
    """Format snippet objects to text for a simple RAG workflow.

    Args:
        snippets: List of snippet dictionaries

    Returns:
        Formatted string with all snippets, or default message if empty
    """
    if not snippets:
        return "No relevant reviews found."

    formatted = []
    for snippet in snippets:
        source_line = f"{snippet['source']} | {snippet['date'] or 'Unknown date'} | Score: {snippet['rating'] or 'N/A'}"
        formatted.append(f"[{source_line}] {snippet['text'].strip()}")

    return "\n\n".join(formatted)