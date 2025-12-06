from clients import get_vector_store


TOP_K = 12 # number of results to return
FETCH_K = 30 # sconsider FETCH_K results, then diversify to TOP_K
LAMBDA_MMR = 0.5 # 0 - 1 (0 = most diverse, 1 = most relevant)


def create_retriever(query, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Dynamically create retrieval query."""
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

def retrieve_documents(question, chunk_type="sentence", vendor=None, top_k=TOP_K, fetch_k=FETCH_K):
    """Retrieve documents as structured objects."""
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

    # Process metadata consistently
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

def format_snippets_to_text(snippets):
    """Format snippet objects to text for a simple RAG workflow."""
    if not snippets:
        return "No relevant reviews found."

    formatted = []
    for snippet in snippets:
        source_line = f"{snippet['source']} | {snippet['date'] or 'Unknown date'} | Score: {snippet['rating'] or 'N/A'}"
        formatted.append(f"[{source_line}] {snippet['text'].strip()}")

    return "\n\n".join(formatted)