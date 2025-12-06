from retrieval import retrieve_documents, format_snippets_to_text
from prompts import get_rag_prompt, get_agent_prompt
from clients import get_llm
from langchain_core.messages import HumanMessage, ToolMessage
from graph import get_agent_graph
import uuid

def rag_chain(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Simple RAG chain to answer questions using retrieved reviews from vector database."""
    prompt = get_rag_prompt()
    llm = get_llm()

    try:
        # Use same retrieval path as other functions
        snippets = retrieve_documents(question, chunk_type, vendor, top_k, fetch_k)
        context = format_snippets_to_text(snippets)

        # Format prompt and invoke LLM
        formatted_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        return response.content, snippets
    except Exception as e:
        return f"Error processing question: {str(e)}", []

def simple_rag_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None, conversation_history=None):
    try:
        # For simple RAG, we don't use full conversation history in LLM call
        response, snippets = rag_chain(question, chunk_type, vendor, top_k, fetch_k)

        return {
            "response": response,
            "tool_outputs": [],
            "snippets": snippets
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": [],
            "snippets": []
        }

def agentic_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None, conversation_history=None):
    """Agentic RAG with LangGraph for intelligent analysis.

    Args:
        question: Current user question
        chunk_type: Type of chunks to retrieve (sentence/review)
        vendor: Optional vendor filter
        top_k: Number of results to return
        fetch_k: Number of candidates for MMR
        conversation_history: DEPRECATED - handled via LangGraph checkpointing
    """
    try:
        graph = get_agent_graph()

        # Get thread_id from Streamlit session (will be set in app.py)
        # This enables conversation persistence across queries
        try:
            import streamlit as st
            if "thread_id" not in st.session_state:
                st.session_state.thread_id = str(uuid.uuid4())
            thread_id = st.session_state.thread_id
        except Exception:
            # Fallback for non-Streamlit usage (e.g., testing)
            thread_id = "default"

        # Create config with thread_id for checkpointing
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke graph with new user message
        # LangGraph automatically handles conversation history via checkpointing
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "tool_outputs": [],
                "snippets": []
            },
            config=config
        )

        # Extract final response from last non-tool message
        final_text = ""
        for message in reversed(result["messages"]):
            # Skip tool messages, get last AI message
            if not isinstance(message, ToolMessage) and hasattr(message, "content"):
                final_text = message.content
                break

        return {
            "response": final_text,
            "tool_outputs": result.get("tool_outputs", []),
            "snippets": result.get("snippets", [])
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": [],
            "snippets": []
        }