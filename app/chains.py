from retrieval import retrieve_documents, format_snippets_to_text
from prompts import get_rag_prompt, get_agent_prompt 
from clients import get_llm
from tools import summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews
from langchain.agents import create_agent
import json

def rag_chain(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Simple RAG chain for question answering with citations."""
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
    """Simple RAG response without tools for basic Q&A.

    Args:
        question: Current user question
        chunk_type: Type of chunks to retrieve (sentence/review)
        vendor: Optional vendor filter
        top_k: Number of results to return
        fetch_k: Number of candidates for MMR
        conversation_history: List of previous messages for context
    """
    try:
        # For simple RAG, we don't use full conversation history in LLM call
        # but we could optionally append it to context in future
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
    """Agentic RAG with LLM-driven tool selection for intelligent analysis.

    Args:
        question: Current user question
        chunk_type: Type of chunks to retrieve (sentence/review)
        vendor: Optional vendor filter
        top_k: Number of results to return
        fetch_k: Number of candidates for MMR
        conversation_history: List of previous messages for context
    """
    try:
        llm = get_llm()
        tools = [summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews]
        system_text = get_agent_prompt()
        agent = create_agent(model=llm, tools=tools, system_prompt=system_text)

        # Build full message history
        messages = []

        # Add conversation history if provided (excluding tool outputs and metadata)
        if conversation_history:
            for msg in conversation_history:
                # Only include user and assistant messages (not tool outputs)
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Add current user question
        messages.append({"role": "user", "content": question})

        # Invoke LLM with full conversation context
        result = agent.invoke({
            "messages": messages
        })

        messages = result.get("messages", [])

        # Final text: content of the last AI message
        final_text = ""
        for message in reversed(messages):
            role = getattr(message, "role", None) or getattr(message, "type", None)
            if role in ("ai", "assistant"):
                final_text = getattr(message, "content", "")
                break

        # Collect simple tool outputs from ToolMessage entries
        # Use list instead of dict to preserve multiple calls to same tool
        tool_outputs = []
        snippets = []

        for m in messages:
            mtype = getattr(m, "type", None) or getattr(m, "role", None)
            if mtype == "tool":
                name = getattr(m, "name", None) or getattr(m, "tool_name", None) or "tool"
                content = getattr(m, "content", None)

                # Normalize string content to dict when possible
                parsed = content
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        parsed = content

                # Special-case: capture snippets from retrieve_reviews
                if name == "retrieve_reviews" and isinstance(parsed, dict) and parsed.get("snippets"):
                    # Accumulate snippets from multiple retrieve_reviews calls
                    if isinstance(parsed["snippets"], list):
                        snippets.extend(parsed["snippets"])
                    # Don't add retrieve_reviews to tool_outputs - snippets go to separate field
                    continue

                # Normalize known analysis tool outputs to dict for UI formatters
                if name in ("sentiment_analysis", "aspect_extraction", "jtbd_analysis") and isinstance(parsed, dict):
                    # Add as list entry with name and output
                    tool_outputs.append({"name": name, "output": parsed})

        return {
            "response": final_text,
            "tool_outputs": tool_outputs,
            "snippets": snippets
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": [],
            "snippets": []
        }