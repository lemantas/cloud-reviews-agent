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

def simple_rag_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Simple RAG response without tools for basic Q&A."""
    try:
        response, snippets = rag_chain(question, chunk_type, vendor, top_k, fetch_k)

        return {
            "response": response,
            "tool_outputs": {},
            "snippets": snippets
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": {},
            "snippets": []
        }

def agentic_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Agentic RAG with LLM-driven tool selection for intelligent analysis."""
    try:
        llm = get_llm()
        tools = [summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews]
        system_text = get_agent_prompt()
        agent = create_agent(model=llm, tools=tools, system_prompt=system_text)

        # Invoke LLM with messages per new Agents API
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
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
        tool_outputs = {}
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
                    tool_outputs[name] = parsed


        return {
            "response": final_text,
            "tool_outputs": tool_outputs,
            "snippets": snippets
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": {},
            "snippets": []
        }