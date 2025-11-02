from retrieval import retrieve_documents, format_snippets_to_text
from prompts import get_rag_prompt, get_agent_prompt
from clients import get_llm, get_vector_store
from tools import summarize_sentiment, extract_top_aspects, infer_jtbd
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

def route_query_to_tools(question):
    """Route user query to appropriate analysis tools based on keywords."""
    question_lower = question.lower()

    # Keyword-based routing
    tools_to_use = []

    # Sentiment-related keywords
    sentiment_keywords = ["sentiment", "feel", "opinion", "positive", "negative", "happy", "unhappy", "satisfied", "disappointed"]
    if any(keyword in question_lower for keyword in sentiment_keywords):
        tools_to_use.append("sentiment")

    # Aspect-related keywords
    aspect_keywords = ["feature", "theme", "aspect", "topic", "discuss", "mention", "talk about", "performance", "pricing", "support", "reliability"]
    if any(keyword in question_lower for keyword in aspect_keywords):
        tools_to_use.append("aspects")

    # JTBD-related keywords
    jtbd_keywords = ["job", "accomplish", "goal", "task", "trying to", "need to", "want to", "use for", "purpose", "choose"]
    if any(keyword in question_lower for keyword in jtbd_keywords):
        tools_to_use.append("jtbd")

    # If no specific keywords found, use all tools for comprehensive analysis
    if not tools_to_use:
        tools_to_use = ["sentiment", "aspects", "jtbd"]

    return tools_to_use

def agentic_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Simple RAG with smart tool routing based on query analysis."""
    try:
        # Step 1: Retrieve documents
        snippets = retrieve_documents(question, chunk_type, vendor, top_k, fetch_k)

        if not snippets:
            return {
                "response": "No relevant reviews found for your question. Try adjusting your search terms or filters.",
                "tool_outputs": {},
                "snippets": []
            }

        # Step 2: Route query to appropriate tools
        tools_to_use = route_query_to_tools(question)

        # Step 3: Run selected tools
        tools_data = json.dumps(snippets)
        tool_outputs = {}

        if "sentiment" in tools_to_use:
            tool_outputs["sentiment_analysis"] = summarize_sentiment.invoke({"snippets_data": tools_data, "question": question})

        if "aspects" in tools_to_use:
            tool_outputs["aspect_extraction"] = extract_top_aspects.invoke({"snippets_data": tools_data, "question": question})

        if "jtbd" in tools_to_use:
            tool_outputs["jtbd_analysis"] = infer_jtbd.invoke({"snippets_data": tools_data, "question": question})

        # Step 4: Create enriched context
        context_snippets = format_snippets_to_text(snippets)

        # Add tool analysis to context
        tool_context = ""
        if tool_outputs:
            tool_context = "\n\n## Analysis Results:\n"
            for tool_name, output in tool_outputs.items():
                tool_context += f"\n### {tool_name.replace('_', ' ').title()}:\n{output}\n"

        full_context = context_snippets + tool_context

        # Step 5: Generate response with enriched context
        prompt = get_agent_prompt()
        llm = get_llm()

        formatted_prompt = prompt.format(context=full_context, question=question)
        response = llm.invoke(formatted_prompt)

        return {
            "response": response.content,
            "tool_outputs": tool_outputs,
            "snippets": snippets
        }

    except Exception as e:
        return {
            "response": f"Error processing your question: {str(e)}",
            "tool_outputs": {},
            "snippets": []
        }