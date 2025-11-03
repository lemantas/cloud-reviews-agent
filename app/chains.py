from retrieval import retrieve_documents, format_snippets_to_text
from prompts import get_rag_prompt, get_agent_prompt, get_tool_selection_prompt
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

def agentic_response(question, chunk_type="sentence", vendor=None, top_k=None, fetch_k=None):
    """Agentic RAG with LLM-driven tool selection for intelligent analysis."""
    try:
        # Step 1: Retrieve documents
        snippets = retrieve_documents(question, chunk_type, vendor, top_k, fetch_k)

        if not snippets:
            return {
                "response": "No relevant reviews found for your question. Try adjusting your search terms or filters.",
                "tool_outputs": {},
                "snippets": []
            }

        # Step 2: Prepare tools for LLM
        tools = [summarize_sentiment, extract_top_aspects, infer_jtbd]
        tools_data = json.dumps(snippets)

        # Step 3: Let LLM decide which tools to use
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools)

        # Build tool selection messages via prompts
        tool_select_prompt = get_tool_selection_prompt()
        messages = tool_select_prompt.format_messages(question=question)

        # Invoke LLM with tools
        ai_msg = llm_with_tools.invoke(messages)

        # Step 4: Execute selected tools
        tool_outputs = {}

        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]

                # Execute the selected tool
                if tool_name == "summarize_sentiment":
                    result = summarize_sentiment.invoke({"snippets_data": tools_data, "question": question})
                    tool_outputs["sentiment_analysis"] = result
                elif tool_name == "extract_top_aspects":
                    result = extract_top_aspects.invoke({"snippets_data": tools_data, "question": question})
                    tool_outputs["aspect_extraction"] = result
                elif tool_name == "infer_jtbd":
                    result = infer_jtbd.invoke({"snippets_data": tools_data, "question": question})
                    tool_outputs["jtbd_analysis"] = result

        # Step 5: Create enriched context
        context_snippets = format_snippets_to_text(snippets)

        # Add tool analysis to context
        tool_context = ""
        if tool_outputs:
            tool_context = "\n\n## Analysis Results:\n"
            for tool_name, output in tool_outputs.items():
                tool_context += f"\n### {tool_name.replace('_', ' ').title()}:\n{json.dumps(output, indent=2)}\n"

        full_context = context_snippets + tool_context

        # Step 6: Generate final response with enriched context
        prompt = get_agent_prompt()
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