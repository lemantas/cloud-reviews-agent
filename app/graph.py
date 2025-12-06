from typing import TypedDict, Annotated, Sequence
import json
import sqlite3

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from clients import get_llm
from prompts import get_agent_prompt
from tools import summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews


# Custom reducer for accumulating list items
def add_list_items(left: list, right: list) -> list:
    """Accumulate list items by concatenating."""
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = []
    return left + right


class AgentState(TypedDict):
    """State for the agentic RAG workflow.

    Attributes:
        messages: Conversation history (auto-appended via add_messages reducer)
        tool_outputs: Accumulated tool outputs for UI rendering
        snippets: Accumulated review snippets from retrieve_reviews calls
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_outputs: Annotated[list, add_list_items]
    snippets: Annotated[list, add_list_items]


def agent_node(state: AgentState) -> dict:
    """Agent reasoning node - calls LLM with tools bound.

    This node:
    1. Retrieves current messages from state
    2. Adds system prompt if first message
    3. Binds tools to LLM
    4. Invokes LLM to decide next action
    5. Returns updated messages

    Args:
        state: Current agent state with messages history

    Returns:
        Partial state update with new AI message
    """
    try:
        llm = get_llm()
        tools = [summarize_sentiment, extract_top_aspects, infer_jtbd, retrieve_reviews]
        model = llm.bind_tools(tools)

        messages = list(state["messages"])

        # Add system prompt if this is the first turn (no SystemMessage exists)
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_text = get_agent_prompt()
            messages = [SystemMessage(content=system_text)] + messages

        # Invoke LLM with tools
        response = model.invoke(messages)

        return {"messages": [response]}

    except Exception as e:
        # Return error message instead of crashing
        error_msg = f"Error in agent reasoning: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


def tools_node(state: AgentState) -> dict:
    """Execute requested tools and track outputs.

    This node:
    1. Extracts tool calls from last AI message
    2. Executes each tool with provided arguments
    3. Creates ToolMessage objects for LLM
    4. Tracks tool outputs and snippets separately for UI

    Args:
        state: Current agent state with messages history

    Returns:
        Partial state update with tool messages, outputs, and snippets
    """
    try:
        last_message = state["messages"][-1]

        # Extract tool calls (modern LangChain format)
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        tools_dict = {
            "sentiment_analysis": summarize_sentiment,
            "aspect_extraction": extract_top_aspects,
            "jtbd_analysis": infer_jtbd,
            "retrieve_reviews": retrieve_reviews
        }

        tool_messages = []
        new_tool_outputs = []
        new_snippets = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_func = tools_dict.get(tool_name)

            if not tool_func:
                # Unknown tool - return error
                tool_messages.append(ToolMessage(
                    tool_call_id=call["id"],
                    content=f"Error: Unknown tool '{tool_name}'"
                ))
                continue

            try:
                result = tool_func.invoke(call["args"])

                # Convert result to string for ToolMessage
                if isinstance(result, dict):
                    content_str = json.dumps(result)
                else:
                    content_str = str(result)

                tool_messages.append(ToolMessage(
                    tool_call_id=call["id"],
                    content=content_str,
                    name=tool_name
                ))

                # Track outputs for UI
                if tool_name == "retrieve_reviews" and isinstance(result, dict):
                    if "snippets" in result:
                        new_snippets.extend(result["snippets"])
                elif tool_name in ["sentiment_analysis", "aspect_extraction", "jtbd_analysis"]:
                    new_tool_outputs.append({"name": tool_name, "output": result})

            except Exception as e:
                # Tool execution error
                tool_messages.append(ToolMessage(
                    tool_call_id=call["id"],
                    content=f"Error executing {tool_name}: {str(e)}",
                    name=tool_name
                ))

        return {
            "messages": tool_messages,
            "tool_outputs": new_tool_outputs,
            "snippets": new_snippets
        }

    except Exception as e:
        # Return error instead of crashing
        error_msg = f"Error in tool execution: {str(e)}"
        return {"messages": [ToolMessage(content=error_msg, tool_call_id="error")]}


def should_continue(state: AgentState) -> str:
    """Decide whether to continue with tools or end.

    This routing function checks if the last AI message contains tool calls.
    If yes, route to tools node. If no, end the conversation.

    Args:
        state: Current agent state with messages history

    Returns:
        "continue" to execute tools, "end" to finish
    """
    last_message = state["messages"][-1]

    # Check if LLM made tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # No tool calls - end conversation
    return "end"


def create_agent_graph() -> CompiledStateGraph:
    """Create and compile the agent workflow graph.

    This function builds the LangGraph workflow with:
    - Entry point at agent node
    - Conditional routing from agent (continue to tools or end)
    - Loop from tools back to agent for multi-turn reasoning
    - SQLite checkpointing for conversation persistence

    Returns:
        Compiled graph with checkpointing enabled
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )

    # Add edge from tools back to agent (for multi-step reasoning)
    workflow.add_edge("tools", "agent")

    # Compile with SQLite checkpointing for persistence
    # Create persistent connection for checkpointing (thread-safe for reuse)
    conn = sqlite3.connect("data/agent_checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


# Singleton pattern for graph reuse
_agent_graph = None


def get_agent_graph() -> CompiledStateGraph:
    """Get or create the compiled agent graph.

    Uses singleton pattern to avoid recompiling the graph on every request.

    Returns:
        Compiled LangGraph workflow
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph
