import streamlit as st
import pandas as pd
from datetime import datetime
from chains import agentic_response, simple_rag_response
from clients import get_vector_store, set_callbacks, get_review_stats
from token_tracker import TokenTracker
import uuid


def format_sentiment_analysis(data):
    """Format sentiment analysis JSON data for display."""
    if "error" in data:
        return f"❌ {data['error']}"

    result = "**Sentiment Analysis Summary**\n\n"
    result += f"• **Total Reviews:** {data['total_reviews']}\n"

    if data['mean_rating']:
        result += f"• **Average Rating:** {data['mean_rating']}/5\n"
        result += f"• **Positive Reviews:** {data['positive_share']}% (rating ≥4)\n"
        result += f"• **Negative Reviews:** {data['negative_share']}% (rating ≤2)\n"

    if data['positive_themes']:
        result += f"\n**Top Positive Themes:**\n"
        for theme in data['positive_themes']:
            result += f"  - {theme}\n"

    if data['negative_themes']:
        result += f"\n**Top Negative Themes:**\n"
        for theme in data['negative_themes']:
            result += f"  - {theme}\n"

    return result

def format_aspect_extraction(data):
    """Format aspect extraction JSON data for display."""
    if "error" in data:
        return f"❌ {data['error']}"

    result = f"**Top Discussed Aspects** ({data['total_aspects']} found)\n\n"

    for i, aspect in enumerate(data['aspects'], 1):
        sentiment_info = f" • Avg Rating: {aspect['sentiment_score']}/5" if aspect['sentiment_score'] else ""
        result += f"**{i}. {aspect['name'].title()}** ({aspect['frequency']} mentions{sentiment_info})\n"

        if aspect['positive_examples']:
            result += f"   *Positive:* {aspect['positive_examples'][0]}\n"

        if aspect['neutral_examples']:
            result += f"   *Neutral:* {aspect['neutral_examples'][0]}\n"

        if aspect['negative_examples']:
            result += f"   *Negative:* {aspect['negative_examples'][0]}\n"

        result += "\n"

    return result

def format_jtbd_analysis(data):
    """Format JTBD analysis JSON data for display."""
    if "error" in data:
        return f"❌ {data['error']}"

    result = f"**Jobs-to-Be-Done Analysis**\n"
    result += f"*Based on {data['total_reviews']} reviews*\n\n"

    result += f"**Job:** {data['job']}\n\n"
    result += f"**Situation:** {data['situation']}\n\n"
    result += f"**Motivation:** {data['motivation']}\n\n"
    result += f"**Expected Outcome:** {data['expected_outcome']}\n\n"

    if data['frustrations']:
        result += f"**Common Frustrations:**\n"
        for frustration in data['frustrations']:
            result += f"• {frustration}\n"
        result += "\n"

    if data['quotes']:
        result += f"**Supporting Quotes:**\n"
        for quote in data['quotes']:
            result += f"• \"{quote}\"\n"

    return result

# Page configuration here
st.set_page_config(
    page_title="Cloud Vendor Analysis RAG",
    page_icon="⭐",
    layout="wide"
)

# Initialize token tracker
tracker = TokenTracker(max_tokens=100000)
set_callbacks([tracker])

# Initialize conversation history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize thread_id for LangGraph checkpointing
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# LAYOUT GOES NEXT

# Title and header
st.title("⭐ Cloud Vendor Analysis RAG")
st.markdown("*Analyze customer reviews of selected cloud vendors*")

# Sidebar filters
st.sidebar.write(f"**Tokens:** {tracker.used:,} / {tracker.max_tokens:,}")
st.sidebar.progress(tracker.used / tracker.max_tokens if tracker.max_tokens > 0 else 0)
if tracker.used / tracker.max_tokens > 0.9:
    st.sidebar.error("⚠️ Token limit almost reached!")

# Clear conversation button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    # Reset thread_id to start a new conversation in LangGraph
    st.session_state.thread_id = str(uuid.uuid4())
    tracker.reset()
    st.rerun()

st.sidebar.markdown("---")

analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["agent", "simple"],
    index=0,
    help="agent: uses agentic RAG; simple: uses simple RAG"
)

# Only show these options in simple mode - agent decides parameters autonomously
if analysis_mode == "simple":
    # Vendor selection
    vendor_options = ["All", "cherry_servers", "ovh", "hetzner", "digital_ocean", "scaleway", "vultr"]
    selected_vendor = st.sidebar.selectbox(
        "Cloud Provider",
        vendor_options,
        help="Filter reviews by specific Cloud Infrastructure provider"
    )

    # Chunk type selection
    chunk_type = st.sidebar.selectbox(
        "Search Granularity",
        ["sentence", "review"],
        index=0,
        help="sentence: Analyze each sentence; review: Analyze the entire review"
    )

    # Advanced retrieval settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        st.markdown("**MMR Retrieval Parameters**")

        top_k = st.number_input(
            "Reviews to return",
            min_value=10,
            max_value=100,
            value=12,
            help="Number of reviews to return"
        )

        fetch_k = st.number_input(
            "Reviews to consider",
            min_value=top_k,  # Ensure fetch_k is always >= top_k
            max_value=300,
            value=max(30, top_k),  # Default to 30 or top_k, whichever is higher
            help="Number of reviews to consider before diversification (MMR). Must be >= Reviews to return."
        )

        st.caption(f"Will fetch {fetch_k} candidates and return top {top_k} diverse results")
else:
    # Set default fallbacks for agent mode. Agent will decide actual parameters itself.
    selected_vendor = None
    chunk_type = "sentence"
    top_k = 12
    fetch_k = 30

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.messages:
        st.metric("Messages", len(st.session_state.messages))

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show tool outputs for assistant messages
            if msg.get("tool_outputs"):
                # Count occurrences of each tool to add index for duplicates
                tool_counts = {}
                for i, tool_entry in enumerate(msg["tool_outputs"]):
                    tool_name = tool_entry.get("name", "unknown")
                    output = tool_entry.get("output", {})

                    # Track count for this tool name
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                    current_count = tool_counts[tool_name]

                    # Add index if tool is called multiple times
                    display_name = tool_name.replace('_', ' ').title()
                    if msg["tool_outputs"].count(tool_entry) > 1 or sum(1 for t in msg["tool_outputs"] if t.get("name") == tool_name) > 1:
                        display_name = f"{display_name} #{current_count}"

                    with st.expander(f"📊 {display_name}", expanded=False):
                        # Format tool outputs nicely
                        if tool_name == "sentiment_analysis":
                            st.markdown(format_sentiment_analysis(output))
                        elif tool_name == "aspect_extraction":
                            st.markdown(format_aspect_extraction(output))
                        elif tool_name == "jtbd_analysis":
                            st.markdown(format_jtbd_analysis(output))
                        else:
                            st.markdown(str(output))

            # Show snippets for assistant messages
            if msg.get("snippets"):
                with st.expander(f"📄 Retrieved Context ({len(msg['snippets'])} snippets)", expanded=False):
                    for i, snippet in enumerate(msg["snippets"], 1):
                        source_info = f"**{snippet.get('source', 'Anonymous')}** - {snippet.get('date', 'Unknown date')}"
                        vendor_info = f" - {snippet.get('vendor', '').upper()}" if snippet.get('vendor') else ""
                        rating_info = f" - ⭐ {snippet.get('rating', 'N/A')}/5" if snippet.get('rating') else ""

                        st.markdown(f"**#{i}** {source_info}{vendor_info}{rating_info}")
                        if snippet.get('review_header'):
                            st.markdown(f"*{snippet['review_header']}*")
                        st.markdown(f'"{snippet.get("text", "")}"')
                        st.markdown("---")

    # Example questions - show at top if no messages yet
    if len(st.session_state.messages) == 0:
        with st.expander("💡 Example Questions", expanded=True):
            example_questions = [
                "How customers feel about Cherry Servers?",
                "Compare worst features of Hetzner vs. OVH.",
                "What is the main UX issue with most cloud vendors?",
                "Can you dig deeper into Scaleway pricing concerns?",
            ]

            for i, eq in enumerate(example_questions):
                if st.button(eq, key=f"example_{i}"):
                    st.session_state.selected_question = eq
                    st.rerun()

# Statistics section
with col2:
    try:
        vector_store = get_vector_store()
        if vector_store:
            st.metric("Database Status", "Connected ✅")

        review_stats = get_review_stats()
        if review_stats:
            total_reviews = sum(review_stats.values())
            st.metric("Total Reviews", f"{total_reviews:,}")

            for vendor, count in review_stats.items():
                vendor_display = vendor.replace('_', ' ').title()
                percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
                st.markdown(f"**{vendor_display}:** {count:,} ({percentage:.1f}%)")

    except Exception as e:
        st.error(f"Database connection issue: {str(e)}")

# Check if user selected a suggested question
if st.session_state.get('selected_question'):
    question = st.session_state.selected_question
    st.session_state.selected_question = None  # Clear after use
else:
    question = st.chat_input("Ask about customer reviews...")

# Process chat input
if question:
    if tracker.is_exceeded:
        st.error("🚨 Token limit exceeded! Click 'Clear Conversation' to start fresh.")
    else:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now()
        })

        # Sanitize vendor_param
        vendor_param = None if selected_vendor == "All" else selected_vendor

        # Pass all messages except the one we just added (which is already in question)
        conversation_history = st.session_state.messages[:-1]

        # Stream the response
        try:
            # Create assistant message container
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # Get streaming generator
                if analysis_mode == "agent":
                    stream_gen = agentic_response(
                        question,
                        chunk_type,
                        vendor_param,
                        top_k,
                        fetch_k,
                        conversation_history=conversation_history
                    )
                else:
                    stream_gen = simple_rag_response(
                        question,
                        chunk_type,
                        vendor_param,
                        top_k,
                        fetch_k,
                        conversation_history=conversation_history
                    )

                # Stream and display chunks, capturing return value
                metadata = {"tool_outputs": [], "snippets": []}
                try:
                    while True:
                        chunk = next(stream_gen)
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                except StopIteration as e:
                    # Capture the return value from the generator
                    if e.value:
                        metadata = e.value

                # Remove cursor and show final response
                response_placeholder.markdown(full_response)

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now(),
                "tool_outputs": metadata.get("tool_outputs", []),
                "snippets": metadata.get("snippets", [])
            })

            # Force rerun to display new messages with tool outputs and snippets
            st.rerun()

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.info("Try simplifying your question or checking your filters.")
            # Remove the user message that caused the error
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

# Download option for last assistant message with snippets
if st.session_state.messages:
    last_assistant_msg = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("snippets"):
            last_assistant_msg = msg
            break

    if last_assistant_msg:
        df = pd.DataFrame(last_assistant_msg["snippets"])
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Last Retrieved Data (CSV)",
            data=csv,
            file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.markdown("---")
