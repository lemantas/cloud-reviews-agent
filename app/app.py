import streamlit as st
import pandas as pd
from datetime import datetime
from chains import agentic_response, simple_rag_response
from clients import get_vector_store, set_callbacks
from token_tracker import TokenTracker

# Formatting functions for tool outputs
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

# Page config
st.set_page_config(
    page_title="Customer Reviews RAG",
    page_icon="⭐",
    layout="wide"
)

# Initialize token tracker
tracker = TokenTracker(max_tokens=100000)
set_callbacks([tracker])

# Initialize conversation history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title and header
st.title("⭐ Customer Reviews RAG")
st.markdown("*Analyze customer reviews of your favorite Cloud Infrastructure provider*")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Token usage display
st.sidebar.markdown("---")
st.sidebar.write(f"**Tokens:** {tracker.used:,} / {tracker.max_tokens:,}")
st.sidebar.progress(tracker.used / tracker.max_tokens if tracker.max_tokens > 0 else 0)
if tracker.used / tracker.max_tokens > 0.9:
    st.sidebar.error("⚠️ Token limit almost reached!")

# Clear conversation button
if st.sidebar.button("🗑️ Clear Conversation", help="Start a new conversation"):
    st.session_state.messages = []
    tracker.reset()
    st.rerun()

st.sidebar.markdown("---")

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

# Analysis mode
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["agent", "simple"],
    index=0,
    help="agent: uses domain tools (sentiment, aspects, JTBD) for deeper analysis; simple: uses simple RAG"
)

# Advanced retrieval settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    st.markdown("**MMR Retrieval Parameters**")

    top_k = st.number_input(
        "Reviews to return",
        min_value=1,
        max_value=50,
        value=12,
        help="Number of reviews to return"
    )

    fetch_k = st.number_input(
        "Reviews to consider",
        min_value=1,
        max_value=100,
        value=30,
        help="Number of reviews to consider before diversification (MMR)"
    )

    st.caption(f"Will fetch {fetch_k} candidates and return top {top_k} diverse results")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show tool outputs for assistant messages
            if msg.get("tool_outputs"):
                for tool_name, output in msg["tool_outputs"].items():
                    with st.expander(f"📊 {tool_name.replace('_', ' ').title()}", expanded=False):
                        # Use existing formatters
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
                # Sentiment analysis triggers
                "What are customers' overall sentiment about the service?",
                # Aspect extraction triggers
                "What are the most discussed aspects of the service?",
                # JTBD analysis triggers
                "What job are customers trying to accomplish?",
                # Multiple tools (comprehensive)
                "Give me a complete analysis of customer feedback.",
                # Follow-up examples
                "Tell me about customer issues",
                "Can you dig deeper into pricing concerns?",
            ]

            for i, eq in enumerate(example_questions):
                if st.button(eq, key=f"example_{i}"):
                    st.session_state.selected_question = eq
                    st.rerun()

with col2:
    try:
        vector_store = get_vector_store()

        # Get collection stats
        st.metric("Database Status", "Connected ✅")
        st.info("**Tip**: Use Example Questions to get started. The agent can analyze sentiment, extract key aspects, identify customer jobs-to-be-done, or provide comprehensive analysis. Ask follow-up questions to dig deeper!")

        # Show conversation stats
        if st.session_state.messages:
            st.metric("Messages", len(st.session_state.messages))

    except Exception as e:
        st.error(f"Database connection issue: {str(e)}")

# Chat input - placed outside columns for better layout
# Check for selected question from example buttons
if st.session_state.get('selected_question'):
    question = st.session_state.selected_question
    st.session_state.selected_question = None  # Clear after use
else:
    question = st.chat_input("Ask about customer reviews...")

# Process chat input
if question:
    # Check token limit
    if tracker.is_exceeded:
        st.error("🚨 Token limit exceeded! Click 'Clear Conversation' to start fresh.")
    else:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now()
        })

        # Prepare parameters
        vendor_param = None if selected_vendor == "All" else selected_vendor

        # Show spinner while processing
        with st.spinner("Analyzing reviews..."):
            try:
                # Choose analysis function based on mode
                # Pass all messages except the one we just added (which is already in question)
                conversation_history = st.session_state.messages[:-1]

                if analysis_mode == "agent":
                    result = agentic_response(
                        question,
                        chunk_type,
                        vendor_param,
                        top_k,
                        fetch_k,
                        conversation_history=conversation_history
                    )
                else:
                    result = simple_rag_response(
                        question,
                        chunk_type,
                        vendor_param,
                        top_k,
                        fetch_k,
                        conversation_history=conversation_history
                    )

                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": datetime.now(),
                    "tool_outputs": result.get("tool_outputs", {}),
                    "snippets": result.get("snippets", [])
                })

                # Force rerun to display new messages
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.7em;'>
    🚀 Powered by Turing College<br>
    Mantas Levinas @ 2025
    </div>
    """,
    unsafe_allow_html=True
)