# token_tracker.py
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

class TokenTracker(BaseCallbackHandler):

    def __init__(self, max_tokens=100000):
        self.max_tokens = max_tokens
        if 'tokens' not in st.session_state:
            st.session_state.tokens = 0

    def on_llm_end(self, response, **kwargs):
        """Auto-track tokens from LLM."""
        if hasattr(response, 'llm_output') and response.llm_output is not None:
            tokens = response.llm_output.get('token_usage', {}).get('total_tokens', 0)
            st.session_state.tokens += tokens

    @property
    def used(self):
        return st.session_state.tokens

    @property
    def remaining(self):
        return self.max_tokens - st.session_state.tokens

    @property
    def is_exceeded(self):
        return st.session_state.tokens >= self.max_tokens

    def reset(self):
        st.session_state.tokens = 0
