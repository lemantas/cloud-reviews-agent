from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path


def load_prompt(filename):
    """Load prompt content from <project_root>/data/prompts."""
    prompts_dir = Path(__file__).resolve().parent.parent / "data" / "prompts"
    filepath = prompts_dir / filename
    return filepath.read_text(encoding="utf-8").strip()

# Load prompts from files
RAG_SYSTEM_PROMPT = load_prompt('rag_system.txt')
AGENT_SYSTEM_PROMPT = load_prompt('agent_system.txt')
JTBD_ANALYSIS_PROMPT = load_prompt('jtbd_analysis.txt')
ASPECTS_ANALYSIS_PROMPT = load_prompt('aspects_analysis.txt')
SENTIMENT_ANALYSIS_PROMPT = load_prompt('sentiment_analysis.txt')

# Template constructors
def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Question: {question}")
    ])

def get_agent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "Question: {question}")
    ])

def get_sentiment_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SENTIMENT_ANALYSIS_PROMPT),
        ("human", "You may use this human question to help you analyze the sentiment of the reviews: {question}")
    ])

def get_aspects_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", ASPECTS_ANALYSIS_PROMPT),
        ("human", "You may use this human question to help you analyze the aspects of the reviews: {question}")
    ])

def get_jtbd_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", JTBD_ANALYSIS_PROMPT),
        ("human", "You may use this human question to help you analyze the JTBD of the reviews: {question}")
    ])