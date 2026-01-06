import os
from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

# --- Tools ---

# 1. RAG Tool
CHROMA_PATH = r"chroma_db"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@tool
def lookup_resume(query: str):
    """Consult the user's resume and portfolio data to answer questions about skills, projects, and experience."""
    # A retriever-as-tool
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# 2. GitHub Stats Tool (Mocked or Real API could be added here)
@tool
def get_github_stats():
    """Fetch live GitHub statistics (repos, stars, contributions) for the portfolio owner Harshit4705."""
    import requests
    username = "Harshit4705"
    try:
        # Basic user info
        user_url = f"https://api.github.com/users/{username}"
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
        
        user_response = requests.get(user_url)
        repos_response = requests.get(repos_url)
        
        if user_response.status_code == 200 and repos_response.status_code == 200:
            user_data = user_response.json()
            repos_data = repos_response.json()
            
            # Calculate total stars
            total_stars = sum(repo.get("stargazers_count", 0) for repo in repos_data)
            
            # Get top 5 repos by stars
            top_repos = sorted(repos_data, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
            top_repos_str = "\n".join([f"  - {r['name']} ({r['stargazers_count']} ⭐)" for r in top_repos])
            
            return (
                f"GitHub User: {user_data.get('login')}\n"
                f"Name: {user_data.get('name')}\n"
                f"Bio: {user_data.get('bio')}\n"
                f"Public Repos: {user_data.get('public_repos')}\n"
                f"Followers: {user_data.get('followers')}\n"
                f"Following: {user_data.get('following')}\n"
                f"Total Stars: {total_stars}\n"
                f"Top Repositories:\n{top_repos_str}"
            )
        else:
            return "Could not fetch GitHub data at this time."
    except Exception as e:
        return f"Error fetching GitHub data: {str(e)}"

tools = [lookup_resume, get_github_stats]

# --- LLM ---
if not os.environ.get("GROQ_API_KEY"):
    # Fallback for dev if key is missing, though it will crash on run.
    print("WARNING: GROQ_API_KEY not found in env.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

# --- Graph Definition ---

class State(TypedDict):
    messages: Annotated[list, add_messages]

# System prompt for concise responses
SYSTEM_PROMPT = """You are Harshit Chawla's AI portfolio assistant. Be CONCISE and DIRECT.

**CRITICAL - ACCURACY RULES:**
- ALWAYS use the lookup_resume tool before answering ANY question about Harshit
- ONLY use information from the resume lookup tool - NEVER make up or assume information
- If the resume doesn't mention something, say "I don't have that information in the resume"
- Harshit is an **Aspiring AI/ML Engineer**, NOT a full stack developer
- Harshit is pursuing **BCA (Bachelor of Computer Applications)** at **GGSIPU**, NOT B.Tech at IIIT

**Response Rules:**
- Keep responses under 100 words unless the user asks for details
- Use bullet points for lists
- Don't repeat information
- Be friendly but brief
- For simple questions, give simple answers
- Only elaborate when specifically asked
- **IMPORTANT**: ALWAYS format URLs as Markdown links, e.g., [Title](url). Never output raw URLs.

**Contact Info (You can share this):**
- Phone: +91-9560700282
- LinkedIn: [Harshit Chawla](https://linkedin.com/in/harshitchawla4705)

You have access to tools to lookup Harshit's resume and GitHub stats. ALWAYS use lookup_resume before answering."""

def chatbot(state: State):
    # Prepend system message if not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
