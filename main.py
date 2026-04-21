import os
import json
import re
import sqlite3
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag import get_retriever

load_dotenv()

# 1. DATABASE INITIALIZATION
def init_db():
    conn = sqlite3.connect('leads.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            platform TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 2. OPENROUTER CONFIGURATION (Claude 3 Haiku)
MODEL_ID = "anthropic/claude-3-haiku" 

llm = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# 3. STATE DEFINITION [cite: 89, 91]
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Conversation history"]
    intent: str
    user_data: dict
    is_complete: bool

retriever = get_retriever()

# 4. TOOL EXECUTION & DATABASE STORAGE [cite: 53, 54, 74]
def mock_lead_capture(name, email, platform):
    """Stores the lead details in the local SQLite database."""
    try:
        conn = sqlite3.connect('leads.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO leads (name, email, platform) VALUES (?, ?, ?)',
            (name, email, platform)
        )
        conn.commit()
        conn.close()
        print(f"\n>>> [DATABASE SUCCESS] Lead stored: {name}, {email}, {platform}")
        return f"Lead captured and saved to database successfully: {name}, {email}, {platform}"
    except Exception as e:
        return f"Error saving lead to database: {e}"

# 5. NODES
def intent_classifier(state: AgentState):
    """Classifies intent: greeting, inquiry, or high_intent [cite: 20-23]"""
    last_msg = state["messages"][-1].content
    prompt = f"Classify intent: '{last_msg}'. Choices: greeting, inquiry, high_intent. Reply with ONLY the word."
    res = llm.invoke(prompt)
    return {"intent": res.content.strip().lower()}

def rag_node(state: AgentState):
    """RAG retrieval from knowledge base [cite: 24, 25]"""
    last_msg = state["messages"][-1].content
    docs = retriever.invoke(last_msg)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\nUser: {last_msg}\nAnswer concisely:"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content)]}

def lead_capture_node(state: AgentState):
    """Handles multi-turn data collection and triggers DB storage [cite: 44-51]"""
    last_msg = state["messages"][-1].content
    data = state.get("user_data") or {"name": None, "email": None, "platform": None}

    # Strict extraction to prevent invalid inputs like 'none-of-your-business'
    extract_prompt = f"""
    Message: "{last_msg}"
    Current Data: {data}
    Extract 'name', 'email', and 'platform'. 
    - Set 'email' ONLY if it is a valid address format. 
    - Ignore refusals or sarcasm (set as null).
    Return ONLY JSON: {{"name": val, "email": val, "platform": val}}
    """
    try:
        res = llm.invoke(extract_prompt)
        ext = json.loads(res.content.replace("```json", "").replace("```", "").strip())
        if ext.get("name"): data["name"] = ext["name"]
        if ext.get("platform"): data["platform"] = ext["platform"]
        # Basic regex check for email extraction
        if ext.get("email") and re.match(r'[^@]+@[^@]+\.[^@]+', ext["email"]):
            data["email"] = ext["email"]
    except:
        pass

    # Success: All 3 fields collected [cite: 51, 73, 74]
    if data["name"] and data["email"] and data["platform"]:
        result = mock_lead_capture(data["name"], data["email"], data["platform"])
        return {"messages": [AIMessage(content=result)], "user_data": data, "is_complete": True}
    
    # Prompting for missing details [cite: 68-70]
    if not data["name"]: msg = "I'd love to help you sign up! What is your full name?"
    elif not data["email"]: msg = f"Nice to meet you, {data['name']}! What is your email address?"
    else: msg = "And which creator platform do you use (e.g., YouTube, TikTok)?"
    
    return {"messages": [AIMessage(content=msg)], "user_data": data}

# 6. GRAPH CONSTRUCTION [cite: 78, 79]
workflow = StateGraph(AgentState)
workflow.add_node("classify", intent_classifier)
workflow.add_node("respond", rag_node)
workflow.add_node("capture", lead_capture_node)
workflow.set_entry_point("classify")

def router(state):
    if state.get("user_data") and not state.get("is_complete"): return "capture"
    if "high_intent" in state["intent"]: return "capture"
    if "inquiry" in state["intent"]: return "respond"
    return END

workflow.add_conditional_edges("classify", router)
workflow.add_edge("respond", END)
workflow.add_edge("capture", END)

app = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "final_db_demo"}}
    print("--- AutoStream Agent (Claude + SQLite) Ready ---")
    while True:
        u = input("\nUser: ")
        if u.lower() in ["exit", "quit"]: break
        out = app.invoke({"messages": [HumanMessage(content=u)]}, config=config)
        print(f"Agent: {out['messages'][-1].content}")