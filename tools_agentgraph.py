from langgraph.graph import StateGraph, START
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun

import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")


def search_duckduckgo(query: str):
    """Search DuckDuckGo for real-time info."""
    return DuckDuckGoSearchRun().invoke(query)

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [search_duckduckgo, add, multiply]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    
    system_msg = SystemMessage(content="""
    You are a helpful assistant. 
    - Use ONLY: search_duckduckgo, add, multiply.
    - Never use brave_search or any other tool.
    - For weather: use search_duckduckgo.
    """)
    messages = [system_msg] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


graph = StateGraph(State)
graph.add_node("assistant", chatbot)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")
app = graph.compile()


response = app.invoke({
    "messages": [HumanMessage(content="What is the current population in karachi? and after giving the answer of population multiply it by 2. ok?")]
})

print(response["messages"][-1].content)
