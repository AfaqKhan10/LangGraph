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

# === LLM ===
llm = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")

# === Tools ===
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

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

# === Nodes ===
def chatbot(state: State):
    # Force LLM to use only given tools
    system_msg = SystemMessage(content="""
    You are a helpful assistant. 
    - Use ONLY: search_duckduckgo, add, multiply.
    - Never use brave_search or any other tool.
    - For weather: use search_duckduckgo.
    """)
    messages = [system_msg] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

# === Graph ===
graph = StateGraph(State)
graph.add_node("assistant", chatbot)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")
app = graph.compile()

# === Run ===
response = app.invoke({
    "messages": [HumanMessage(content="What is the current population in karachi? and after giving the answer of population multiply it by 2. ok?")]
})

print(response["messages"][-1].content)


















































# # from IPython.display import Image,display
# from langgraph.graph import StateGraph,START
# from langchain_groq import ChatGroq 
# import requests
# from langchain_core.messages import SystemMessage, HumanMessage
# from langgraph.graph import MessagesState
# from langchain_community.tools import DuckDuckGoSearchRun

# from langgraph.prebuilt import ToolNode, tools_condition

# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages

# import os
# from dotenv import load_dotenv
# load_dotenv()




# class State(TypedDict):
#     messages: Annotated[list, add_messages]



# groq_api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(
#     temperature=0,
#     api_key=groq_api_key,
#     model="llama-3.1-8b-instant"
# )
# # result = llm.invoke('hello').content
# # print(result)


# def search_duckduckgo(query: str):
#     """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
#     search = DuckDuckGoSearchRun()
#     return search.invoke(query)
# # result = search_duckduckgo("what are AI agent")
# # print(result)
     

# def multiply(a:int,b:int) -> int:
#     """
#     Multiply a and b
#     """
#     return a* b

# def add(a:int,b:int) -> int:
#     """
#     Adds a and b
#     """
#     return a + b


# tools = [search_duckduckgo, add, multiply]
# llm_with_tools = llm.bind_tools(tools)

# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}




# graph_builder = StateGraph(State)
# # Define nodes
# graph_builder.add_node("assistant",chatbot)
# graph_builder.add_node("tools",ToolNode(tools))

# #define edges
# graph_builder.add_edge(START,"assistant")
# graph_builder.add_conditional_edges("assistant",tools_condition)
# graph_builder.add_edge("tools","assistant")

# react_graph=graph_builder.compile()


# response = react_graph.invoke({"messages": [HumanMessage(content="what is the weather in Lahore. Multiply it by 2 and add 5.")]})
# print(response["messages"])


