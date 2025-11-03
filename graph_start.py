import os
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class State(Dict):
    messages: List[Dict[str, str]]


graph_builder = StateGraph(State)


llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.7,  
    api_key=GROQ_API_KEY
)


def chatbot(state: State):
    response = llm.invoke(state["messages"])  
    state["messages"].append({"role": "assistant", "content": str(response.content)})
    return {"messages": state["messages"]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(state):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"], flush=True) 


if __name__ == "__main__":
    print("Chatbot started with Groq! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
