# LANGGRAPH CUSTOMER SUPPORT BOT

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class State(TypedDict):
    query: str
    category: str
    response: str


# Supervisor: Classifies the query
def supervisor_agent(state: State):
    prompt = f"""
    Query: {state['query']}
    
    Reply with exactly ONE word from this list:
    tech
    billing
    refund
    general
    
    Nothing else.
    """
    category = llm.invoke(prompt).content.strip().lower()
    print(f"[Supervisor] → {category}")
    return {"category": category}

# Worker: Generates the final answer
def worker_agent(state: State):
    category = state["category"]
    query = state["query"]

    prompts = {
        "tech": f"You are a tech support agent. Answer in 2-3 short sentences, giving clear and simple troubleshooting steps.\nUser query: {query}",
        "billing": f"You are a billing support agent. Answer in a polite and helpful way using only 2-3 short sentences.\nUser query: {query}",
        "refund": f"You are a refund specialist. Provide a short, honest, and helpful reply (2-3 sentences max).\nUser query: {query}",
        "general": f"You are a friendly support assistant. Respond naturally in 1-2 short sentences.\nUser query: {query}"
    }
    response = llm.invoke(prompts.get(category, prompts["general"])).content
    print(f"[Worker] → {category} category handled\n")
    return {"response": response}


def route_next(state: State) -> Literal["worker", END]:
    return "worker" if state["category"] in ["tech", "billing", "refund", "general"] else END


# Build the graph
graph = StateGraph(State)
# add nodes
graph.add_node("supervisor", supervisor_agent)
graph.add_node("worker", worker_agent)
graph.set_entry_point("supervisor")
# add edges
graph.add_conditional_edges("supervisor", route_next)
graph.add_edge("worker", END)
app = graph.compile()

# Test function
def ask(question: str):
    result = app.invoke({"query": question})
    print("User:", question)
    print("Bot:", result["response"])

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    ask(user_query)

