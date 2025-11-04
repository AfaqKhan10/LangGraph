
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
load_dotenv()



api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0,
    api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

llm_tool = llm.bind_tools([wiki_tool])
agent_executor = create_react_agent(llm, [wiki_tool])  

response = agent_executor.invoke({
    "messages": [HumanMessage(content="What is agentic AI")]
})

print(response["messages"][-1].content)
