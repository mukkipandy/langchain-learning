from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


# 1. Initialize the Tavily Search tool
tavily_search_tool = TavilySearch()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return "hard coded response"


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [tavily_search_tool]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-learning!")
    response = agent.invoke(
        {"messages": HumanMessage(
            content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
        )})
    # Extract just the final AI message content
    final_message = response['messages'][-1]  # Get the last message in the list
    print(f"Final Answer: {final_message.content}")


if __name__ == "__main__":
    main()
