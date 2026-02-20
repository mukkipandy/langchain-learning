from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tavily_search_tool = TavilySearch()

    tools = [tavily_search_tool, multiply]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

res = agent_executor.invoke(
    {
        "input": "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious",
    }
)

print(res)
