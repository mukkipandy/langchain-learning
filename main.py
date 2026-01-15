from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()