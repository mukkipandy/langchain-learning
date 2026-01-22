from typing import List, Union

from dotenv import load_dotenv
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Create the agent chain
    agent = (
            {
                "input": lambda x: x["input"],
                "intermediate_steps": lambda x: x.get("intermediate_steps", []),
            }
            | prompt
            | llm
            | ReActSingleInputOutputParser()
    )

    # Initialize variables
    intermediate_steps = []
    max_iterations = 5  # Prevent infinite loops

    # Agent input
    agent_input = {
        "input": "What is the length of 'DOG' in characters?",
        "intermediate_steps": intermediate_steps,
    }

    # Run the agent loop
    for i in range(max_iterations):
        print(f"\n=== Step {i + 1} ===")

        # Get agent's next step
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(agent_input)
        print(f"Agent Step: {agent_step}")

        if isinstance(agent_step, AgentFinish):
            # Agent has finished
            print(f"\n✅ Final Answer: {agent_step.return_values['output']}")
            break

        elif isinstance(agent_step, AgentAction):
            # Agent wants to use a tool
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            # Execute the tool
            observation = tool_to_use.func(str(tool_input))
            print(f"Tool Observation: {observation}")

            # Update intermediate steps
            intermediate_steps.append((agent_step, observation))

            # Update input for next iteration
            agent_input = {
                "input": agent_input["input"],
                "intermediate_steps": intermediate_steps,
            }
        else:
            print(f"Unexpected agent step type: {type(agent_step)}")
            break

        if i == max_iterations - 1:
            print(f"\n⚠️ Maximum iterations ({max_iterations}) reached without final answer")