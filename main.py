import re

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool
from langchain_groq import ChatGroq

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )

    return len(text)


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

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, stop=["\nObservation", "Observation"])

    # Create a chain that combines prompt and LLM
    chain = prompt | llm

    # Create a tool map for easy lookup
    tool_map = {tool.name: tool for tool in tools}


    # ReAct agent loop
    def run_agent(input_text: str, max_iterations: int = 3):
        """Run the ReAct agent loop"""
        agent_input = {"input": input_text}
        i = 0

        while i < max_iterations:
            i += 1
            print(f"\n--- Iteration {i} ---")

            # Get response from LLM
            response = chain.invoke(agent_input)
            agent_input["agent_scratchpad"] = response.content
            print(f"LLM Response:\n{response.content}")

            # Parse the response to extract action
            action_match = re.search(r"Action\s*:\s*(\w+)", response.content)
            action_input_match = re.search(r"Action\s*Input\s*:\s*(.*?)(?:\n|$)", response.content)

            if not action_match:
                # No action found, assume final answer
                print("No action found. Agent finished.")
                final_answer_match = re.search(r"Final\s*Answer\s*:\s*(.*?)$", response.content, re.DOTALL)
                if final_answer_match:
                    print(f"Final Answer: {final_answer_match.group(1).strip()}")
                break

            action = action_match.group(1)
            action_input = action_input_match.group(1).strip() if action_input_match else ""

            print(f"Action: {action}")
            print(f"Action Input: {action_input}")

            # Execute the tool
            if action in tool_map:
                tool_result = tool_map[action].invoke(action_input)
                print(f"Observation: {tool_result}")

                # Append to scratchpad
                agent_input["agent_scratchpad"] += f"\nObservation: {tool_result}\nThought: "
            else:
                print(f"Unknown action: {action}")
                break


    # Example usage
    user_question = "What is the length of the text 'Hello World'?"
    run_agent(user_question)
