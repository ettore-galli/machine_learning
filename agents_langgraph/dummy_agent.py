from typing import List

from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# Tool annotation identifies a function as a tool automatically
@tool()
def find_sum(x: int, y: int) -> int:
    # The docstring comment describes the capabilities of the function
    # It is used by the agent to discover the function's inputs, outputs and capabilities
    """
    This function is used to add two numbers and return their sum.
    It takes two integers as inputs and returns an integer as output.
    """
    return x + y


@tool()
def find_product(x: int, y: int) -> int:
    """
    This function is used to multiply two numbers and return their product.
    It takes two integers as inputs and returns an integer as ouput.
    """
    return x * y


class DummyModel:
    def __init__(self):
        pass

    def bind_tools(self, *args, **kwargs):
        print("\n===== BIND_TOOLS CALLED =====")
        # print(args)
        # print(kwargs)
        # Deve restituire self per chaining
        return self

    def invoke(self, messages, *args, **kwargs):
        print("\n===== INVOKE CALLED =====")
        # print(args)
        # print(kwargs)

        # print(messages)

        if any(m.type == "tool" for m in messages):
            tool_message = [m for m in messages if m.type == "tool"][0]
            return AIMessage(content=f"Risultato: {tool_message.content}")

        if check_word_in_message("sum", messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "find_sum", "name": "find_sum", "args": {"x": 2, "y": 3}}
                ],
            )

        if check_word_in_message("product", messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "find_product",
                        "name": "find_product",
                        "args": {"x": 2, "y": 3},
                    }
                ],
            )

        return AIMessage(content="don't know")


def check_word_in_message(word: str, message: AIMessage) -> bool:
    return any(word.lower() in block for block in retrieve_ai_messages(message=message))


def retrieve_ai_messages(message: AIMessage) -> List[str]:
    return [message.content for message in message]


agent_tools = [find_sum, find_product]

# System prompt
system_prompt = SystemMessage(
    """You are a Math genius who can solve math problems. Solve the
    problems provided by the user, by using only tools available. 
    Do not solve the problem yourself"""
)


agent_graph = create_agent(
    model=DummyModel(), system_prompt=system_prompt, tools=agent_tools
)


def main():
    inputs = {"messages": [("user", "what is the sum of 2 and 3 ?")]}

    result = agent_graph.invoke(inputs)

    # Get the final answer
    print(f"Agent returned : {result['messages'][-1].content} \n")

    # print("Step by Step execution : ")

    # for message in result["messages"]:
    #     print(message.pretty_repr())


if __name__ == "__main__":
    main()
