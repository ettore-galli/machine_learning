from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Tool annotation identifies a function as a tool automatically
@tool
def find_sum(x: int, y: int) -> int:
    # The docstring comment describes the capabilities of the function
    # It is used by the agent to discover the function's inputs, outputs and capabilities
    """
    This function is used to add two numbers and return their sum.
    It takes two integers as inputs and returns an integer as output.
    """
    return x + y


@tool
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
        print(args)
        print(kwargs)
        # Deve restituire self per chaining
        return self

    def invoke(self, *args, **kwargs):
        print("\n===== INVOKE CALLED =====")
        print(args)
        print(kwargs)

        # L'agent si aspetta un AIMessage come risposta
        return AIMessage(content="(dummy response)")


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

    print("Step by Step execution : ")

    for message in result["messages"]:
        print(message.pretty_repr())


if __name__ == "__main__":
    main()
