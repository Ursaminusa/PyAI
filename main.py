from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def addint(a: float, b: float) -> str:
    """Useful for performing basic arithmeric calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"


@tool
def greet(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hey {name}, how is life treating treating you"


def main():
    model = ChatOpenAI(temperature=0)

    tools = [addint, greet]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type N to exit.")
    print("You can ask me to perform addition or chat with me or make me use Chatgpt LLM.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "N":
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()


if __name__ == "__main__":
    main()
