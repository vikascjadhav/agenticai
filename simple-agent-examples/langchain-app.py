from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import warnings

warnings.filterwarnings("ignore")

load_dotenv()


def local_gemma_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=os.environ["API_KEY"],
        model="google/gemma-3-4b",
    )


@tool
def fruits_and_vegetables():
    """Tool to generate list items for fruits and vegetables."""
    return [
        "Apple",
        "Banana",
        "Carrot",
        "Spinach",
        "Grapes",
        "Broccoli",
        "Custom-fruit-list-item-1",
        "Custom-fruit-list-item-2",
    ]


@tool
def weather_tool(location: str) -> str:
    # In a real implementation, you would call a weather API here.
    # For demonstration purposes, we'll return a static response.
    """Tool to get weather information for a given location."""
    if location == "Tokyo":
        return "The current weather in Tokyo is cloudy with a temperature of 20°C."
    elif location == "Mumbai":
        return "The current weather in Mumbai is rainy with a temperature of 28°C."
    elif location == "New York":
        return "The current weather in New York is sunny with a temperature of 22°C."
    else:
        return f"The current weather in {location} is not available."


def weather_agent():

    llm = local_gemma_model()

    agent = create_agent(
        model=llm,
        system_prompt="You are a helpful assistant that provides weather information.",
        tools=[weather_tool, fruits_and_vegetables],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "weather in Tokyo and Mumbai"}]}
    )
    for msg in result["messages"]:
        print(type(msg).__name__)
        print(msg.content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("Tool calls:\n", msg.tool_calls)
        print("-----")


def fruit_weather_agent():
    llm = local_gemma_model()
    agent = create_agent(
        model=llm,
        system_prompt="You are a helpful assistant that provides information about fruits and vegetables and there might custom names returned by tool.",
        tools=[weather_tool, fruits_and_vegetables],
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "list fruits and vegetables & weather in Tokyo, Mumbai and New York",
                }
            ]
        }
    )
    for msg in result["messages"]:
        print(type(msg).__name__)
        print(msg.content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("Tool calls:\n", msg.tool_calls)
        print("-----")


def main():

    fruit_weather_agent()
    weather_agent()


if __name__ == "__main__":
    main()
