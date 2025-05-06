import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool

# Load environment variables
load_dotenv()

@function_tool
def get_weather(city: str) -> str:
    """
    Get the weather for a city.
    
    Args:
        city: The name of the city to get the weather for
        
    Returns:
        The weather forecast
    """
    return f"The weather in {city} is sunny and 25°C."

async def main():
    """
    Simple test for OpenAI Agents SDK
    """
    print("Testing OpenAI Agents SDK...")
    
    # Create a simple agent
    agent = Agent(
        name="Weather Agent",
        instructions="You help users check the weather.",
        tools=[get_weather]
    )
    
    # Run the agent
    result = await Runner.run(agent, input="What's the weather in Istanbul?")
    print(f"Agent response: {result.final_output}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 