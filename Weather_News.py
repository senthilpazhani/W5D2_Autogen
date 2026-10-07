import asyncio
from dotenv import load_dotenv
load_dotenv() 

from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.teams import RoundRobinGroupChat 
from autogen_agentchat.conditions import TextMentionTermination 
from autogen_agentchat.ui import Console 
from autogen_ext.models.openai import OpenAIChatCompletionClient 

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini") 

city = input("Enter the City Name.")

def web_search(city: str) -> str:
    """Search the web for the weather report for specified city."""
    
    from ddgs import DDGS
    
    results = DDGS().text(query=f"{city} weather today", max_results=5)
    
    if not results:
        return "No results found."
    # Turn the results into simple text the agent can read
    
    return "\n\n".join(f"{r['title']}\n{r['body']}" for r in results)

agent1 = AssistantAgent(
    name="Weather_Reporter", 
    model_client=model_client, 
    tools=[web_search],
    reflect_on_tool_use=True,
    system_message="Search the web for the weather report for the requested city and summarize it in 3 lines."
) 

async def main():
    # Watch the team work LIVE with Console 
    result = await Console(agent1.run_stream(task=f"Give the Weather News of the city {city}")) 
    await model_client.close()
    #print(result.messages[-1].content) 


if __name__ == "__main__":
    asyncio.run(main())