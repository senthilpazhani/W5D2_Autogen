import asyncio
from dotenv import load_dotenv
load_dotenv()
	
from autogen_agentchat.agents import AssistantAgent
#from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
	
# A web search tool using DuckDuckGo (free, no API key).
def web_search(query: str) -> str:
    """Search the web for the query and return the top results as text."""
    
    from ddgs import DDGS
    
    results = DDGS().text(query, max_results=3)
    
    if not results:
        return "No results found."
    # Turn the results into simple text the agent can read
    
    return "\n\n".join(f"{r['title']}\n{r['body']}" for r in results)
	
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
	
searcher = AssistantAgent(
    name="searcher",
    model_client=model_client,
    tools=[web_search],                 # the web search tool
    reflect_on_tool_use=True,
    system_message="You answer questions using the web_search tool. Keep answers short.",
)

async def main():

    result = await searcher.run(
        task="Who won the latest FIFA World Cup? Search the web."
    )

    print("Number of messages:", len(result.messages))
    print("Who answered:", result.messages[-1].source)
    print("Request:", result.messages[0].content)
    print("Answer:\n", result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())