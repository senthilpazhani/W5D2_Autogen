import asyncio
from dotenv import load_dotenv
load_dotenv() 

from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.teams import RoundRobinGroupChat 
from autogen_agentchat.conditions import TextMentionTermination 
from autogen_agentchat.ui import Console 
from autogen_ext.models.openai import OpenAIChatCompletionClient 

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini") 

writer = AssistantAgent(name="writer", model_client=model_client, system_message="Write one short slogan. Improve it if asked.") 

reviewer = AssistantAgent(name="reviewer", model_client=model_client, system_message="If the slogan is catchy, reply APPROVE. Else suggest one change.") 

team = RoundRobinGroupChat([writer, reviewer], termination_condition=TextMentionTermination("APPROVE")) 

async def main():
    # Watch the team work LIVE with Console 
    result = await Console(team.run_stream(task="Create a slogan for a coffee shop.")) 
    print(result.messages[-1].content) 

    # Clear the team's memory of the coffee slogan 
    await team.reset() 

    # Now run a brand-new, unrelated task 
    result = await team.run(task="Create a slogan for a book store.") 
    print(result.messages[-1].content) 


if __name__ == "__main__":
    asyncio.run(main())