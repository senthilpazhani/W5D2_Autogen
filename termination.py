import asyncio
from dotenv import load_dotenv
load_dotenv()
	
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
	
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
	
writer = AssistantAgent(name="writer", model_client=model_client,
    system_message="Write a short thank-you note. Improve it if asked.")

reviewer = AssistantAgent(name="reviewer", model_client=model_client,
    system_message="If the note is polite, reply APPROVE. Else suggest one change.")
	
# Stop when APPROVE appears OR after 6 messages (whichever comes first)
stop = TextMentionTermination("APPROVE") | MaxMessageTermination(6)

team = RoundRobinGroupChat([writer, reviewer], termination_condition=stop)

async def main():
    result = await team.run(task="Write a thank-you note to a teacher.")
    print("Total messages:", len(result.messages))
    print("Why it stopped:", result.stop_reason)

if __name__ == "__main__":
    asyncio.run(main())