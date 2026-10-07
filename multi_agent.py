import asyncio
from dotenv import load_dotenv
load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Agent 1: the writer
writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="You write a short birthday message. Improve it if the reviewer asks.",
)

# Agent 2: the reviewer
reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="Review the message. If it is good, reply with the single word APPROVE. Otherwise suggest one change.",
)

# Stop when the reviewer says APPROVE
stop = TextMentionTermination("APPROVE")

# Build the team: they take turns (writer, reviewer, writer, ...)
team = RoundRobinGroupChat([writer, reviewer], termination_condition=stop)

async def main():
    result = await team.run(task="Write a warm birthday message for my friend Sam.")

    # Print the whole conversation
    for m in result.messages:
        print(f"[{m.source}] {m.content}\n")

if __name__ == "__main__":
    asyncio.run(main())