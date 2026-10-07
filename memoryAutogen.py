import asyncio
from dotenv import load_dotenv
load_dotenv()
	
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import ListMemory, MemoryContent
	
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
	
async def main():
    # 1) Make a memory and 2) add some facts to remember
    memory = ListMemory()
    await memory.add(MemoryContent(content="The user's name is Senthilkumar.", mime_type="text/plain"))
    await memory.add(MemoryContent(content="The user loves Swimming.", mime_type="text/plain"))

    # 3) Give the memory to the agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[memory],                    # the agent will read these notes
        system_message="You are a friendly assistant.",
    )
    # 4) Ask something that needs the remembered facts
    result = await agent.run(task="What is my name, and suggest a sport I would enjoy.")

    print("Number of messages:", len(result.messages))
    print("Who answered:", result.messages[-1].source)
    print("Request:", result.messages[0].content)
    print("Answer:\n", result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())