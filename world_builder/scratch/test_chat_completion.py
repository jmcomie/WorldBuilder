import asyncio

from gstk.models.chatgpt import Message, Role
from gstk.llmlib.async_openai import get_chat_completion_response
from openai import ChatCompletion


async def main():
    messages: list[Message] = [
        Message(
            role=Role.USER,
            content="You are a wizard.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I am a wizard.",
        ),
        Message(
            role=Role.USER,
            content="You are a wizard.",
        ),
    ]
    return await get_chat_completion_response(messages)


if __name__ == "__main__":
    chat_completion: ChatCompletion = asyncio.run(main())