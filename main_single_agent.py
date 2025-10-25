"""Single-agent AutoGen sample backed by a CLI model client."""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent

from cli_clients import GeminiCliChatCompletionClient


async def main() -> None:
    assistant = AssistantAgent(
        name="gemini_cli",
        system_message="You are a concise assistant powered by Gemini CLI.",
        model_client=GeminiCliChatCompletionClient(debug=True),
    )

    try:
        result = await assistant.run(task="Summarize why agent-to-agent workflows can be useful.")
    except Exception as exc:
        print(f"[Gemini CLI error] {exc}")
        return

    last_message = next(
        (msg for msg in reversed(result.messages) if hasattr(msg, "to_text") and msg.to_text()),
        None,
    )
    if last_message is None:
        print("No text response returned.")
        return

    print(last_message.to_text())


if __name__ == "__main__":
    asyncio.run(main())
