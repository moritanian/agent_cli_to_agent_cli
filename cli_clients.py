"""CLI-backed ChatCompletionClient implementations for AutoGen."""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Callable, Optional, Sequence

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _render_user_content(content: str | Sequence[Any]) -> str:
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
    return "\n".join(chunks)


def _render_assistant_content(content: str | Sequence[FunctionCall]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(f"[FunctionCall: {call.name}]" for call in content)


def _format_messages(messages: Sequence[LLMMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            lines.append(f"System: {message.content}")
        elif isinstance(message, UserMessage):
            user_text = _render_user_content(message.content)
            lines.append(f"{message.source}: {user_text}")
        elif isinstance(message, AssistantMessage):
            assistant_text = _render_assistant_content(message.content)
            lines.append(f"{message.source}: {assistant_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


class CLIChatCompletionClient(ChatCompletionClient):
    """Generic CLI-backed chat completion client."""

    def __init__(
        self,
        *,
        make_argv: Callable[[str], Sequence[str]],
        parse_response: Callable[[str, str], str],
        model_family: ModelFamily.ANY | str = ModelFamily.UNKNOWN,
        vision: bool = False,
        function_calling: bool = False,
        json_output: bool = False,
        structured_output: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._make_argv = make_argv
        self._parse_response = parse_response
        self._model_info = ModelInfo(
            vision=vision,
            function_calling=function_calling,
            json_output=json_output,
            structured_output=structured_output,
            multiple_system_messages=True,
            family=model_family,
        )
        self._capabilities = ModelCapabilities(
            vision=vision,
            function_calling=function_calling,
            json_output=json_output,
        )
        self._last_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._debug = debug

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[Any]] = None,
        extra_create_args: Optional[dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        text, usage = await self._execute(messages)
        self._record_usage(usage)
        return CreateResult(
            finish_reason="stop",
            content=text,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=None,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[Any]] = None,
        extra_create_args: Optional[dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        async def _generator():
            text, usage = await self._execute(messages)
            self._record_usage(usage)
            yield text
            yield CreateResult(
                finish_reason="stop",
                content=text,
                usage=usage,
                cached=False,
                logprobs=None,
                thought=None,
            )

        return _generator()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last_usage

    def total_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._prompt_tokens_total,
            completion_tokens=self._completion_tokens_total,
        )

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        prompt = _format_messages(messages)
        return max(len(prompt) // 4, 1)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        estimated_limit = 131072
        return max(estimated_limit - self.count_tokens(messages), 0)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return self._capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    async def _execute(self, messages: Sequence[LLMMessage]) -> tuple[str, RequestUsage]:
        prompt = _format_messages(messages)
        argv = list(self._make_argv(prompt))

        if self._debug:
            print("=== CLI debug ===")
            print("Command:", " ".join(argv))
            preview = prompt[:200].replace("\n", "\\n")
            trailer = "..." if len(prompt) > 200 else ""
            print("Prompt preview:", preview, trailer)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")

        if self._debug:
            print("Return code:", proc.returncode)
            print("STDOUT:", stdout.strip()[:500])
            print("STDERR:", stderr.strip()[:500])
            print("=== End CLI debug ===")

        if proc.returncode != 0:
            raise RuntimeError(stderr.strip() or stdout.strip() or f"CLI exited with {proc.returncode}")

        text = self._parse_response(stdout, stderr)
        usage = RequestUsage(
            prompt_tokens=self.count_tokens(messages),
            completion_tokens=max(len(text) // 4, 1) if text else 0,
        )
        return text, usage

    def _record_usage(self, usage: RequestUsage) -> None:
        self._last_usage = usage
        self._prompt_tokens_total += usage.prompt_tokens
        self._completion_tokens_total += usage.completion_tokens


class GeminiCliChatCompletionClient(CLIChatCompletionClient):
    """Gemini CLI-backed chat completion client."""

    def __init__(
        self,
        *,
        cli_path: str = "gemini",
        model: str | None = DEFAULT_GEMINI_MODEL,
        extra_flags: Sequence[str] | None = None,
        debug: bool = False,
    ) -> None:
        extra_flags = list(extra_flags or [])

        def make_argv(prompt: str) -> Sequence[str]:
            argv: list[str] = [cli_path]
            if model:
                argv.extend(["-m", model])
            argv.extend(["-p", prompt, "-o", "json"])
            argv.extend(extra_flags)
            return argv

        super().__init__(
            make_argv=make_argv,
            parse_response=self._parse_response,
            model_family=model or ModelFamily.UNKNOWN,
            json_output=True,
            debug=debug,
        )

    @staticmethod
    def _parse_response(stdout: str, stderr: str) -> str:
        payload = _parse_json(stdout)
        if _payload_has_error(payload):
            raise RuntimeError(
                "Gemini CLI reported an API error. Check CLI stderr or /tmp/gemini-client-error-*.json for details."
            )
        return _extract_text_from_payload(payload, stdout)


class CodexCliChatCompletionClient(CLIChatCompletionClient):
    """Codex CLI-backed chat completion client (text or JSONL output)."""

    def __init__(
        self,
        *,
        cli_path: str = "codex",
        model: str | None = None,
        prompt_flag: str | None = None,
        subcommand: Sequence[str] | None = ("exec", "--json"),
        output_flags: Sequence[str] | None = None,
        extra_flags: Sequence[str] | None = None,
        debug: bool = False,
    ) -> None:
        output_flags = list(output_flags or [])
        extra_flags = list(extra_flags or [])
        subcommand = list(subcommand or [])

        def make_argv(prompt: str) -> Sequence[str]:
            argv: list[str] = [cli_path]
            argv.extend(subcommand)
            if model:
                argv.extend(["-m", model])
            argv.extend(output_flags)
            argv.extend(extra_flags)
            if prompt_flag:
                argv.extend([prompt_flag, prompt])
            else:
                argv.append(prompt)
            return argv

        super().__init__(
            make_argv=make_argv,
            parse_response=self._parse_response,
            model_family=model or ModelFamily.UNKNOWN,
            debug=debug,
        )

    @staticmethod
    def _parse_response(stdout: str, stderr: str) -> str:
        text = stdout.strip()
        if not text:
            if stderr.strip():
                raise RuntimeError(stderr.strip())
            return ""

        lines = [line for line in text.splitlines() if line.strip()]
        agent_messages: list[str] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                msg_text = item.get("text")
                if isinstance(msg_text, str):
                    agent_messages.append(msg_text)
        if agent_messages:
            return "\n".join(agent_messages).strip()
        return text


_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


class MockCliChatCompletionClient(ChatCompletionClient):
    """Mock client that randomly selects one of the provided legal actions."""

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__()
        self._model_info = ModelInfo(family="mock", multiple_system_messages=True, vision=False)
        self._capabilities = ModelCapabilities(json_output=False, function_calling=False, vision=False)
        self._rng = random.Random(seed)
        self._last_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    def actual_usage(self) -> RequestUsage:
        return self._last_usage

    def total_usage(self) -> RequestUsage:
        return self._last_usage

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[Any]] = None,
        extra_create_args: Optional[dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        content = self._mock_response(messages)
        usage = RequestUsage(prompt_tokens=1, completion_tokens=1)
        self._last_usage = usage
        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=None,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[Any]] = None,
        extra_create_args: Optional[dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        async def _generator():
            content = self._mock_response(messages)
            usage = RequestUsage(prompt_tokens=1, completion_tokens=1)
            self._last_usage = usage
            yield content
            yield CreateResult(
                finish_reason="stop",
                content=content,
                usage=usage,
                cached=False,
                logprobs=None,
                thought=None,
            )

        return _generator()

    def _mock_response(self, messages: Sequence[LLMMessage]) -> str:
        prompt = ""
        for message in messages:
            if isinstance(message, UserMessage):
                prompt = _render_user_content(message.content)
        candidates = _JSON_BLOCK_RE.findall(prompt)
        payload = {}
        for raw in reversed(candidates):
            try:
                payload = json.loads(raw)
                break
            except json.JSONDecodeError:
                continue
        legal_actions = payload.get("legal_actions") if isinstance(payload, dict) else None
        if not isinstance(legal_actions, list) or not legal_actions:
            return json.dumps({"action": "wait"})
        action = self._rng.choice(legal_actions)
        kind = action.get("action")
        if kind == "move":
            direction = action.get("direction", "up")
            return json.dumps({"action": "move", "direction": direction})
        if kind == "talk":
            target = action.get("target", "ally")
            alias = action.get("target_title") or target
            message = f"Hey {alias}, let's keep moving!"
            return json.dumps({"action": "talk", "target": target, "message": message})
        return json.dumps({"action": "wait"})

    async def close(self) -> None:
        return None

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 1

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 10_000

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return self._capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info


def _parse_json(raw_output: str) -> dict[str, Any]:
    stripped = raw_output.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {}


def _payload_has_error(payload: dict[str, Any]) -> bool:
    stats = payload.get("stats")
    if isinstance(stats, dict):
        models = stats.get("models")
        if isinstance(models, dict):
            for meta in models.values():
                api = meta.get("api") if isinstance(meta, dict) else None
                if isinstance(api, dict) and api.get("totalErrors", 0):
                    return True
    if "error" in payload:
        return True
    return False


def _extract_text_from_payload(payload: dict[str, Any], fallback: str) -> str:
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        text_chunks = list(_walk_text(candidates[0]))
        if text_chunks:
            return "".join(text_chunks).strip()

    for key in ("output", "text", "response"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()

    return fallback.strip()


def _walk_text(node: Any) -> Sequence[str]:
    chunks: list[str] = []

    def _recurse(value: Any) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            for item in value:
                _recurse(item)
        elif isinstance(value, dict):
            if isinstance(value.get("text"), str):
                chunks.append(value["text"])
            if "parts" in value:
                _recurse(value["parts"])
            if "content" in value:
                _recurse(value["content"])

    _recurse(node)
    return chunks
