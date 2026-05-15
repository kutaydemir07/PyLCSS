# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
PydanticAI runner -- the modern path for the assistant.

The legacy orchestrator generates JSON plans by *prompt-engineering* the LLM
to emit tool-call dicts which we then parse and execute.  That worked, but:

- small / local models hallucinate tool names and miss required parameters;
- there's no auto-recovery on schema mismatch;
- every provider needs its own custom JSON-extract heuristics.

This module replaces that with `pydantic_ai.Agent`, which:

- speaks the **native** function-calling protocol on every supported provider
  (OpenAI, Anthropic, Google, plus any OpenAI-compatible local server like
  LM Studio / Ollama / vLLM);
- enforces strict-mode JSON schemas, so the LLM literally cannot produce
  invalid tool args;
- raises `ModelRetry` and lets the LLM auto-correct its arguments;
- lets us reuse every existing legacy `Tool` without rewriting handlers
  (see `tools/pydantic_adapter.py`).

Usage from the manager
----------------------
    runner = PydanticAgentRunner.from_legacy_registry(
        registry, provider="local", model="qwen2.5-14b-instruct",
        base_url="http://localhost:1234/v1",
    )
    answer = runner.run_sync("Add a width design variable from 10 to 100 mm.")

The legacy orchestrator stays untouched; this is opt-in via the manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic_ai import Agent

from pylcss.assistant_systems.tools.registry import Tool, ToolRegistry
from pylcss.assistant_systems.tools.pydantic_adapter import wrap_legacy_tool

logger = logging.getLogger(__name__)


# Default system prompt -- intentionally short.  PydanticAI advertises every
# tool's description in the JSON schema sent with the request, so we don't
# need to repeat tool names here.  Tone + safety + scope guidance only.
_DEFAULT_SYSTEM_PROMPT = """\
You are the PyLCSS engineering assistant, controlling a parametric CAD +
FEA + optimization desktop tool from natural-language requests.

Rules:
- Prefer calling a tool over describing what to do.
- One step at a time: small, validated tool calls beat one giant plan.
- If a tool reports an error, read the message, correct the inputs, and try
  again -- don't abandon the request.
- Ask the user to confirm only when an action is destructive or expensive
  (deleting a model, running a long FEA solve, overwriting a saved file).
- Reply in the language the user wrote in.
"""


@dataclass
class PydanticAgentResult:
    """Outcome of one ``run`` call -- mirrors the shape the legacy
    orchestrator emits so the manager can consume both transparently."""
    output: str
    tool_calls: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class PydanticAgentRunner:
    """Thin wrapper around ``pydantic_ai.Agent`` that maps onto PyLCSS's
    legacy ``ToolRegistry`` and supports OpenAI / OpenAI-compatible local
    servers.

    Anthropic and Google provider plumbing is left as a TODO -- they slot in
    by swapping the model factory below; the rest of the runner is provider
    agnostic.
    """

    def __init__(
        self,
        agent: Agent,
        tool_handlers: Dict[str, Callable[..., Any]],
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._agent = agent
        self._tool_handlers = tool_handlers
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_registry(
        cls,
        registry: ToolRegistry,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tool_filter: Optional[Sequence[str]] = None,
    ) -> "PydanticAgentRunner":
        """Build a runner from the existing tool registry.

        Parameters
        ----------
        registry : ToolRegistry
            Source of truth for tool definitions + handlers.
        provider : {"openai", "local", "anthropic", "google"}
            Which LLM backend.  ``"local"`` is shorthand for an
            OpenAI-compatible HTTP server (LM Studio, Ollama, vLLM); pass its
            URL via ``base_url``.
        model : str
            Provider-specific model id.
        api_key : str, optional
            Required for cloud providers; safely ignored for local servers
            that don't authenticate.
        base_url : str, optional
            Override the provider's default endpoint.  For ``"local"`` this
            is the only way to point at the local server (default
            ``http://localhost:1234/v1`` mirrors LM Studio's default port).
        system_prompt : str, optional
            Override the default system prompt.
        tool_filter : sequence[str], optional
            If given, only the named tools are exposed to the LLM.  Useful
            for scoped per-workflow copilots (e.g. "only CAD tools").
        """
        legacy_tools = list(registry.all_tools)
        if tool_filter is not None:
            legacy_tools = [t for t in legacy_tools if t.name in set(tool_filter)]
        if not legacy_tools:
            raise RuntimeError(
                "No tools to register with PydanticAgentRunner. "
                "Did you forget to call create_pylcss_tools(registry, dispatcher)?"
            )

        # Build the underlying chat model.  We use OpenAIChatModel for both
        # cloud OpenAI and OpenAI-compatible local servers because their
        # wire protocol is identical.
        chat_model = _build_chat_model(provider, model, api_key, base_url)

        # Pydantic-ai builds tool schemas from each callable's Pydantic
        # ``BaseModel`` automatically.  We register every tool by hand below
        # via ``agent.tool_plain``.
        agent = Agent(
            chat_model,
            system_prompt=system_prompt or _DEFAULT_SYSTEM_PROMPT,
        )

        handlers: Dict[str, Callable[..., Any]] = {}
        for tool in legacy_tools:
            try:
                args_model, call = wrap_legacy_tool(tool)
            except Exception as exc:
                logger.warning("Skipping tool %r: adapter failed (%s)", tool.name, exc)
                continue
            # tool_plain registers a no-context tool: the LLM sees a JSON
            # schema derived from the args_model, and pydantic-ai validates
            # the LLM's call against it before invoking ``call``.
            agent.tool_plain(call)
            handlers[tool.name] = call

        return cls(agent, handlers, system_prompt or _DEFAULT_SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_sync(self, prompt: str) -> PydanticAgentResult:
        """Run the agent loop to completion and return the structured result.

        Synchronous wrapper around ``Agent.run_sync`` so the existing Qt
        threading code (which already runs LLM calls on a background thread)
        can drop this in without an asyncio rework.
        """
        try:
            run = self._agent.run_sync(prompt)
        except Exception as exc:
            logger.exception("PydanticAgentRunner.run_sync failed")
            error_msg = _classify_run_error(exc)
            return PydanticAgentResult(
                output="",
                tool_calls=[],
                success=False,
                error=error_msg,
            )

        tool_calls = _extract_tool_calls(run)
        output = getattr(run, "output", None) or getattr(run, "data", None) or ""
        return PydanticAgentResult(
            output=str(output),
            tool_calls=tool_calls,
            success=True,
            error=None,
        )

    @property
    def tool_names(self) -> List[str]:
        return sorted(self._tool_handlers.keys())


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _classify_run_error(exc: BaseException) -> str:
    """Return a user-friendly error string for a ``run_sync`` failure.

    Walks the exception chain to detect network / connection issues and
    returns an actionable message instead of a raw Python traceback line.
    """
    # Walk the full cause chain looking for known connection-error types.
    current: Optional[BaseException] = exc
    while current is not None:
        name = type(current).__name__
        msg = str(current).lower()
        if name in ("ConnectError", "APIConnectionError") or "connection" in msg:
            return (
                "Could not connect to the LLM server. "
                "Please make sure your local LLM server (e.g. LM Studio, Ollama, vLLM) "
                "is running and that the base URL in the assistant settings is correct."
            )
        if name in ("AuthenticationError", "PermissionDeniedError") or "api key" in msg or "unauthorized" in msg:
            return (
                "Authentication failed. "
                "Please check that the API key in the assistant settings is correct."
            )
        if name in ("RateLimitError",) or "rate limit" in msg:
            return "Rate limit reached. Please wait a moment and try again."
        if name in ("ModelAPIError",):
            # Generic pydantic-ai wrapper -- try to surface the inner message.
            inner = getattr(current, "message", None) or str(current)
            return f"LLM error: {inner}"
        current = current.__cause__ or current.__context__

    # Fallback: include the type so developers can still diagnose.
    return f"{type(exc).__name__}: {exc}"


def _build_chat_model(
    provider: str, model: str, api_key: Optional[str], base_url: Optional[str],
):
    """Return a pydantic-ai ChatModel for the given provider.

    All four providers PyLCSS supports (OpenAI, local OpenAI-compatible,
    Anthropic, Google) get native function-calling for free; the manager
    just picks the provider via config.
    """
    provider = provider.lower()

    if provider in ("openai", "local"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        if provider == "local":
            # LM Studio's default; user can override.  api_key is irrelevant
            # for unauth'd local servers but the OpenAI client expects a
            # non-empty string, so default to a placeholder.
            base_url = base_url or "http://localhost:1234/v1"
            api_key = api_key or "lm-studio"
        prov = OpenAIProvider(api_key=api_key or "", base_url=base_url) if (api_key or base_url) else OpenAIProvider()
        return OpenAIChatModel(model, provider=prov)

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        prov = AnthropicProvider(api_key=api_key) if api_key else AnthropicProvider()
        return AnthropicModel(model, provider=prov)

    if provider in ("google", "gemini"):
        # Pydantic-ai exposes Google as either GoogleModel (Gemini API key) or
        # via Vertex; the API-key path is what end-users typically have.
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        prov = GoogleProvider(api_key=api_key) if api_key else GoogleProvider()
        return GoogleModel(model, provider=prov)

    raise ValueError(f"Unknown provider {provider!r}")


def _extract_tool_calls(run: Any) -> List[Dict[str, Any]]:
    """Pull a flat list of tool invocations out of a pydantic-ai run result.

    The shape of ``run.all_messages()`` evolved across pydantic-ai 1.x
    minor versions -- this helper is intentionally permissive so a minor
    bump doesn't break the manager.
    """
    calls: List[Dict[str, Any]] = []
    try:
        messages = run.all_messages()
    except Exception:
        return calls
    for msg in messages:
        for part in getattr(msg, "parts", ()) or ():
            # ToolCallPart is the canonical name in 1.x.
            if part.__class__.__name__ in ("ToolCallPart", "ToolCall"):
                calls.append({
                    "name": getattr(part, "tool_name", None) or getattr(part, "name", None),
                    "args": getattr(part, "args", None) or getattr(part, "args_dict", None) or {},
                    "tool_call_id": getattr(part, "tool_call_id", None),
                })
    return calls
