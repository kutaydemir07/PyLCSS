# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""PydanticAI-backed execution loop for the PyLCSS assistant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Sequence

from pydantic_ai import Agent, DeferredToolRequests

from pylcss.assistant_systems.tools.pydantic_adapter import build_pydantic_tool
from pylcss.assistant_systems.tools.tool_types import ToolRegistry

logger = logging.getLogger(__name__)


# Default system prompt -- intentionally short.  PydanticAI advertises every
# tool's description in the JSON schema sent with the request, so we don't
# need to repeat tool names here.  Tone + safety + scope guidance only.
#
_DEFAULT_SYSTEM_PROMPT = """\
You are the PyLCSS engineering assistant. You control a parametric CAD +
FEA + optimization desktop tool by calling its tools.

How to act:
- Read the whole user request, then **decompose it into every tool call
  needed to satisfy it**. Do not stop after one tool when the user asked
  for more.
- Treat tools as sequential because they share the active window and graph.
  Do not assume that two UI or graph mutations can safely run in parallel.
- Prefer one complete graph-creation call over many partial mutations. When
  a later call depends on an earlier result, inspect that result first.
- Prefer calling a tool over describing what you would do.
- If a tool reports an error, read the message, correct the inputs, and
  retry. Do not abandon the request after one failure.
- Ask the user to confirm only before destructive or expensive actions
  (deleting a model, running a long FEA solve, overwriting a saved file).
- Reply in the same language the user wrote in.
- When finished, give a short summary of what you actually did, not a
  rehearsal of what you would do.

CAD pipeline rules (this is the most common workload):
- Prefer the GUI-native `com.cad.geometry.*` nodes for boxes, cylinders,
  tubes, cylindrical shells, booleans, holes, fillets, transforms, and
  linear patterns. Their engineering dimensions remain understandable
  in the Inspector.
- Use `com.cad.code_part` only for geometry that native nodes cannot
  express. Expose every tuneable code-part dimension through its
  `parameters` property (`name=value` lines).
- Before changing an existing graph, inspect it with
  `get_design_studio_state`. Validate complex new graph JSON before
  creating it.
- After modifying geometry, call `execute_cad` so the 3-D viewport
  refreshes -- otherwise the user sees stale results.
- If several solver terminals exist, pass the exact `terminal_node` to
  `execute_cad`; never run unrelated FEA, crash, or topology workflows.
"""


@dataclass
class PydanticAgentResult:
    """Outcome of one ``run`` call -- mirrors the shape the legacy
    orchestrator emits so the manager can consume both transparently."""

    output: str
    tool_calls: list[dict[str, Any]]
    success: bool
    error: str | None = None


class PydanticAgentRunner:
    """Run validated PyLCSS tools through a provider-agnostic agent."""

    def __init__(
        self,
        agent: Agent,
        tool_handlers: dict[str, Callable[..., Any]],
    ) -> None:
        self._agent = agent
        self._tool_handlers = tool_handlers

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_registry(
        cls,
        registry: ToolRegistry,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        tool_filter: Sequence[str] | None = None,
        auto_approve_confirmation: bool = False,
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
            requested_tools = set(tool_filter)
            legacy_tools = [t for t in legacy_tools if t.name in requested_tools]
        if not legacy_tools:
            raise RuntimeError(
                "No tools to register with PydanticAgentRunner. "
                "Did you forget to call create_pylcss_tools(registry, dispatcher)?"
            )

        # Build the underlying chat model.  We use OpenAIChatModel for both
        # cloud OpenAI and OpenAI-compatible local servers because their
        # wire protocol is identical.
        chat_model = _build_chat_model(provider, model, api_key, base_url)

        pydantic_tools = []
        handlers: dict[str, Callable[..., Any]] = {}
        for tool in legacy_tools:
            try:
                registered_tool, call = build_pydantic_tool(
                    tool,
                    auto_approve_confirmation=auto_approve_confirmation,
                )
            except Exception as exc:
                logger.warning("Skipping tool %r: adapter failed (%s)", tool.name, exc)
                continue
            pydantic_tools.append(registered_tool)
            handlers[tool.name] = call

        if not pydantic_tools:
            raise RuntimeError("No valid tools could be registered with PydanticAI")

        agent = Agent(
            chat_model,
            system_prompt=system_prompt or _DEFAULT_SYSTEM_PROMPT,
            tools=pydantic_tools,
        )

        return cls(agent, handlers)

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
        if isinstance(output, DeferredToolRequests):
            action_names = [
                call.tool_name for call in output.approvals if call.tool_name
            ]
            requested = ", ".join(action_names) or "the requested action"
            return PydanticAgentResult(
                output="",
                tool_calls=tool_calls,
                success=False,
                error=(
                    f"Confirmation required for {requested}. "
                    "Enable auto-execute in Assistant Settings to allow it."
                ),
            )
        return PydanticAgentResult(
            output=str(output),
            tool_calls=tool_calls,
            success=True,
            error=None,
        )

    @property
    def tool_names(self) -> list[str]:
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
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__
        msg = str(current).lower()
        if name in ("ConnectError", "APIConnectionError") or "connection" in msg:
            return (
                "Could not connect to the LLM server. "
                "Please make sure your local LLM server (e.g. LM Studio, Ollama, vLLM) "
                "is running and that the base URL in the assistant settings is correct."
            )
        if (
            name in ("AuthenticationError", "PermissionDeniedError")
            or "api key" in msg
            or "unauthorized" in msg
        ):
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
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> Any:
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
        prov = (
            OpenAIProvider(api_key=api_key or "", base_url=base_url)
            if (api_key or base_url)
            else OpenAIProvider()
        )
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


def _extract_tool_calls(run: Any) -> list[dict[str, Any]]:
    """Pull a flat list of tool invocations out of a pydantic-ai run result.

    The shape of ``run.all_messages()`` evolved across pydantic-ai 1.x
    minor versions -- this helper is intentionally permissive so a minor
    bump doesn't break the manager.
    """
    calls: list[dict[str, Any]] = []
    try:
        messages = run.all_messages()
    except Exception:
        return calls
    for msg in messages:
        for part in getattr(msg, "parts", ()) or ():
            # ToolCallPart is the canonical name in 1.x.
            if part.__class__.__name__ in ("ToolCallPart", "ToolCall"):
                calls.append(
                    {
                        "name": getattr(part, "tool_name", None)
                        or getattr(part, "name", None),
                        "args": getattr(part, "args", None)
                        or getattr(part, "args_dict", None)
                        or {},
                        "tool_call_id": getattr(part, "tool_call_id", None),
                    }
                )
    return calls
