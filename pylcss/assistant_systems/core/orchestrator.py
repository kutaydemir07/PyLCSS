# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hierarchical Agent Orchestrator — agents that spawn agents.

Architecture
============
Complex requests are decomposed by a lightweight **Supervisor** into
domain-specific sub-tasks.  Each sub-task is planned by a focused
specialist (CAD / Modeling / Analysis / General) whose prompt contains
only the tools and schema for its domain, keeping the context window
small enough for GPT-4o-class models to produce reliable plans.

Flow::

    User Request
        │
        ▼
    Supervisor  ── classify & decompose
        │
    ┌── CAD Planner ─────┐
    ├── Modeling Planner ─┤ → Executor ↔ Critic → Result
    ├── Analysis Planner ─┤
    └── General Planner ──┘
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

from .agent import (
    BaseAgent, AgentRole, ExecutionPlan, CritiqueResult,
    ToolCall, ToolResult, ReActLoop, AgentStep,
)
from pylcss.assistant_systems.tools.registry import (
    Tool, ToolRegistry, get_cad_schema_for_prompt, get_modeling_schema_for_prompt,
)

if TYPE_CHECKING:
    from pylcss.assistant_systems.services.llm import LLMProvider

logger = logging.getLogger(__name__)


# ===================================================================
#  Prompts — deliberately compact for smaller-model reliability
# ===================================================================

# -- Supervisor (NOT templated — uses single braces in JSON examples) --

SUPERVISOR_PROMPT = '''You are the Supervisor for PyLCSS.
Classify the request and decompose only when needed.

Domains:
- cad: geometry creation or CAD graph edits
- modeling: inputs, outputs, equations, variables, functions, validation, model build
- analysis: sensitivity, surrogate, optimization, DOE
- navigation: tab switching
- project: save, load, new project
- conversation: questions or explanations only

Complexity:
- simple: 1 action
- compound: 2-4 related actions in one domain
- complex: multiple domains, dependencies, or 5+ actions

Rules:
1. Classify by intent, not only verbs.
2. Modeling requests mention inputs, outputs, variables, equations, functions, systems, or validation.
3. CAD requests mention shapes, parts, assemblies, booleans, transforms, patterns, or 3D objects.
4. Simple and compound requests usually return one sub_task.
5. Add navigation sub_tasks only if a tab switch is required.
6. Do not return discussion-only sub_tasks.

Respond with ONLY JSON:
```json
{
    "domain": "<domain>",
    "complexity": "simple|compound|complex",
    "sub_tasks": [
        {"domain": "...", "task": "Detailed instruction for the specialist", "context": ""}
    ],
    "reasoning": "brief"
}
```'''


# -- Domain planners (templated via format_map — use double braces for literal JSON) --

CAD_PLANNER_PROMPT = '''You are the CAD Specialist for PyLCSS.
Create a plan using only the tools and node types below.

Tools
{cad_tools}

Node Types
{cad_schema}

Current Graph
{current_state}

Rules:
1. Build the requested part with one `create_cad_geometry` call whenever practical.
2. Always finish with `execute_cad`.
3. Use only listed node types and exact property names.
4. Do not add the `com.cad.` prefix.
5. Connections use `{{"from": "id.port", "to": "id.port"}}`.
6. Boolean inputs are `shape_a` and `shape_b`; most other modifiers use `shape`.
7. Reuse existing node IDs when modifying current geometry.
8. Connect directly to prior part IDs when they are available; do not invent proxy nodes.
9. Keep the graph simple and practical; prefer primitives, booleans, transforms, patterns, and sketch+extrude.
10. For FEA or topology optimization, use the exact simulation nodes from the schema: `sim.mesh`, `sim.material`, `sim.constraint`, `sim.load`, `sim.solver`, `sim.topopt`.
11. Do not invent unsupported names like `fixed_constraint`, `force_load`, or other non-schema simulation nodes.
12. For face-based FEA constraints and loads, connect `select_face.workplane` to `sim.constraint.target_face` or `sim.load.target_face`.

Common patterns:
- box with hole: box + cylinder + boolean Cut
- tube: outer cylinder + inner cylinder + boolean Cut
- flange: disc + bore cut + hole pattern
- bracket/profile: sketch + polyline + extrude + optional fillet
- gear: hub + one tooth + union + circular pattern
- topology optimization beam: box -> sim.mesh, box -> select_face, select_face.workplane -> sim.constraint.target_face, sim.mesh -> sim.constraint.mesh, sim.mesh/material/constraints/loads -> sim.topopt

Respond with ONLY JSON:
```json
{{
    "goal": "...",
    "reasoning": "shape, approach, key dimensions in mm",
    "steps": [
        {{"tool": "create_cad_geometry", "params": {{"nodes": [...], "connections": [...]}}, "description": "...", "expected": "..."}},
        {{"tool": "execute_cad", "params": {{}}, "description": "Render", "expected": "Shape in viewer"}}
    ],
    "complexity": 1
}}
```'''


MODELING_PLANNER_PROMPT = '''You are the Modeling Specialist for PyLCSS.
Create a plan using only the tools and node types below.

Tools
{modeling_tools}

System Node Types
{modeling_schema}

Current Graph
{current_state}

Rules:
1. Use `create_system_model` to create or update all nodes and connections in one call when possible.
2. `com.pfd.input`: set `var_name`, `min`, `max`, optional `unit`.
3. `com.pfd.output`: set `var_name`, optional `unit`, `req_min`, `req_max`, `minimize`, `maximize`.
4. `com.pfd.intermediate`: set `var_name`, optional `unit`.
5. `com.pfd.custom_block`: set `num_inputs`, `num_outputs`, `code_content`.
6. Input/output/intermediate ports use `var_name`; custom blocks use `in_1..in_n` and `out_1..out_n`.
7. Connections use `{{"from": "nodeId.port", "to": "nodeId.port"}}`.
8. After creation, call `validate_model`.
9. Add `build_model` only if the task explicitly needs analysis transfer.

Respond with ONLY JSON:
```json
{{
    "goal": "...",
    "reasoning": "brief",
    "steps": [
        {{"tool": "create_system_model", "params": {{"nodes": [...], "connections": [...]}}, "description": "...", "expected": "..."}},
        {{"tool": "validate_model", "params": {{}}, "description": "Validate", "expected": "No errors"}}
    ],
    "complexity": 1
}}
```'''


ANALYSIS_PLANNER_PROMPT = '''You are the **Analysis Specialist** for PyLCSS.
Create a plan using ONLY the tools listed below.

## Tools
{analysis_tools}

## Current State
{current_state}

## Typical Workflows
- **Sensitivity**: switch_tab → sensitivity, then run_sensitivity_analysis
- **Optimization**: switch_tab → optimization, then run_optimization
- **Surrogate**: switch_tab → surrogate, then train_surrogate
- **DOE Sampling**: switch_tab → solution_space, then generate_samples
- Ensure the model is built first (build_model from modeling tab) if not already done.

Respond with ONLY JSON:
```json
{{
  "goal": "...",
  "reasoning": "...",
  "steps": [{{"tool": "...", "params": {{}}, "description": "...", "expected": "..."}}],
  "complexity": 1
}}
```'''


GENERAL_PLANNER_PROMPT = '''You are a General Assistant for PyLCSS.
Create a plan using the tools below.

## Tools
{tools}

## Tab Names for switch_tab
modeling, cad, surrogate, solution_space, optimization, sensitivity

Respond with ONLY JSON:
```json
{{
  "goal": "...",
  "reasoning": "...",
  "steps": [{{"tool": "...", "params": {{}}, "description": "...", "expected": "..."}}],
  "complexity": 1
}}
```'''


ASSEMBLY_PLANNER_PROMPT = '''You are the Assembly Specialist for PyLCSS CAD.
Plan assembly and arrangement work using the tools and node types below.

Tools
{cad_tools}

Node Types
{cad_schema}

Current Graph
{current_state}

Rules:
1. Prefer transforms, mirror/pattern tools, and boolean union over rebuilding finished parts.
2. Reuse existing part IDs from the current graph whenever possible.
3. Create only the minimum extra helper nodes needed for placement or combination.
4. For multipart requests, focus on positioning, alignment, symmetry, and final combination.
5. Always finish with `execute_cad`.

Respond with ONLY JSON:
```json
{{
    "goal": "...",
    "reasoning": "brief",
    "steps": [
        {{"tool": "create_cad_geometry", "params": {{"nodes": [...], "connections": [...]}}, "description": "...", "expected": "..."}},
        {{"tool": "execute_cad", "params": {{}}, "description": "Render", "expected": "Assembly in viewer"}}
    ],
    "complexity": 1
}}
```'''


SYSTEM_MODEL_AGENT_PROMPT = '''You are the System Model Specialist for PyLCSS.
Plan complete modeling workflows using the tools and node types below.

Tools
{modeling_tools}

System Node Types
{modeling_schema}

Current Graph
{current_state}

Rules:
1. Build coherent system models with variables, equations, constraints, objectives, and units.
2. Prefer one `create_system_model` call containing the full local graph update.
3. Use `com.pfd.custom_block` for equations or transformations.
4. Set objective and requirement properties explicitly on outputs when relevant.
5. Always include `validate_model`; include `build_model` only when analysis transfer is requested.

Respond with ONLY JSON:
```json
{{
    "goal": "...",
    "reasoning": "brief",
    "steps": [
        {{"tool": "create_system_model", "params": {{"nodes": [...], "connections": [...]}}, "description": "...", "expected": "..."}},
        {{"tool": "validate_model", "params": {{}}, "description": "Validate", "expected": "No errors"}}
    ],
    "complexity": 1
}}
```'''


GRAPH_EDIT_PLANNER_PROMPT = '''You are the Graph Edit Specialist for PyLCSS.
Modify existing CAD or modeling graphs using the tools below.

Tools
{tools}

Current Graph
{current_state}

Rules:
1. Prefer targeted edits over rebuilding the entire graph.
2. Reuse existing node IDs from the current graph state.
3. For CAD edits, prefer `modify_cad_node`, `connect_cad_nodes`, or a minimal `create_cad_geometry` patch.
4. For modeling edits, prefer `modify_system_node` or a minimal `create_system_model` patch.
5. Only create new nodes when the requested change cannot be expressed as a property or connection edit.
6. If execution/validation is needed after the edit, include the appropriate follow-up tool.

Respond with ONLY JSON:
```json
{{
    "goal": "...",
    "reasoning": "brief",
    "steps": [
        {{"tool": "...", "params": {{}}, "description": "...", "expected": "..."}}
    ],
    "complexity": 1
}}
```'''


# -- Executor recovery (templated via .format()) --

EXECUTOR_PROMPT = '''A tool failed. Diagnose the problem and suggest a recovery action.

Tool: {tool_name}
Error: {error}
Params: {parameters}
State: {graph_state}

Common fixes:
- Unknown tool → use a real tool name from the available list
- Wrong node type → check the schema; build complex objects from primitives
- Wrong property name → use exact names (box_length not length)
- Boolean fail → shapes must intersect
- Geometry error → simplify dimensions

Respond with ONLY JSON:
```json
{{
  "diagnosis": "...",
  "should_retry": true,
  "recovery": {{"tool": "...", "params": {{}}, "description": "..."}}
}}
```'''


# -- Critic (templated via .format()) --

CRITIC_PROMPT = '''Evaluate this execution result.

Step {step_num}/{total_steps} | Tool: {tool_name} | Success: {success}
Output: {output}
Expected: {expected}
State: {observation}
User Goal: {user_request}

Rules:
- APPROVE if the tool returned success=True and no error occurred.
- APPROVE intermediate steps (step < total) as long as the tool succeeded.
- REJECT only on HARD failures: tool returned an error, tool not found, or crash.
- Do NOT reject based on aesthetic or geometric judgment — you cannot see the 3D result.
- Do NOT reject execute_cad if it succeeded — the geometry is rendered in the viewer.
- If the graph was created with all nodes and connections, APPROVE it.

Respond with ONLY JSON:
```json
{{"approved": true, "issues": [], "suggestions": [], "design_feedback": "", "severity": 1}}
```
Severity: 1=cosmetic, 2=suboptimal, 3=functionality issue, 4=significant error, 5=critical'''


# ===================================================================
#  Data classes
# ===================================================================

@dataclass
class SubTask:
    """A single sub-task produced by the Supervisor."""
    domain: str
    task: str
    context: str = ""


@dataclass
class TaskClassification:
    """Supervisor output: domain, complexity, and ordered sub-tasks."""
    domain: str
    complexity: str
    sub_tasks: List[SubTask]
    reasoning: str = ""


@dataclass
class AgentSessionTelemetry:
    """Small in-memory telemetry store for planner, tool, and verifier outcomes."""
    request_count: int = 0
    planner_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    verifier_rule_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def start_request(self) -> None:
        self.request_count += 1

    def record_plan(self, planner_name: str, step_count: int) -> None:
        stats = self.planner_stats.setdefault(planner_name, {"plans": 0, "steps": 0})
        stats["plans"] += 1
        stats["steps"] += step_count

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        stats = self.tool_stats.setdefault(tool_name, {"calls": 0, "successes": 0, "failures": 0})
        stats["calls"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    def record_verifier_result(self, target_tool: str, output: Dict[str, Any]) -> None:
        issues = output.get("issues", []) or []
        repairs = output.get("applied_repairs", []) or []
        resolved = bool(output.get("ok", False))
        for issue in issues:
            rule_name = self._rule_name(issue)
            stats = self.verifier_rule_stats.setdefault(rule_name, {"hits": 0, "resolved": 0, "failed": 0, "repairs": 0})
            stats["hits"] += 1
            if resolved:
                stats["resolved"] += 1
            else:
                stats["failed"] += 1
            stats["repairs"] += len(repairs)

        if not issues and repairs:
            rule_name = f"repaired:{target_tool}"
            stats = self.verifier_rule_stats.setdefault(rule_name, {"hits": 0, "resolved": 0, "failed": 0, "repairs": 0})
            stats["hits"] += 1
            stats["resolved"] += 1
            stats["repairs"] += len(repairs)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "requests": self.request_count,
            "planners": self.planner_stats,
            "tools": self.tool_stats,
            "verifier_rules": self.verifier_rule_stats,
        }

    def compact_summary(self) -> str:
        best_planner = self._top_entry(self.planner_stats, key="plans")
        busiest_tool = self._top_entry(self.tool_stats, key="calls")
        hottest_rule = self._top_entry(self.verifier_rule_stats, key="hits")
        parts = [f"requests={self.request_count}"]
        if best_planner:
            parts.append(f"planner={best_planner[0]}:{best_planner[1].get('plans', 0)}")
        if busiest_tool:
            parts.append(f"tool={busiest_tool[0]}:{busiest_tool[1].get('successes', 0)}/{busiest_tool[1].get('calls', 0)}")
        if hottest_rule:
            parts.append(f"rule={hottest_rule[0]}:{hottest_rule[1].get('resolved', 0)}/{hottest_rule[1].get('hits', 0)}")
        return " | ".join(parts)

    @staticmethod
    def _rule_name(issue: str) -> str:
        if " — " in issue:
            return issue.split(" — ", 1)[0][:80]
        if ":" in issue:
            return issue.split(":", 1)[0][:80]
        return issue[:80]

    @staticmethod
    def _top_entry(stats: Dict[str, Dict[str, int]], key: str) -> Optional[tuple]:
        if not stats:
            return None
        return max(stats.items(), key=lambda item: item[1].get(key, 0))


# ===================================================================
#  Supervisor Agent
# ===================================================================

class SupervisorAgent(BaseAgent):
    """Classifies requests by domain/complexity and decomposes complex ones."""

    role = AgentRole.SUPERVISOR

    def process(self, context: Dict[str, Any]) -> TaskClassification:
        request = context.get("user_request", "")
        response = self._call_llm_sync(request, SUPERVISOR_PROMPT)
        return self._parse(response, request)

    def _parse(self, response: str, fallback_request: str) -> TaskClassification:
        data = self._extract_json(response)
        if data and "sub_tasks" in data and data["sub_tasks"]:
            subs = [
                SubTask(
                    domain=s.get("domain", "cad"),
                    task=s.get("task", fallback_request),
                    context=s.get("context", ""),
                )
                for s in data["sub_tasks"]
            ]
            return TaskClassification(
                domain=data.get("domain", subs[0].domain),
                complexity=data.get("complexity", "compound"),
                sub_tasks=subs,
                reasoning=data.get("reasoning", ""),
            )

        # Fallback: treat entire request as single compound task.
        # Guess domain from keywords.
        domain = self._guess_domain(fallback_request)
        return TaskClassification(
            domain=domain,
            complexity="compound",
            sub_tasks=[SubTask(domain=domain, task=fallback_request)],
            reasoning="Supervisor parse failed — fallback keyword detection",
        )

    @staticmethod
    def _guess_domain(text: str) -> str:
        """Keyword heuristic used only when the LLM response is unparseable."""
        t = text.lower()
        if any(w in t for w in ("optim", "sensitiv", "surrogate", "doe", "sampling")):
            return "analysis"
        if any(w in t for w in ("variable", "input", "output", "function", "model",
                                "system", "equation")):
            return "modeling"
        # Default to CAD — the most common domain
        return "cad"


# ===================================================================
#  Domain Planner Agent (one class, configured per domain)
# ===================================================================

class DomainPlannerAgent(BaseAgent):
    """
    Plans tool calls for a specific engineering domain.

    A single class is instantiated once per domain, each with its own
    prompt template, tool-category filter, and optional node-schema
    function.  This keeps every planning prompt small and focused.
    """

    role = AgentRole.PLANNER

    def __init__(
        self,
        llm_provider: 'LLMProvider',
        tool_registry: ToolRegistry,
        prompt_template: str,
        tool_categories: List[str],
        schema_func: Optional[Callable[[], str]] = None,
    ):
        super().__init__(llm_provider)
        self.tools = tool_registry
        self.prompt_template = prompt_template
        self.tool_categories = tool_categories
        self.schema_func = schema_func

    def process(self, context: Dict[str, Any]) -> ExecutionPlan:
        task = context.get("task_description", context.get("user_request", ""))
        raw_state = context.get("graph_state", "Empty graph")

        # Compress graph state to prevent LLM from parroting the verbose
        # internal format (type_, name, visible, subgraph_session, etc.)
        # and to save tokens — later sub-parts have huge state.
        state = self._compress_state(raw_state)

        # Build the domain-specific prompt.
        # All possible template keys are populated so that any prompt
        # template can be used without KeyError.
        tools_desc = self.tools.get_tools_description(categories=self.tool_categories)
        schema = self.schema_func() if self.schema_func else ""

        fmt = {
            "tools": tools_desc,
            "cad_tools": tools_desc,
            "modeling_tools": tools_desc,
            "analysis_tools": tools_desc,
            "current_state": state,
            "cad_schema": schema,
            "modeling_schema": schema,
        }

        prompt = self.prompt_template.format_map(fmt)
        recent_feedback = self._format_recent_feedback(context.get("recent_planner_feedback", []))
        if recent_feedback:
            prompt += (
                "\n\nRecent Verifier Feedback From This Session\n"
                + recent_feedback
                + "\nRules:\n"
                + "- Avoid repeating these verifier failures.\n"
                + "- Prefer the patterns that were repaired or sanitized successfully."
            )

        logger.info(f"DomainPlanner{self.tool_categories}: planning '{task[:60]}…'")
        response = self._call_llm_sync(task, prompt, max_tokens=4096)
        plan = self._parse_plan(response)
        if plan.steps:
            return plan

        retry_prompt = (
            prompt
            + "\n\nYour previous response was truncated or unparseable. "
            + "Respond again with compact JSON only, no markdown fences, and keep it under 1200 characters. "
            + "Return at most 4 steps."
        )
        logger.warning(f"DomainPlanner{self.tool_categories}: retrying with compact JSON output")
        retry_response = self._call_llm_sync(task, retry_prompt, max_tokens=1400)
        retry_plan = self._parse_plan(retry_response)
        return retry_plan if retry_plan.steps else plan

    # ------------------------------------------------------------------

    @staticmethod
    def _compress_state(raw_state) -> str:
        """Compress graph state to just node IDs + short types.

        The full state includes internal properties (type_, name, visible,
        layout_direction, subgraph_session, …) which causes the LLM to
        parrot them and wastes thousands of context tokens.  For planning
        purposes the LLM only needs to know what nodes exist and their IDs.
        """
        if not raw_state or raw_state == "Empty graph":
            return "Empty graph"
        if isinstance(raw_state, str):
            try:
                raw_state = json.loads(raw_state)
            except Exception:
                return raw_state[:500] if len(raw_state) > 500 else raw_state
        if not isinstance(raw_state, dict):
            return str(raw_state)[:500]

        nodes = raw_state.get("nodes", [])
        if not nodes:
            return "Empty graph"

        # Build a compact summary: "12 nodes: p1_body(box), p1_sk(sketch), …"
        summaries = []
        for n in nodes:
            nid = n.get("id", "?")
            ntype = n.get("type", "?")
            # Strip com.cad. prefix and .FooNode suffix for brevity
            short = ntype.replace("com.cad.", "")
            import re as _re
            short = _re.sub(r'\.[A-Z][a-zA-Z]*Node$', '', short)
            summaries.append(f"{nid}({short})")

        count = raw_state.get("node_count", len(nodes))
        return f"{count} nodes: {', '.join(summaries)}"

    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse an ExecutionPlan from the LLM's JSON response."""
        data = self._extract_json(response)
        if data:
            try:
                steps = [
                    ToolCall(
                        tool_name=s.get("tool", ""),
                        parameters=s.get("params", {}),
                        expected_outcome=s.get("expected", ""),
                        description=s.get("description", ""),
                    )
                    for s in data.get("steps", [])
                ]
                return ExecutionPlan(
                    goal=data.get("goal", ""),
                    steps=steps,
                    reasoning=data.get("reasoning", ""),
                    estimated_complexity=data.get("complexity", 1),
                    requires_confirmation=data.get("requires_confirmation", False),
                )
            except Exception as e:
                logger.error(f"DomainPlanner: parse error: {e}")

        safe = response.encode("ascii", errors="replace").decode("ascii")
        logger.error(f"DomainPlanner: no valid JSON — {safe[:300]}")
        return ExecutionPlan(goal="", steps=[], reasoning=f"Parse failed: {safe[:200]}")

    @staticmethod
    def _format_recent_feedback(entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""

        lines = []
        for entry in entries[-4:]:
            issues = "; ".join(entry.get("issues", [])[:2]) or "unknown issue"
            repairs = ", ".join(entry.get("applied_repairs", [])[:2]) or "none"
            target_tool = entry.get("target_tool", "tool")
            planner = entry.get("planner", "planner")
            lines.append(f"- {planner} on {target_tool}: issues={issues}; repairs={repairs}")
        return "\n".join(lines)


class AssemblyAgent(DomainPlannerAgent):
    """Specialized CAD planner for assemblies, placement, and combination work."""

    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(
            llm_provider,
            tool_registry,
            ASSEMBLY_PLANNER_PROMPT,
            tool_categories=["cad", "navigation"],
            schema_func=get_cad_schema_for_prompt,
        )


class SystemModelAgent(DomainPlannerAgent):
    """Specialized planner for richer system-model construction requests."""

    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(
            llm_provider,
            tool_registry,
            SYSTEM_MODEL_AGENT_PROMPT,
            tool_categories=["modeling", "navigation"],
            schema_func=get_modeling_schema_for_prompt,
        )


class GraphEditAgent(DomainPlannerAgent):
    """Specialized planner for editing existing graphs with minimal changes."""

    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(
            llm_provider,
            tool_registry,
            GRAPH_EDIT_PLANNER_PROMPT,
            tool_categories=["cad", "modeling", "navigation"],
        )


# ===================================================================
#  Decomposer Agent — "Think first, then act"
# ===================================================================

DECOMPOSER_PROMPT = '''You are a Geometry Decomposition Engineer for PyLCSS CAD.
Break complex geometry into simple ordered sub-parts.

Rules:
1. Describe parts in engineering terms, not node JSON.
2. Keep each part simple enough for one `create_cad_geometry` call.
3. Include realistic dimensions in millimeters.
4. Order parts by dependency.
5. The last part should be an assembly step when the request is multipart.
6. Use 1 part for simple requests, 2-5 for most compound requests, up to 8 only for large assemblies.

Respond with ONLY JSON:
```json
{
    "analysis": "brief",
    "sub_parts": [
        {"description": "Part description with dimensions and positioning", "depends_on": []},
        {"description": "Assembly or dependent part", "depends_on": [1]}
    ]
}
```'''


@dataclass
class DecomposedPart:
    """A single sub-part from the decomposer."""
    description: str
    depends_on: List[int] = field(default_factory=list)


class DecomposerAgent(BaseAgent):
    """Breaks complex CAD requests into ordered, buildable sub-parts.

    Sits between the Supervisor and the Domain Planners.  For any CAD
    request that is non-trivial it asks the LLM to think about the
    geometry first — what parts are needed, in what order, with what
    dimensions — before any tool calls are planned.

    Each sub-part description is then fed individually to the CAD
    Planner, keeping the planning context small and focused.

    For simple requests (single primitive, basic boolean) it returns
    None and the CAD Planner handles them directly.

    Architecture::

        DecomposerAgent
            ├── simple request → None → CAD Planner handles directly
            └── complex request → [SubPart1, SubPart2, ...] → each → CAD Planner
    """

    role = AgentRole.PLANNER

    # Requests with these keywords are likely simple enough to skip decomposition
    SIMPLE_KEYWORDS = frozenset({
        "box", "cube", "cylinder", "sphere", "cone",
    })
    NON_GEOMETRIC_KEYWORDS = frozenset({
        "workflow", "optimization", "topology", "topopt", "mesh",
        "constraint", "constraints", "load", "loads", "solver",
        "stress", "fea", "simulation", "analysis",
    })
    SINGLE_PART_FEATURE_KEYWORDS = frozenset({
        "hole", "holes", "bore", "bores", "drill", "drilled",
        "cutout", "cutouts", "pocket", "pockets", "slot", "slots",
    })
    SINGLE_PART_BASE_KEYWORDS = frozenset({
        "box", "plate", "block", "flange", "bracket", "beam",
    })

    def process(self, context: Dict[str, Any]) -> Optional[List[DecomposedPart]]:
        """Analyze a CAD request and optionally decompose it.

        Returns:
            List[DecomposedPart] if the request should be decomposed,
            None if the CAD planner should handle it directly.
        """
        task = context.get("task_description", context.get("user_request", ""))
        task_lower = task.lower()

        if any(keyword in task_lower for keyword in self.NON_GEOMETRIC_KEYWORDS):
            logger.info(f"Decomposer: non-geometry workflow detected, skipping → '{task[:60]}'")
            return None

        if (
            any(keyword in task_lower for keyword in self.SINGLE_PART_FEATURE_KEYWORDS)
            and any(keyword in task_lower for keyword in self.SINGLE_PART_BASE_KEYWORDS)
            and not any(word in task_lower for word in ("assembly", "assemble", "multipart", "gearbox"))
        ):
            logger.info(f"Decomposer: single-part feature workflow detected, skipping → '{task[:60]}'")
            return None

        # Simple primitives never need decomposition
        words = set(task_lower.split())
        if (
            len(task_lower) < 40
            and any(kw in words for kw in self.SIMPLE_KEYWORDS)
            and not any(w in task_lower for w in ("with", "and", "pattern", "assembly"))
        ):
            logger.info(f"Decomposer: simple request, skipping → '{task[:50]}'")
            return None

        logger.info(f"Decomposer: analyzing '{task[:60]}…'")
        response = self._call_llm_sync(task, DECOMPOSER_PROMPT, max_tokens=2048)
        return self._parse(response)

    def _parse(self, response: str) -> Optional[List[DecomposedPart]]:
        data = self._extract_json(response)
        if not data or "sub_parts" not in data:
            logger.warning("Decomposer: no valid JSON — falling through")
            return None

        parts = data.get("sub_parts", [])
        if len(parts) <= 1:
            # Single part = no decomposition needed
            logger.info("Decomposer: single-part response — skipping decomposition")
            return None

        result = [
            DecomposedPart(
                description=p.get("description", ""),
                depends_on=p.get("depends_on", []),
            )
            for p in parts
            if p.get("description")
        ]

        analysis = data.get("analysis", "")
        logger.info(
            f"Decomposer: {len(result)} sub-parts — {analysis[:80]}"
        )
        return result if len(result) > 1 else None


class PlanVerifierAgent:
    """Deterministically injects JSON verification steps into execution plans.

    This keeps verifier behavior reliable and cheap: graph JSON is checked
    before execution without spending another LLM call.
    """

    role = AgentRole.VERIFIER

    _VERIFY_TOOL_BY_TARGET = {
        "create_cad_geometry": "verify_cad_graph_json",
        "create_system_model": "verify_system_graph_json",
    }

    def process(self, plan: ExecutionPlan) -> ExecutionPlan:
        if not plan.steps:
            return plan

        verified_steps: List[ToolCall] = []
        for step in plan.steps:
            verify_tool = self._VERIFY_TOOL_BY_TARGET.get(step.tool_name)
            if verify_tool:
                verify_params = dict(step.parameters)
                verify_params.setdefault("goal", plan.goal)
                verify_params.setdefault("target_tool", step.tool_name)
                verified_steps.append(
                    ToolCall(
                        tool_name=verify_tool,
                        parameters=verify_params,
                        description=f"Verify JSON for {step.tool_name}",
                        expected_outcome="Verifier reports no structural or semantic issues",
                    )
                )
            verified_steps.append(step)

        return ExecutionPlan(
            goal=plan.goal,
            steps=verified_steps,
            reasoning=plan.reasoning,
            estimated_complexity=plan.estimated_complexity,
            requires_confirmation=plan.requires_confirmation,
        )


# ===================================================================
#  Executor Agent
# ===================================================================

class ExecutorAgent(BaseAgent):
    """Runs tools and handles failures with LLM-assisted recovery."""

    role = AgentRole.EXECUTOR

    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(llm_provider)
        self.tools = tool_registry

    def process(self, context: Dict[str, Any]) -> ToolResult:
        tc = context.get("tool_call")
        return self.execute_tool(tc, context) if tc else ToolResult(success=False, error="No tool_call")

    def execute_tool(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        tool = self.tools.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown tool: {tool_call.tool_name}. "
                    f"Available: {[t.name for t in self.tools.all_tools]}"
                ),
            )
        try:
            if tool.validator and not tool.validator(tool_call.parameters):
                return ToolResult(success=False, error=f"Validation failed for {tool_call.tool_name}")

            start = time.time()
            logger.info(f"Executor: {tool_call.tool_name}({tool_call.parameters})")
            output = tool.handler(tool_call.parameters)

            get_state = context.get("get_graph_state")
            return ToolResult(
                success=True,
                output=output,
                observation=get_state() if get_state else "",
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Executor: {tool_call.tool_name} failed — {e}")
            return ToolResult(success=False, error=str(e))

    def suggest_recovery(
        self,
        failed: ToolResult,
        original: ToolCall,
        context: Dict[str, Any],
    ) -> Optional[ToolCall]:
        if original.tool_name.startswith("verify_"):
            return None

        prompt = EXECUTOR_PROMPT.format(
            tool_name=original.tool_name,
            error=failed.error,
            parameters=json.dumps(original.parameters, indent=2),
            graph_state=context.get("graph_state", "{}"),
        )
        resp = self._call_llm_sync(
            f"Tool {original.tool_name} failed: {failed.error}", prompt
        )
        data = self._extract_json(resp)
        if data and data.get("should_retry") and "recovery" in data:
            r = data["recovery"]
            return ToolCall(
                tool_name=r.get("tool", ""),
                parameters=r.get("params", {}),
                description=r.get("description", "Recovery attempt"),
            )
        return None

    def apply_fix(
        self,
        original: ToolCall,
        suggestions: List[str],
        context: Dict[str, Any],
    ) -> Optional[ToolCall]:
        if not suggestions:
            return None
        prompt = (
            f"Fix the parameters based on these suggestions.\n"
            f"Tool: {original.tool_name}\n"
            f"Params: {json.dumps(original.parameters, indent=2)}\n"
            f"User Request: {context.get('user_request', '')}\n"
            f"Last Result: {json.dumps(context.get('last_result', {}), indent=2)}\n"
            f"Suggestions: {suggestions}\n\n"
            'Respond with JSON: {"params": {...}}'
        )
        resp = self._call_llm_sync(prompt, "You fix tool parameters based on suggestions.")
        data = self._extract_json(resp)
        if data and "params" in data:
            return ToolCall(
                tool_name=original.tool_name,
                parameters=data["params"],
                description=f"Fixed: {original.description}",
                expected_outcome=original.expected_outcome,
            )
        return None


# ===================================================================
#  Critic Agent
# ===================================================================

class CriticAgent(BaseAgent):
    """Evaluates execution results against expectations."""

    role = AgentRole.CRITIC

    def process(self, context: Dict[str, Any]) -> CritiqueResult:
        result = context.get("result")
        return self.evaluate(result, context) if result else CritiqueResult(approved=True)

    def evaluate(
        self,
        result: ToolResult,
        context: Dict[str, Any],
        expected_outcome: str = "",
        step_num: int = 1,
        total_steps: int = 1,
    ) -> CritiqueResult:
        if isinstance(result.output, dict) and "ok" in result.output and context.get("tool_name", "").startswith("verify_"):
            issues = result.output.get("issues", [])
            return CritiqueResult(
                approved=bool(result.output.get("ok", False)),
                issues=issues,
                suggestions=["Repair the graph plan until the verifier returns ok=true"],
                design_feedback="Preflight verification result",
                severity=3 if issues else 1,
            )

        get_state = context.get("get_graph_state")
        state = get_state() if get_state else context.get("graph_state", "{}")

        prompt = CRITIC_PROMPT.format(
            step_num=step_num,
            total_steps=total_steps,
            success=result.success,
            tool_name=context.get("tool_name", "?"),
            output=str(result.output)[:500] if result.output else "None",
            expected=expected_outcome or "N/A",
            observation=state[:1000] if state else "{}",
            user_request=context.get("user_request", "?"),
        )

        logger.info("Critic: evaluating step result")
        resp = self._call_llm_sync("Evaluate this execution result.", prompt)
        return self._parse(resp)

    def _parse(self, response: str) -> CritiqueResult:
        data = self._extract_json(response)
        if data:
            return CritiqueResult(
                approved=data.get("approved", True),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                design_feedback=data.get("design_feedback", ""),
                severity=data.get("severity", 1),
            )
        return CritiqueResult(approved=True)


# ===================================================================
#  Orchestrator
# ===================================================================

class AgentOrchestrator:
    """
    Hierarchical orchestrator with domain-specialist planners.

    Complex requests are first classified by a lightweight **Supervisor**
    that decomposes them into domain-specific sub-tasks.  Each sub-task
    is planned by a focused specialist whose prompt contains *only* the
    tools and schema relevant to its domain — dramatically reducing the
    context window so that GPT-4o-class models produce reliable plans.
    """

    def __init__(
        self,
        llm_provider: 'LLMProvider',
        tool_registry: ToolRegistry,
        use_critic: bool = True,
        validate_design_intent: bool = True,  # accepted for API compat
        max_retries: int = 3,
        on_plan_created: Optional[Callable[[ExecutionPlan], None]] = None,
        on_step_complete: Optional[Callable[[AgentStep], None]] = None,
        on_complete: Optional[Callable[[List[ToolResult]], None]] = None,
    ):
        self.llm = llm_provider
        self.tool_registry = tool_registry

        # --- agents ---
        self.supervisor = SupervisorAgent(llm_provider)
        self.decomposer = DecomposerAgent(llm_provider)

        self.cad_planner = DomainPlannerAgent(
            llm_provider, tool_registry, CAD_PLANNER_PROMPT,
            tool_categories=["cad", "navigation"],
            schema_func=get_cad_schema_for_prompt,
        )
        self.assembly_agent = AssemblyAgent(llm_provider, tool_registry)
        self.modeling_planner = DomainPlannerAgent(
            llm_provider, tool_registry, MODELING_PLANNER_PROMPT,
            tool_categories=["modeling", "navigation"],
            schema_func=get_modeling_schema_for_prompt,
        )
        self.system_model_agent = SystemModelAgent(llm_provider, tool_registry)
        self.analysis_planner = DomainPlannerAgent(
            llm_provider, tool_registry, ANALYSIS_PLANNER_PROMPT,
            tool_categories=["analysis", "modeling", "navigation"],
        )
        self.general_planner = DomainPlannerAgent(
            llm_provider, tool_registry, GENERAL_PLANNER_PROMPT,
            tool_categories=["navigation", "project"],
        )
        self.graph_edit_agent = GraphEditAgent(llm_provider, tool_registry)

        self.verifier = PlanVerifierAgent()
        self.executor = ExecutorAgent(llm_provider, tool_registry)
        self.critic = CriticAgent(llm_provider) if use_critic else None

        # Compat alias (manager.py historically accessed .planner)
        self.planner = self.cad_planner

        self.max_retries = max_retries
        self.on_plan_created = on_plan_created
        self.on_step_complete = on_step_complete
        self.on_complete = on_complete
        self.telemetry = AgentSessionTelemetry()
        self._recent_planner_feedback: List[Dict[str, Any]] = []

        self.react_loop = ReActLoop(
            executor=self.executor,
            critic=self.critic,
            max_iterations=max_retries,
            on_step=self._on_step_complete_internal,
        )

        logger.info(
            f"Hierarchical AgentOrchestrator ready "
            f"(critic={'on' if use_critic else 'off'}, "
            f"tools={len(tool_registry.all_tools)})"
        )

    # ---- public API ------------------------------------------------

    def update_provider(self, provider: 'LLMProvider') -> None:
        """Propagate a new LLM provider to every agent."""
        self.llm = provider
        for agent in (
            self.supervisor,
            self.decomposer,
            self.cad_planner,
            self.assembly_agent,
            self.modeling_planner,
            self.system_model_agent,
            self.analysis_planner,
            self.general_planner,
            self.graph_edit_agent,
            self.executor,
        ):
            agent.llm = provider
        if self.critic:
            self.critic.llm = provider

    def process_request(
        self,
        user_request: str,
        graph_state: str = "",
        get_graph_state: Optional[Callable[[], str]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point — identical return format to the old orchestrator.

        Returns dict with keys: success, results, plan, message.
        """
        get_state = get_graph_state or (lambda: graph_state)
        context: Dict[str, Any] = {
            "user_request": user_request,
            "graph_state": graph_state,
            "get_graph_state": get_state,
            "recent_planner_feedback": self._get_recent_feedback(),
        }
        self.telemetry.start_request()

        # 1. Fast-path for trivially obvious requests
        classification = self._fast_classify(user_request)

        # 2. Otherwise ask the Supervisor
        if classification is None:
            logger.info("Supervisor: classifying request…")
            classification = self.supervisor.process(context)

        logger.info(
            f"Classification → domain={classification.domain}, "
            f"complexity={classification.complexity}, "
            f"sub_tasks={len(classification.sub_tasks)}"
        )

        # 2b. Guard: if Supervisor returned no CAD sub-tasks but the request
        #     clearly asks to create geometry, override to a single CAD task.
        has_cad = any(s.domain == "cad" for s in classification.sub_tasks)
        if not has_cad and classification.domain != "conversation":
            r_low = user_request.lower()
            if any(v in r_low for v in ("create", "make", "build", "design", "draw")):
                logger.warning(
                    "Supervisor returned no CAD sub-tasks for a creation "
                    "request — overriding to CAD compound"
                )
                classification = TaskClassification(
                    domain="cad",
                    complexity="compound",
                    sub_tasks=[SubTask("cad", user_request)],
                    reasoning="Override: creation request must go to CAD",
                )

        # 3. Conversation — answer directly, no tools
        if classification.domain == "conversation":
            return self._handle_conversation(user_request, context)

        # 4. Execute each sub-task via its domain planner → executor → critic
        all_results: List[ToolResult] = []
        combined_steps: List[ToolCall] = []

        for i, sub in enumerate(classification.sub_tasks):
            sub_context = {
                **context,
                "task_description": sub.task,
                "graph_state": get_state(),
            }

            # Think first: for CAD tasks, try decomposition into sub-parts
            if sub.domain == "cad":
                decomposed = self.decomposer.process(sub_context)
                if decomposed:
                    # Execute each sub-part through the CAD planner
                    logger.info(
                        f"Decomposer split into {len(decomposed)} sub-parts"
                    )
                    decomp_ok = True
                    # Track the final output node ID of each sub-part
                    # so the assembly step can reference them directly.
                    sub_part_outputs: Dict[int, str] = {}  # 1-based index → node ID
                    for pi, part in enumerate(decomposed):
                        # Prefix instruction to avoid node-ID collisions
                        prefix_hint = (
                            f"[Sub-part {pi+1}/{len(decomposed)}] "
                            f"Prefix ALL node IDs with 'p{pi+1}_' "
                            f"(e.g. p{pi+1}_cyl, p{pi+1}_sk). "
                        )

                        # If this sub-part depends on earlier ones,
                        # list the available output shapes it can reference.
                        avail_shapes = ""
                        if part.depends_on and sub_part_outputs:
                            refs = []
                            for dep_idx in part.depends_on:
                                nid = sub_part_outputs.get(dep_idx)
                                if nid:
                                    refs.append(f"p{dep_idx}: {nid}.shape")
                            if refs:
                                avail_shapes = (
                                    "\nAvailable shapes from earlier sub-parts "
                                    "(connect to these directly, do NOT create "
                                    "proxy/parameter nodes): "
                                    + ", ".join(refs) + ". "
                                )

                        part_context = {
                            **context,
                            "task_description": prefix_hint + part.description + avail_shapes,
                            "graph_state": get_state(),
                            "active_domain": "cad",
                        }
                        part_planner = self._select_planner("cad", part_context)
                        part_context["planner_name"] = self._planner_name(part_planner)
                        part_context["recent_planner_feedback"] = self._get_recent_feedback("cad")
                        part_plan = part_planner.process(part_context)
                        part_plan = self.verifier.process(part_plan)
                        self.telemetry.record_plan(part_context["planner_name"], len(part_plan.steps))
                        combined_steps.extend(part_plan.steps)

                        if self.on_plan_created:
                            self.on_plan_created(part_plan)

                        if not part_plan.steps:
                            logger.warning(
                                f"Sub-part {pi+1}/{len(decomposed)} empty"
                            )
                            continue

                        # Extract the final output node ID from the plan
                        for step in part_plan.steps:
                            if step.tool_name == "create_cad_geometry":
                                pnodes = step.parameters.get("nodes", [])
                                if pnodes:
                                    sub_part_outputs[pi + 1] = pnodes[-1]["id"]

                        part_exec_context = {
                            **context,
                            "graph_state": get_state(),
                            "active_domain": "cad",
                            "planner_name": part_context["planner_name"],
                            "recent_planner_feedback": self._get_recent_feedback("cad"),
                        }
                        part_results = self.react_loop.run(part_plan, part_exec_context)
                        self._merge_planner_feedback(part_exec_context)
                        all_results.extend(part_results)

                        if part_results and not part_results[-1].success:
                            logger.error(
                                f"Sub-part {pi+1} failed — skipping rest"
                            )
                            decomp_ok = False
                            break

                        context["graph_state"] = get_state()

                    if not decomp_ok:
                        break
                    continue  # skip the normal planner path below

            # Normal path: single planner call
            planner = self._select_planner(sub.domain, sub_context)
            sub_context["planner_name"] = self._planner_name(planner)
            sub_context["active_domain"] = sub.domain
            sub_context["recent_planner_feedback"] = self._get_recent_feedback(sub.domain)
            plan = planner.process(sub_context)
            plan = self.verifier.process(plan)
            self.telemetry.record_plan(sub_context["planner_name"], len(plan.steps))
            combined_steps.extend(plan.steps)

            if self.on_plan_created:
                self.on_plan_created(plan)

            if not plan.steps:
                logger.warning(f"Sub-task {i + 1} yielded empty plan: {sub.task[:60]}")
                all_results.append(
                    ToolResult(success=False, error="Planner returned empty plan")
                )
                break

            exec_context = {
                **context,
                "graph_state": get_state(),
                "active_domain": sub.domain,
                "planner_name": sub_context["planner_name"],
                "recent_planner_feedback": self._get_recent_feedback(sub.domain),
            }
            results = self.react_loop.run(plan, exec_context)
            self._merge_planner_feedback(exec_context)
            all_results.extend(results)

            # Stop on failure
            if results and not results[-1].success:
                logger.error(f"Sub-task {i + 1} failed — aborting remaining sub-tasks")
                break

            # Refresh state for the next sub-task
            context["graph_state"] = get_state()

        # 5. Aggregate and return
        combined_plan = ExecutionPlan(
            goal=user_request,
            steps=combined_steps,
            reasoning=classification.reasoning,
            estimated_complexity=len(classification.sub_tasks),
        )

        all_ok = all(r.success for r in all_results) if all_results else False

        if self.on_complete:
            self.on_complete(all_results)

        return {
            "success": all_ok,
            "results": all_results,
            "plan": combined_plan,
            "message": self._summarize(combined_plan, all_results, user_request),
            "telemetry": self.telemetry.snapshot(),
            "telemetry_summary": self.telemetry.compact_summary(),
        }

    def _on_step_complete_internal(self, step: AgentStep) -> None:
        if step.action and step.result:
            self.telemetry.record_tool_result(step.action.tool_name, step.result.success)
            if step.action.tool_name.startswith("verify_") and isinstance(step.result.output, dict):
                target_tool = step.action.parameters.get("target_tool", step.action.tool_name)
                self.telemetry.record_verifier_result(target_tool, step.result.output)

        if self.on_step_complete:
            self.on_step_complete(step)

    def _merge_planner_feedback(self, run_context: Dict[str, Any]) -> None:
        entries = run_context.get("planner_feedback", [])
        if not entries:
            return
        for entry in entries:
            if entry not in self._recent_planner_feedback:
                self._recent_planner_feedback.append(entry)
        self._recent_planner_feedback = self._recent_planner_feedback[-8:]

    def _get_recent_feedback(self, domain: str = "") -> List[Dict[str, Any]]:
        if not domain:
            return self._recent_planner_feedback[-4:]
        filtered = [entry for entry in self._recent_planner_feedback if entry.get("domain") in ("", domain)]
        return filtered[-4:]

    def _planner_name(self, planner: BaseAgent) -> str:
        if planner is self.cad_planner:
            return "cad_planner"
        if planner is self.assembly_agent:
            return "assembly_agent"
        if planner is self.modeling_planner:
            return "modeling_planner"
        if planner is self.system_model_agent:
            return "system_model_agent"
        if planner is self.analysis_planner:
            return "analysis_planner"
        if planner is self.general_planner:
            return "general_planner"
        if planner is self.graph_edit_agent:
            return "graph_edit_agent"
        return planner.__class__.__name__

    def get_available_tools_summary(self) -> str:
        """Get a summary of available tools for display."""
        return self.tool_registry.get_tools_description()

    # ---- private helpers -------------------------------------------

    _CAD_VERBS = frozenset({"create", "make", "build", "design", "draw", "model",
                             "generate", "construct", "add", "place"})
    _CAD_SIM_HINTS = frozenset({
        "topology", "topopt", "mesh", "constraint", "constraints",
        "load", "loads", "solver", "stress", "fea", "simulation",
        "material", "compliance",
    })
    _ANALYSIS_HINTS = frozenset({
        "optimization", "optimize", "sensitivity", "surrogate", "doe",
        "sampling", "workflow", "analysis",
    })
    _ASSEMBLY_HINTS = frozenset({
        "assembly", "assemble", "combine", "union", "position", "place",
        "arrange", "mount", "attach", "align", "mirror", "pattern",
    })
    _EDIT_HINTS = frozenset({
        "edit", "modify", "change", "update", "adjust", "tweak", "rename",
        "connect", "disconnect", "set", "move", "reposition", "resize",
        "increase", "decrease", "replace",
    })
    _SYSTEM_MODEL_HINTS = frozenset({
        "variable", "variables", "equation", "equations", "function", "functions",
        "input", "inputs", "output", "outputs", "constraint", "constraints",
        "objective", "objectives", "system", "architecture", "intermediate",
    })

    def _fast_classify(self, request: str) -> Optional[TaskClassification]:
        """Skip the Supervisor for trivially obvious single-action requests."""
        r = request.lower().strip()
        words = set(r.split())

        # Navigation
        if any(r.startswith(p) for p in ("switch to ", "go to ", "open ")):
            return TaskClassification(
                "navigation", "simple",
                [SubTask("navigation", request)],
            )

        # Project
        if any(w in r for w in ("save project", "load project", "new project")):
            return TaskClassification(
                "project", "simple",
                [SubTask("project", request)],
            )

        # Conversational (questions / explanations)
        if r.startswith(("what ", "how ", "why ", "who ", "when ",
                         "explain ", "tell me", "describe ", "is ", "can ")):
            return TaskClassification(
                "conversation", "simple",
                [SubTask("conversation", request)],
            )

        # CAD simulation workflows are still CAD graph construction, not generic analysis.
        if any(hint in r for hint in self._CAD_SIM_HINTS) and (
            words & self._CAD_VERBS or any(noun in r for noun in ("beam", "plate", "bracket", "part", "geometry", "shape"))
        ):
            logger.info("Fast-classify: CAD simulation keywords detected")
            return TaskClassification(
                "cad", "compound",
                [SubTask("cad", request)],
            )

        # Analysis/workflow requests must outrank generic creation verbs.
        if any(hint in r for hint in self._ANALYSIS_HINTS):
            logger.info("Fast-classify: analysis/workflow keywords detected")
            return TaskClassification(
                "analysis", "compound",
                [SubTask("analysis", request)],
            )

        # CAD creation — "create me a boeing 747", "make a gear", etc.
        if words & self._CAD_VERBS:
            logger.info(f"Fast-classify: CAD creation verb detected")
            return TaskClassification(
                "cad", "compound",
                [SubTask("cad", request)],
            )

        return None  # needs full Supervisor

    def _get_planner(self, domain: str) -> DomainPlannerAgent:
        """Route a domain name to the appropriate specialist planner."""
        return {
            "cad": self.cad_planner,
            "modeling": self.modeling_planner,
            "analysis": self.analysis_planner,
        }.get(domain, self.general_planner)

    def _select_planner(self, domain: str, context: Dict[str, Any]) -> DomainPlannerAgent:
        """Pick the most appropriate planner for the current task and state."""
        task = str(context.get("task_description", context.get("user_request", ""))).lower()
        graph_state = context.get("graph_state", "")

        if self._looks_like_edit(task, graph_state):
            return self.graph_edit_agent

        if domain == "cad" and any(hint in task for hint in self._ASSEMBLY_HINTS):
            return self.assembly_agent

        if domain == "modeling" and any(hint in task for hint in self._SYSTEM_MODEL_HINTS):
            return self.system_model_agent

        return self._get_planner(domain)

    @staticmethod
    def _has_graph_state(graph_state: Any) -> bool:
        if not graph_state:
            return False
        if isinstance(graph_state, str):
            if graph_state == "Empty graph":
                return False
            try:
                graph_state = json.loads(graph_state)
            except Exception:
                return bool(graph_state.strip())
        if isinstance(graph_state, dict):
            return bool(graph_state.get("nodes"))
        return True

    def _looks_like_edit(self, task: str, graph_state: Any) -> bool:
        return self._has_graph_state(graph_state) and any(hint in task for hint in self._EDIT_HINTS)

    def _handle_conversation(
        self, question: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Answer a conversational question (no tool actions)."""
        state = context.get("graph_state", "")
        prompt = (
            "You are a knowledgeable engineering assistant for PyLCSS, "
            "an engineering design platform covering CAD, system modelling, "
            "optimization, sensitivity analysis, surrogate modelling, "
            "and solution-space exploration.\n"
            "Answer the user's question clearly and concisely."
        )
        if state:
            prompt += f"\n\nCurrent workspace state:\n{state[:1000]}"

        # Re-use the supervisor's LLM connection for the answer
        self.llm.clear_history()
        try:
            completion = self.llm.chat(question, system_prompt=prompt)
            answer = completion.content
        except Exception as e:
            logger.error(f"Conversation LLM call failed: {e}")
            answer = "I couldn't generate a response."

        return {
            "success": True,
            "results": [],
            "plan": ExecutionPlan(
                goal=question, steps=[], reasoning="Conversational response"
            ),
            "message": answer,
        }

    @staticmethod
    def _summarize(
        plan: ExecutionPlan,
        results: List[ToolResult],
        request: str,
    ) -> str:
        """Create a concise human-readable summary."""
        total = len(results)
        if total == 0:
            return f"I did not execute any actions for '{request}'."

        ok = sum(1 for r in results if r.success)
        goal_text = plan.goal or request
        if ok == total:
            return f"I completed the requested workflow for '{goal_text}'."
        if ok > 0:
            errors = "; ".join(
                (r.error or "unknown issue") for r in results if not r.success
            )[:240]
            return (
                f"I completed {ok} of {total} steps for '{goal_text}', "
                f"but I ran into this issue: {errors}."
            )
        return f"I could not complete '{goal_text}' because: {results[0].error or 'unknown error'}."
