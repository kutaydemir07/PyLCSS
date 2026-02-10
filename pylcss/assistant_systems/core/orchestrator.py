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

SUPERVISOR_PROMPT = '''You are the Supervisor for PyLCSS, an engineering design platform.
Classify the request and, if complex, decompose it into sub-tasks.

## Domains
- **cad** — 3D geometry: primitives, booleans, transforms, patterns, extrude, etc.
- **modeling** — System modelling: design variables, functions, connections, validation
- **analysis** — Optimization, sensitivity, surrogate training, DOE sampling
- **navigation** — Switching application tabs
- **project** — Save / load / new project
- **conversation** — Questions, explanations (no tool action needed)

## Complexity
- **simple** — 1 action
- **compound** — 2-4 related actions in ONE domain
- **complex** — Spans multiple domains or 5+ actions

## CRITICAL Rules
1. "Create", "make", "build", "design", "draw" + ANY object → **domain = cad**, always.
   Complex objects (car, aircraft, gearbox) are STILL cad — the Decomposer handles breakdown.
2. simple / compound → return ONE sub_task with the FULL original request, domain = cad.
3. complex → break into sequential sub-tasks, each targeting ONE domain.
4. Insert **navigation** sub-tasks before domain work when a tab switch is needed.
5. If the user asks a question → domain = conversation.
6. NEVER return sub-tasks that "explain", "ask the user", or "discuss".
   Your job is to EXECUTE, not to ask for clarification.
7. When in doubt, classify as **cad** with **compound** complexity.

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

CAD_PLANNER_PROMPT = '''You are the **CAD Specialist** for PyLCSS.
Create a plan using ONLY the tools and node types listed below.

## Tools
{cad_tools}

## Node Types (condensed)
{cad_schema}

## Current Graph
{current_state}

## ⚠ THINK BEFORE YOU ACT (mandatory)
Before writing the steps, write a short 1-2 sentence \"reasoning\" explaining your approach.
Include: what shape, which recipe/approach, key dimensions in mm.

## CRITICAL Rules — violating these causes failure
1. Put ALL nodes AND connections in ONE `create_cad_geometry` call.
2. Always finish with `execute_cad`.
3. ONLY use `type` values that appear in the Node Types list above.
   If a type is NOT listed, it DOES NOT EXIST. Do not guess or invent.
   Do NOT add the `com.cad.` prefix — it is added automatically.
4. ONLY use property names that appear in the Node Types list above.
   If a property is NOT listed for that node type, it DOES NOT EXIST. Do not guess.
5. Connection format: {{"from": "id.port", "to": "id.port"}}
   - Boolean inputs: `shape_a` and `shape_b`  (NEVER `shape_1` or `input`)
   - Other ops (fillet, translate, etc.): `shape`
   - Sketch-based: sketch node → `sketch` port into sketch element → `shape` out
6. ALWAYS use a Recipe below when the request matches one.
   Adapt dimensions but keep the same node types, connections, and structure.
7. Keep it SIMPLE — ≤ 6 nodes per call. Build complex shapes from fewer, bigger primitives.
8. **Cross-referencing earlier sub-parts**: When your task description lists
   "Available shapes" from earlier sub-parts, connect DIRECTLY to those node IDs.
   Example: `{{"from": "p1_body.shape", "to": "p8_union.shape_a"}}`.
   Do NOT create proxy/parameter/reference nodes — the earlier nodes already exist in the graph.

## Recipes — COPY these patterns, only change dimensions

### Gear tooth + hub + circular pattern
Hub cylinder + one tooth (sketch polyline → extrude) at pitch radius + Boolean Union + Circular Pattern.
For helical gears: use twisted_extrude instead of extrude.
```json
{{"nodes": [
  {{"type":"cylinder","id":"hub","properties":{{"cyl_radius":20,"cyl_height":10}}}},
  {{"type":"sketch","id":"sk","properties":{{"plane":"XY"}}}},
  {{"type":"polyline","id":"tooth","properties":{{"points":"[(18,-1.5),(24,-1.2),(25,0),(24,1.2),(18,1.5)]","closed":true}}}},
  {{"type":"extrude","id":"ext","properties":{{"extrude_distance":10}}}},
  {{"type":"boolean","id":"merged","properties":{{"operation":"Union"}}}},
  {{"type":"circular_pattern","id":"gear","properties":{{"count":20,"angle":360,"axis_x":0,"axis_y":0,"axis_z":1}}}}
],
"connections": [
  {{"from":"sk.sketch","to":"tooth.sketch"}},
  {{"from":"tooth.shape","to":"ext.shape"}},
  {{"from":"hub.shape","to":"merged.shape_a"}},
  {{"from":"ext.shape","to":"merged.shape_b"}},
  {{"from":"merged.shape","to":"gear.shape"}}
]}}
```
Tooth profile tips: points go from root radius inward, curve out to tip, back to root.
Scale tooth width ≈ pitch_circumference / (2 × num_teeth). Adjust radius values proportionally.

### Box with hole
```json
{{"nodes": [
  {{"type":"box","id":"base","properties":{{"box_length":100,"box_width":60,"box_depth":40}}}},
  {{"type":"cylinder","id":"hole","properties":{{"cyl_radius":15,"cyl_height":50}}}},
  {{"type":"boolean","id":"cut","properties":{{"operation":"Cut"}}}}
],
"connections": [
  {{"from":"base.shape","to":"cut.shape_a"}},
  {{"from":"hole.shape","to":"cut.shape_b"}}
]}}
```

### Plate with hole pattern
```json
{{"nodes": [
  {{"type":"box","id":"plate","properties":{{"box_length":100,"box_width":60,"box_depth":10}}}},
  {{"type":"array_holes","id":"holes","properties":{{"x_start":15,"y_start":15,"x_spacing":20,"y_spacing":30,"x_count":4,"y_count":2,"diameter":8,"through_all":true,"from_face":">Z"}}}}
],
"connections": [{{"from":"plate.shape","to":"holes.shape"}}]}}
```

### Hollow cylinder (pipe/tube)
```json
{{"nodes": [
  {{"type":"cylinder","id":"outer","properties":{{"cyl_radius":20,"cyl_height":50}}}},
  {{"type":"cylinder","id":"inner","properties":{{"cyl_radius":15,"cyl_height":60}}}},
  {{"type":"boolean","id":"tube","properties":{{"operation":"Cut"}}}}
],
"connections": [
  {{"from":"outer.shape","to":"tube.shape_a"}},
  {{"from":"inner.shape","to":"tube.shape_b"}}
]}}
```

### Flange
```json
{{"nodes": [
  {{"type":"cylinder","id":"disc","properties":{{"cyl_radius":40,"cyl_height":8}}}},
  {{"type":"cylinder","id":"bore","properties":{{"cyl_radius":15,"cyl_height":10}}}},
  {{"type":"boolean","id":"base","properties":{{"operation":"Cut"}}}},
  {{"type":"multi_hole","id":"bolts","properties":{{"coordinates":"[(30,0),(21.2,21.2),(0,30),(-21.2,21.2),(-30,0),(-21.2,-21.2),(0,-30),(21.2,-21.2)]","diameter":6,"through_all":true,"from_face":">Z"}}}}
],
"connections": [
  {{"from":"disc.shape","to":"base.shape_a"}},
  {{"from":"bore.shape","to":"base.shape_b"}},
  {{"from":"base.shape","to":"bolts.shape"}}
]}}
```

### L-Bracket / extruded profile
```json
{{"nodes": [
  {{"type":"sketch","id":"sk","properties":{{"plane":"XY"}}}},
  {{"type":"polyline","id":"profile","properties":{{"points":"[(0,0),(30,0),(30,5),(5,5),(5,25),(0,25)]","closed":true}}}},
  {{"type":"extrude","id":"ext","properties":{{"extrude_distance":10}}}},
  {{"type":"fillet","id":"fil","properties":{{"fillet_radius":2}}}}
],
"connections": [
  {{"from":"sk.sketch","to":"profile.sketch"}},
  {{"from":"profile.shape","to":"ext.shape"}},
  {{"from":"ext.shape","to":"fil.shape"}}
]}}
```

### Shaft with keyway
```json
{{"nodes": [
  {{"type":"cylinder","id":"shaft","properties":{{"cyl_radius":10,"cyl_height":80}}}},
  {{"type":"box","id":"keyway","properties":{{"box_length":20,"box_width":4,"box_depth":3,"center_x":0,"center_y":9,"center_z":0}}}},
  {{"type":"boolean","id":"cut","properties":{{"operation":"Cut"}}}}
],
"connections": [
  {{"from":"shaft.shape","to":"cut.shape_a"}},
  {{"from":"keyway.shape","to":"cut.shape_b"}}
]}}
```

If no Recipe matches, build the shape from primitives + booleans + patterns.
NOTE: You may receive a sub-part description from the decomposer. Build ONLY that part, not the whole assembly.
The assembly/positioning will be handled in subsequent calls.

**NODE IDs**: When building a sub-part, prefix all node IDs with the part label
(e.g. `fuselage_cyl`, `wing_sk`, `engine1_cyl`). This prevents ID collisions when
multiple sub-parts are added to the same graph.

Respond with ONLY a JSON block:
```json
{{
  "goal": "...",
  "reasoning": "Brief: what shape, recipe used, key dimensions in mm",
  "steps": [
    {{"tool": "create_cad_geometry", "params": {{"nodes": [...], "connections": [...]}}, "description": "...", "expected": "..."}},
    {{"tool": "execute_cad", "params": {{}}, "description": "Render", "expected": "Shape in viewer"}}
  ],
  "complexity": N
}}
```'''


MODELING_PLANNER_PROMPT = '''You are the **Modeling Specialist** for PyLCSS.
Create a plan using ONLY the tools and node types listed below.

## Tools
{modeling_tools}

## System Node Types
{modeling_schema}

## Current Graph
{current_state}

## Rules
1. Use `create_system_model` to create all nodes + connections in one call.
2. `com.pfd.input` — design variable. Set `var_name`, `min`, `max`, and optionally `unit`.
3. `com.pfd.output` — quantity of interest (QoI). Set `var_name`, and optionally:
   - `unit` — physical unit (e.g. "kg", "mm", "N/m^2", or "-" for dimensionless)
   - `req_min` / `req_max` — requirement bounds (constraints). Use "-1e9" / "1e9" for no bound.
   - `minimize` (bool) — set True to make this an objective to minimize
   - `maximize` (bool) — set True to make this an objective to maximize
4. `com.pfd.intermediate` — internal variable for chaining blocks. Set `var_name`, optionally `unit`.
5. `com.pfd.custom_block` — Python function. Set `num_inputs`, `num_outputs`, `code_content`.
   Code uses `in_1, in_2, ...` as inputs, assigns to `out_1, out_2, ...` as outputs.
   Supports numpy (np) and math.
6. **Port naming**:
   - Input/Output/Intermediate: port name = `var_name` (e.g. var_name="width" → port "width")
   - CustomBlock: ports are `in_1, in_2, ...` and `out_1, out_2, ...`
   - Connection format: `{{"from": "nodeId.portName", "to": "nodeId.portName"}}`
7. After creation, call `validate_model`.
8. Use `build_model` to transfer for analysis.

## Example 1 — Simple: f(x,y) = x² + y², minimize f
```json
{{
  "goal": "Model f = x² + y², minimize f",
  "reasoning": "Two inputs, one function block, one output with minimize objective",
  "steps": [
    {{
      "tool": "create_system_model",
      "params": {{
        "nodes": [
          {{"type": "com.pfd.input", "id": "x", "properties": {{"var_name": "x", "min": -10, "max": 10}}}},
          {{"type": "com.pfd.input", "id": "y", "properties": {{"var_name": "y", "min": -10, "max": 10}}}},
          {{"type": "com.pfd.custom_block", "id": "fn", "properties": {{"code_content": "out_1 = in_1**2 + in_2**2", "num_inputs": 2, "num_outputs": 1}}}},
          {{"type": "com.pfd.output", "id": "f", "properties": {{"var_name": "f", "minimize": true}}}}
        ],
        "connections": [
          {{"from": "x.x", "to": "fn.in_1"}},
          {{"from": "y.y", "to": "fn.in_2"}},
          {{"from": "fn.out_1", "to": "f.f"}}
        ]
      }},
      "description": "Create model graph",
      "expected": "4-node system model"
    }},
    {{"tool": "validate_model", "params": {{}}, "description": "Validate", "expected": "No errors"}}
  ],
  "complexity": 2
}}
```

## Example 2 — With units, requirements, intermediate, and multi-output function
```json
{{
  "goal": "Beam model: width/height inputs in mm, compute stress and weight with requirements",
  "reasoning": "2 inputs with units + 1 function (2 outputs) + intermediate + 2 QoIs with requirements",
  "steps": [
    {{
      "tool": "create_system_model",
      "params": {{
        "nodes": [
          {{"type": "com.pfd.input", "id": "w", "properties": {{"var_name": "width", "min": 10, "max": 100, "unit": "mm"}}}},
          {{"type": "com.pfd.input", "id": "h", "properties": {{"var_name": "height", "min": 10, "max": 200, "unit": "mm"}}}},
          {{"type": "com.pfd.custom_block", "id": "beam_calc", "properties": {{
            "num_inputs": 2, "num_outputs": 2,
            "code_content": "area = in_1 * in_2\\nout_1 = 1000 / area\\nout_2 = area * 0.001 * 7850"
          }}}},
          {{"type": "com.pfd.intermediate", "id": "s", "properties": {{"var_name": "stress", "unit": "MPa"}}}},
          {{"type": "com.pfd.output", "id": "stress_out", "properties": {{"var_name": "max_stress", "unit": "MPa", "req_max": "250"}}}},
          {{"type": "com.pfd.output", "id": "weight_out", "properties": {{"var_name": "weight", "unit": "kg", "req_max": "50", "minimize": true}}}}
        ],
        "connections": [
          {{"from": "w.width", "to": "beam_calc.in_1"}},
          {{"from": "h.height", "to": "beam_calc.in_2"}},
          {{"from": "beam_calc.out_1", "to": "s.stress"}},
          {{"from": "s.stress", "to": "stress_out.max_stress"}},
          {{"from": "beam_calc.out_2", "to": "weight_out.weight"}}
        ]
      }},
      "description": "Create beam model with units and requirements",
      "expected": "6-node model with stress constraint and weight objective"
    }},
    {{"tool": "validate_model", "params": {{}}, "description": "Validate", "expected": "No errors"}}
  ],
  "complexity": 3
}}
```

Respond with ONLY a JSON block matching the format above.'''


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

        logger.info(f"DomainPlanner{self.tool_categories}: planning '{task[:60]}…'")
        response = self._call_llm_sync(task, prompt, max_tokens=4096)
        return self._parse_plan(response)

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


# ===================================================================
#  Decomposer Agent — "Think first, then act"
# ===================================================================

DECOMPOSER_PROMPT = '''You are a **Geometry Decomposition Engineer** for PyLCSS CAD.
Your job: break complex geometry into simple, ordered sub-parts that can each be built independently.

## Your Process
1. **ANALYZE** — What is the user asking for? What are the key functional parts?
2. **DECOMPOSE** — Break it into the smallest independent sub-parts that can each
   be created with one `create_cad_geometry` call (primitives + booleans + patterns).
3. **ORDER** — Parts must be ordered by dependency.
   Earlier parts may be referenced by later assembly steps.
4. **DIMENSION** — Specify realistic engineering dimensions **in millimeters (mm)**.
   ALL dimensions must be in mm. A Boeing 747 fuselage is ~70000mm long, wingspan ~64000mm.
   Think about proportions and physical plausibility.

## Construction Capabilities (each sub-part can use these)
- Primitives: box, cylinder, sphere, cone, torus, wedge, pyramid
- Sketching: sketch → polyline/spline/polygon → extrude / twisted_extrude / revolve / sweep / loft
- Booleans: Union, Cut, Intersect
- Patterns: circular_pattern, linear_pattern, mirror
- Transforms: translate, rotate, scale
- Features: fillet, chamfer, shell, holes, slots, pockets

## Examples

**"Create a helical gear"** → 3 sub-parts:
1. "Hub: cylinder radius=23.4mm height=12mm"
2. "One tooth: sketch a closed tooth profile polyline on XY plane at pitch radius, twisted_extrude with helix twist"
3. "Assembly: boolean union hub+tooth, then circular_pattern count=20 around Z axis"

**"Create a simple car"** → 5 sub-parts:
1. "Car body: box 4500×1800×800mm for the lower body"
2. "Cabin: box 2500×1700×700mm positioned on top of body, union with body, fillet edges"
3. "Front wheel: cylinder radius=350mm height=250mm, rotated 90° around X, positioned at front-left"
4. "Rear wheel: same as front wheel, positioned at rear-left"
5. "Assembly: mirror both wheels to get right side, union all parts"

**"Create a flange with bolt holes"** → 2 sub-parts:
1. "Base: cylinder radius=40mm height=8mm, cut center bore radius=15mm"
2. "Features: array_holes on bolt circle diameter=60mm, 8 holes, diameter=6mm"

## Rules
- Each sub-part should be SIMPLE (≤6 nodes). The CAD planner handles the actual node details.
- Sub-parts describe WHAT to build in engineering terms, not HOW (no node types or JSON).
- Include key dimensions, positions, and orientations.
- Later sub-parts can reference earlier ones for positioning context.
- For teeth/blades/fins: describe the 2D profile shape + extrusion method, don't try to specify exact coordinates.
- Return 1 sub-part for simple requests, 2-5 for compound, up to 8 for complex assemblies.
- The LAST sub-part should be an "Assembly" step that positions and unions all earlier sub-parts.
  Its `depends_on` MUST list ALL earlier part indices (e.g. [1,2,3,4,5]).
  The assembly step gets direct access to each prior part's output shape —
  it only needs translate nodes to position them and boolean Union nodes to combine them.

Respond with ONLY JSON:
```json
{
  "analysis": "What this is, key features, overall dimensions",
  "sub_parts": [
    {"description": "Part 1: detailed construction description with dimensions", "depends_on": []},
    {"description": "Part 2: description with positioning relative to part 1", "depends_on": [1]},
    {"description": "Assembly: union/position all parts together", "depends_on": [1, 2]}
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

    def process(self, context: Dict[str, Any]) -> Optional[List[DecomposedPart]]:
        """Analyze a CAD request and optionally decompose it.

        Returns:
            List[DecomposedPart] if the request should be decomposed,
            None if the CAD planner should handle it directly.
        """
        task = context.get("task_description", context.get("user_request", ""))
        task_lower = task.lower()

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
        self.modeling_planner = DomainPlannerAgent(
            llm_provider, tool_registry, MODELING_PLANNER_PROMPT,
            tool_categories=["modeling", "navigation"],
            schema_func=get_modeling_schema_for_prompt,
        )
        self.analysis_planner = DomainPlannerAgent(
            llm_provider, tool_registry, ANALYSIS_PLANNER_PROMPT,
            tool_categories=["analysis", "modeling", "navigation"],
        )
        self.general_planner = DomainPlannerAgent(
            llm_provider, tool_registry, GENERAL_PLANNER_PROMPT,
            tool_categories=["navigation", "project"],
        )

        self.executor = ExecutorAgent(llm_provider, tool_registry)
        self.critic = CriticAgent(llm_provider) if use_critic else None

        # Compat alias (manager.py historically accessed .planner)
        self.planner = self.cad_planner

        self.max_retries = max_retries
        self.on_plan_created = on_plan_created
        self.on_step_complete = on_step_complete
        self.on_complete = on_complete

        self.react_loop = ReActLoop(
            executor=self.executor,
            critic=self.critic,
            max_iterations=max_retries,
            on_step=on_step_complete,
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
            self.modeling_planner,
            self.analysis_planner,
            self.general_planner,
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
        }

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
                        }
                        part_plan = self.cad_planner.process(part_context)
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

                        part_results = self.react_loop.run(
                            part_plan,
                            {**context, "graph_state": get_state()},
                        )
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
            planner = self._get_planner(sub.domain)
            plan = planner.process(sub_context)
            combined_steps.extend(plan.steps)

            if self.on_plan_created:
                self.on_plan_created(plan)

            if not plan.steps:
                logger.warning(f"Sub-task {i + 1} yielded empty plan: {sub.task[:60]}")
                all_results.append(
                    ToolResult(success=False, error="Planner returned empty plan")
                )
                break

            results = self.react_loop.run(
                plan, {**context, "graph_state": get_state()}
            )
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
        }

    def get_available_tools_summary(self) -> str:
        """Get a summary of available tools for display."""
        return self.tool_registry.get_tools_description()

    # ---- private helpers -------------------------------------------

    _CAD_VERBS = frozenset({"create", "make", "build", "design", "draw", "model",
                             "generate", "construct", "add", "place"})

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
            return f"No actions were executed for: {request}"

        ok = sum(1 for r in results if r.success)
        if ok == total:
            return f"✅ {plan.goal or request}"
        if ok > 0:
            errors = ", ".join(
                (r.error or "?") for r in results if not r.success
            )[:200]
            return f"⚠️ {ok}/{total} steps succeeded. Issues: {errors}"
        return f"❌ Failed: {results[0].error or 'Unknown error'}"
