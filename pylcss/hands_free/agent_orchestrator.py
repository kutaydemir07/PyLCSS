# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Agent Orchestrator - Coordinates multiple agents.

Implements the Planner → Executor → Critic workflow with:
- Same LLM for all agents (configurable)
- Design intent validation by Critic
- Automatic workflow recording
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

from .agent_core import (
    BaseAgent, AgentRole, ExecutionPlan, CritiqueResult, 
    ToolCall, ToolResult, ReActLoop, AgentStep
)
from .agent_tools import Tool, ToolRegistry, get_cad_schema_for_prompt

if TYPE_CHECKING:
    from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


# === System Prompts ===

PLANNER_PROMPT = '''You are the **Planner Agent** for PyLCSS, a CAD and system modeling application.

## Your Role
Analyze the user's request and create a step-by-step execution plan using the available tools.

## Available Tools
{tools_description}

## CAD Node Reference
{cad_schema}

## Current Graph State
{current_state}

## Planning Guidelines
1. **Understand Intent**: What is the user trying to achieve? 
2. **Use Core Primitives**: Maps ambiguous terms like "shell", "peel", or "hollow" to standard boolean operations.
   - Example: "Peel the gearbox" -> Create Cylinder 1 (outer) -> Create Cylinder 2 (inner) -> Connect (Boolean Difference).
   - "Drill a hole" -> Create Geometry -> Create Cylinder -> Boolean Difference.
3. **Strict Tool Usage**: **NEVER** invent tools. Use ONLY the tool names listed above.
   - WRONG: `com.cad.shell`, `create_hole`, `peel_object`
   - RIGHT: `create_cad_geometry`, `connect_cad_nodes`
4. **Break Down**: Split complex tasks into atomic `create_cad_geometry` and `connect_cad_nodes` calls.
5. **Anticipate Issues**: Ensure boolean operands intersect correctly.

## Response Format
Respond with ONLY a JSON block:
```json
{{
  "goal": "What the user wants to achieve",
  "reasoning": "Step-by-step logic. Explain how you mapped 'shell'/'peel' to booleans here.",
  "steps": [
    {{"tool": "tool_name", "params": {{}}, "expected": "What should happen", "description": "Human-readable action"}}
  ],
  "complexity": 1-5,
  "requires_confirmation": false
}}
```

IMPORTANT: Use exact tool names from the list above. Include all required parameters.
'''

EXECUTOR_PROMPT = '''You are the **Executor Agent** for PyLCSS.

Your job is to handle tool execution failures and suggest recovery actions.

## Current Situation
- **Failed Tool**: {tool_name}
- **Error**: {error}
- **Parameters Used**: {parameters}
- **Graph State**: {graph_state}

## Recovery Guidelines
1. **Unknown Tool Error**:
   - If error matches "Unknown tool", check the list of available tools.
   - Map the HALLUCINATED tool to a REAL tool.
   - Example: `com.cad.shell` -> Use `create_cad_geometry` (inner shape) + `connect_cad_nodes` (boolean difference).

2. **Geometry Error**:
   - If boolean fails -> Check if shapes actually intersect.
   - If profile fails -> Check if it is closed/planar.
   - Try simplifying parameters (e.g., smaller radius, simpler shape).

3. **Alternative Approach**:
   - If one method fails, try another.
   - Example: Loft failed? Try Extrude.

## Response Format
```json
{{
  "diagnosis": "What went wrong (hallucination, geometry error, etc.)",
  "should_retry": true,
  "recovery": {{
    "tool": "tool_name",
    "params": {{}},
    "description": "What this will do differently"
  }}
}}
```

If recovery is not possible, set "should_retry": false and explain in diagnosis.
'''

CRITIC_PROMPT = '''You are the **Critic Agent** for PyLCSS.

## Your Role
Evaluate execution results for:
1. **Geometry Validity**: No degenerate shapes, valid booleans, proper dimensions
2. **Design Intent**: Does the result match what the user actually wanted?
3. **Best Practices**: Are there better approaches?

## Execution Result
- **Success**: {success}
- **Tool Used**: {tool_name}
- **Output**: {output}
- **Observation**: {observation}

## Expected Outcome
{expected}

## Graph State After Execution
{graph_state}

## Original User Request
{user_request}

## Evaluation Guidelines
1. **Approve** if the result achieves the design intent, even if not perfect.
2. **Reject** if there are geometry errors, missing connections, or significant deviation from intent.
3. **Suggest** improvements for future iterations.

## Response Format
```json
{{
  "approved": true,
  "issues": ["list of problems found"],
  "suggestions": ["how to improve"],
  "design_feedback": "Does this match the user's design intent?",
  "severity": 1-5
}}
```

Severity: 1=minor cosmetic, 2=suboptimal, 3=functionality affected, 4=significant error, 5=critical failure
'''


class PlannerAgent(BaseAgent):
    """Creates execution plans from user requests."""
    
    role = AgentRole.PLANNER
    
    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(llm_provider)
        self.tools = tool_registry
        
    def process(self, context: Dict[str, Any]) -> ExecutionPlan:
        """Create an execution plan for the user's request."""
        user_request = context.get("user_request", "")
        current_state = context.get("graph_state", "{}")
        
        # Build prompt
        prompt = PLANNER_PROMPT.format(
            tools_description=self.tools.get_tools_description(),
            cad_schema=get_cad_schema_for_prompt(),
            current_state=current_state if current_state else "Empty graph",
        )
        
        # Call LLM
        logger.info(f"Planner: Creating plan for '{user_request[:50]}...'")
        response = self._call_llm_sync(user_request, prompt)
        
        # Parse plan
        plan = self._parse_plan(response)
        logger.info(f"Planner: Created plan with {len(plan.steps)} steps")
        
        return plan
        
    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse LLM response into an ExecutionPlan."""
        data = self._extract_json(response)
        
        if data:
            try:
                steps = []
                for step in data.get("steps", []):
                    steps.append(ToolCall(
                        tool_name=step.get("tool", ""),
                        parameters=step.get("params", {}),
                        expected_outcome=step.get("expected", ""),
                        description=step.get("description", ""),
                    ))
                    
                return ExecutionPlan(
                    goal=data.get("goal", ""),
                    steps=steps,
                    reasoning=data.get("reasoning", ""),
                    estimated_complexity=data.get("complexity", 1),
                    requires_confirmation=data.get("requires_confirmation", False),
                )
            except Exception as e:
                logger.error(f"Planner: Failed to parse plan: {e}")
                logger.error(f"Raw Response: {response}")
                
        # Fallback: empty plan
        if not data:
             logger.error("Planner: No JSON found in response")
             logger.error(f"Raw Response: {response}")

        return ExecutionPlan(
            goal="", 
            steps=[], 
            reasoning=f"Failed to parse plan from LLM response. Raw: {response[:500]}..."
        )


class ExecutorAgent(BaseAgent):
    """Executes tools and handles failures with recovery."""
    
    role = AgentRole.EXECUTOR
    
    def __init__(self, llm_provider: 'LLMProvider', tool_registry: ToolRegistry):
        super().__init__(llm_provider)
        self.tools = tool_registry
        
    def process(self, context: Dict[str, Any]) -> ToolResult:
        """Execute a tool from context."""
        tool_call = context.get("tool_call")
        if tool_call:
            return self.execute_tool(tool_call, context)
        return ToolResult(success=False, error="No tool call in context")
        
    def execute_tool(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        """Execute a single tool call."""
        tool = self.tools.get(tool_call.tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_call.tool_name}. Available: {[t.name for t in self.tools.all_tools]}",
            )
            
        try:
            # Run validation if present
            if tool.validator and not tool.validator(tool_call.parameters):
                return ToolResult(
                    success=False,
                    error=f"Validation failed for {tool_call.tool_name} parameters",
                )
                
            # Execute handler
            start_time = time.time()
            logger.info(f"Executor: Running {tool_call.tool_name} with {tool_call.parameters}")
            
            output = tool.handler(tool_call.parameters)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Get observation (current graph state) after execution
            get_state = context.get("get_graph_state")
            observation = get_state() if get_state else ""
            
            return ToolResult(
                success=True,
                output=output,
                observation=observation,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            logger.error(f"Executor: Tool {tool_call.tool_name} failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )
            
    def suggest_recovery(
        self, 
        failed_result: ToolResult, 
        original_call: ToolCall,
        context: Dict[str, Any]
    ) -> Optional[ToolCall]:
        """Suggest a recovery action for a failed tool."""
        prompt = EXECUTOR_PROMPT.format(
            tool_name=original_call.tool_name,
            error=failed_result.error,
            parameters=json.dumps(original_call.parameters, indent=2),
            graph_state=context.get("graph_state", "{}"),
        )
        
        logger.info(f"Executor: Requesting recovery for {original_call.tool_name}")
        response = self._call_llm_sync(
            f"The tool {original_call.tool_name} failed: {failed_result.error}",
            prompt
        )
        
        return self._parse_recovery(response)
        
    def apply_fix(
        self,
        original_call: ToolCall,
        suggestions: List[str],
        context: Dict[str, Any]
    ) -> Optional[ToolCall]:
        """Apply critic suggestions to fix a tool call."""
        if not suggestions:
            return None
            
        # Ask LLM to apply the suggestions
        prompt = f"""Apply these fixes to the tool call:

Original Tool: {original_call.tool_name}
Original Parameters: {json.dumps(original_call.parameters, indent=2)}
Suggestions: {suggestions}

Respond with updated parameters in JSON:
```json
{{"params": {{...}}}}
```
"""
        response = self._call_llm_sync(prompt, "You fix tool parameters based on suggestions.")
        data = self._extract_json(response)
        
        if data and "params" in data:
            return ToolCall(
                tool_name=original_call.tool_name,
                parameters=data["params"],
                description=f"Fixed: {original_call.description}",
                expected_outcome=original_call.expected_outcome,
            )
        return None
        
    def _parse_recovery(self, response: str) -> Optional[ToolCall]:
        """Parse recovery suggestion from LLM."""
        data = self._extract_json(response)
        
        if data and data.get("should_retry") and "recovery" in data:
            recovery = data["recovery"]
            return ToolCall(
                tool_name=recovery.get("tool", ""),
                parameters=recovery.get("params", {}),
                description=recovery.get("description", "Recovery attempt"),
            )
        return None


class CriticAgent(BaseAgent):
    """Evaluates execution results and validates design intent."""
    
    role = AgentRole.CRITIC
    
    def __init__(self, llm_provider: 'LLMProvider'):
        super().__init__(llm_provider)
        
    def process(self, context: Dict[str, Any]) -> CritiqueResult:
        """Evaluate the result of an execution."""
        result = context.get("result")
        if result:
            return self.evaluate(result, context)
        return CritiqueResult(approved=True)
        
    def evaluate(
        self, 
        result: ToolResult, 
        context: Dict[str, Any],
        expected_outcome: str = "",
    ) -> CritiqueResult:
        """Evaluate a tool result against design intent."""
        
        # Get current graph state for evaluation
        get_state = context.get("get_graph_state")
        current_state = get_state() if get_state else context.get("graph_state", "{}")
        
        prompt = CRITIC_PROMPT.format(
            success=result.success,
            tool_name=context.get("tool_name", "unknown"),
            output=str(result.output)[:500] if result.output else "None",
            observation=result.observation[:500] if result.observation else "None",
            expected=expected_outcome or "Not specified",
            graph_state=current_state[:1000] if current_state else "{}",
            user_request=context.get("user_request", "Not available"),
        )
        
        logger.info("Critic: Evaluating execution result")
        response = self._call_llm_sync(
            "Evaluate this execution result.",
            prompt
        )
        
        return self._parse_critique(response)
        
    def _parse_critique(self, response: str) -> CritiqueResult:
        """Parse critique from LLM response."""
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


class AgentOrchestrator:
    """
    Coordinates the multi-agent system.
    
    Flow: User Request → Planner → [Executor ↔ Critic] → Result
    """
    
    def __init__(
        self,
        llm_provider: 'LLMProvider',
        tool_registry: ToolRegistry,
        use_critic: bool = True,
        validate_design_intent: bool = True,
        max_retries: int = 3,
        on_plan_created: Optional[Callable[[ExecutionPlan], None]] = None,
        on_step_complete: Optional[Callable[[AgentStep], None]] = None,
        on_complete: Optional[Callable[[List[ToolResult]], None]] = None,
    ):
        self.llm = llm_provider
        self.tool_registry = tool_registry
        
        # Create agents (all use same LLM)
        self.planner = PlannerAgent(llm_provider, tool_registry)
        self.executor = ExecutorAgent(llm_provider, tool_registry)
        self.critic = CriticAgent(llm_provider) if use_critic else None
        
        self.validate_design_intent = validate_design_intent
        self.max_retries = max_retries
        
        # Callbacks
        self.on_plan_created = on_plan_created
        self.on_step_complete = on_step_complete
        self.on_complete = on_complete
        
        # Create ReAct loop
        self.react_loop = ReActLoop(
            executor=self.executor,
            critic=self.critic,
            max_iterations=max_retries,
            on_step=on_step_complete,
        )
        
        logger.info(f"AgentOrchestrator initialized (critic={'enabled' if use_critic else 'disabled'})")
        
    def process_request(
        self,
        user_request: str,
        graph_state: str = "",
        get_graph_state: Optional[Callable[[], str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent system.
        
        Args:
            user_request: Natural language request from user
            graph_state: Current state of the graph (JSON string)
            get_graph_state: Callable to get fresh graph state
            
        Returns:
            Dict with 'success', 'results', 'plan', and 'message'
        """
        context = {
            "user_request": user_request,
            "graph_state": graph_state,
            "get_graph_state": get_graph_state or (lambda: graph_state),
        }
        
        # 1. Create plan
        logger.info(f"Orchestrator: Processing request: {user_request[:100]}...")
        plan = self.planner.process(context)
        
        if self.on_plan_created:
            self.on_plan_created(plan)
            
        if not plan.steps:
            return {
                "success": False,
                "message": f"Could not create a plan for your request. {plan.reasoning}",
                "plan": plan,
                "results": [],
            }
            
        logger.info(f"Orchestrator: Plan created - {plan.goal} ({len(plan.steps)} steps)")
            
        # 2. Execute with ReAct loop
        results = self.react_loop.run(plan, context)
        
        # 3. Summarize
        all_success = all(r.success for r in results)
        
        if self.on_complete:
            self.on_complete(results)
            
        return {
            "success": all_success,
            "results": results,
            "plan": plan,
            "message": self._summarize_results(plan, results, user_request),
        }
        
    def _summarize_results(
        self, 
        plan: ExecutionPlan, 
        results: List[ToolResult],
        user_request: str,
    ) -> str:
        """Create a human-readable summary."""
        success_count = sum(1 for r in results if r.success)
        total = len(results)
        
        if success_count == total:
            # Full success
            return f"✅ {plan.goal}"
        elif success_count > 0:
            # Partial success
            failed = [r for r in results if not r.success]
            errors = ", ".join(r.error or "Unknown error" for r in failed[:2])
            return f"⚠️ Completed {success_count}/{total} steps. Issues: {errors}"
        else:
            # Complete failure
            first_error = results[0].error if results else "Unknown error"
            return f"❌ Failed: {first_error}"
            
    def get_available_tools_summary(self) -> str:
        """Get a summary of available tools for display."""
        return self.tool_registry.get_tools_description()
