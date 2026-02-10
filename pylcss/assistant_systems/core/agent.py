# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Agent Core - Base classes for agentic AI system.

Implements ReAct pattern: Think → Act → Observe → Reflect
Supports multi-agent collaboration with Planner, Executor, and Critic roles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import logging
import json
import re

if TYPE_CHECKING:
    from pylcss.assistant_systems.services.llm import LLMProvider

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles in the multi-agent system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"


@dataclass
class AgentThought:
    """A reasoning step from an agent."""
    content: str
    confidence: float = 1.0  # 0-1 scale
    needs_clarification: bool = False


@dataclass
class ToolCall:
    """A tool invocation request."""
    tool_name: str
    parameters: Dict[str, Any]
    description: str = ""
    expected_outcome: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "params": self.parameters,
            "description": self.description,
            "expected": self.expected_outcome,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        return cls(
            tool_name=data.get("tool", ""),
            parameters=data.get("params", {}),
            description=data.get("description", ""),
            expected_outcome=data.get("expected", ""),
        )


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    observation: str = ""  # Human-readable state description
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "observation": self.observation,
        }


@dataclass
class AgentStep:
    """A single step in the agent loop."""
    thought: AgentThought
    action: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    iteration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought.content,
            "action": self.action.to_dict() if self.action else None,
            "result": self.result.to_dict() if self.result else None,
            "iteration": self.iteration,
        }


@dataclass
class ExecutionPlan:
    """A plan created by the Planner agent."""
    goal: str
    steps: List[ToolCall]
    reasoning: str
    estimated_complexity: int = 1  # 1-5 scale
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "complexity": self.estimated_complexity,
            "requires_confirmation": self.requires_confirmation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionPlan':
        steps = [ToolCall.from_dict(s) for s in data.get("steps", [])]
        return cls(
            goal=data.get("goal", ""),
            steps=steps,
            reasoning=data.get("reasoning", ""),
            estimated_complexity=data.get("complexity", 1),
            requires_confirmation=data.get("requires_confirmation", False),
        )


@dataclass 
class CritiqueResult:
    """Result from the Critic agent."""
    approved: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    design_feedback: str = ""  # Feedback on design intent
    severity: int = 1  # 1-5 scale
    retry_with: Optional[ExecutionPlan] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "design_feedback": self.design_feedback,
            "severity": self.severity,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    role: AgentRole
    
    def __init__(self, llm_provider: 'LLMProvider', system_prompt: str = ""):
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.history: List[AgentStep] = []
        
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Any:
        """Process a request with the given context."""
        pass
    
    def reset(self):
        """Clear agent history."""
        self.history = []
        
    def _call_llm_sync(self, message: str, system_prompt: str = "",
                        **kwargs) -> str:
        """Call the LLM provider synchronously."""
        try:
            prompt = system_prompt or self.system_prompt
            # Clear message history for agentic calls - each agent call should be fresh
            # This prevents accumulation of messages from previous calls/memory
            self.llm.clear_history()
            result = self.llm.chat(message, system_prompt=prompt, **kwargs)
            return result.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
            
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling comments and code blocks."""
        # 1. Try to find JSON in fenced code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        content = json_match.group(1).strip() if json_match else response.strip()
        
        # 2. Extract the first valid-looking JSON object if not already cleaned
        if not json_match:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]
            else:
                return None

        # 3. Strip comments (standard json.loads doesn't support them)
        # Strip single-line comments // ...
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Strip multi-line comments /* ... */
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 4. Handle Truncation: Try to fix incomplete JSON by adding closing braces/brackets
            try:
                temp_content = content
                # Keep track of open characters
                stack = []
                in_string = False
                escape = False
                
                for i, char in enumerate(temp_content):
                    if char == '"' and not escape:
                        in_string = not in_string
                    if in_string:
                        if char == '\\':
                            escape = not escape
                        else:
                            escape = False
                        continue
                    
                    if char == '{':
                        stack.append('}')
                    elif char == '[':
                        stack.append(']')
                    elif char == '}':
                        if stack and stack[-1] == '}':
                            stack.pop()
                    elif char == ']':
                        if stack and stack[-1] == ']':
                            stack.pop()
                
                # Close string if needed
                if in_string:
                    temp_content += '"'
                    
                # Close objects/arrays in reverse order
                while stack:
                    temp_content += stack.pop()
                
                return json.loads(temp_content)
            except Exception:
                pass

            # Last resort: existing find/rfind logic
            try:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
            
        return None


class ReActLoop:
    """
    Implements the ReAct (Reason + Act) pattern.
    
    Loop: Think → Act → Observe → Reflect → (repeat or finish)
    """
    
    def __init__(
        self,
        executor: 'ExecutorAgent',
        critic: Optional['CriticAgent'] = None,
        max_iterations: int = 3,
        on_step: Optional[Callable[[AgentStep], None]] = None,
        on_retry: Optional[Callable[[str, ToolCall], None]] = None,
    ):
        self.executor = executor
        self.critic = critic
        self.max_iterations = max_iterations
        self.on_step = on_step
        self.on_retry = on_retry
        
    def run(self, plan: ExecutionPlan, context: Dict[str, Any]) -> List[ToolResult]:
        """Execute a plan with observation and self-correction."""
        results = []
        
        for i, tool_call in enumerate(plan.steps):
            iteration = 0
            current_call = tool_call
            success = False
            last_result = None
            
            while iteration < self.max_iterations and not success:
                # 1. Execute
                logger.info(f"ReAct: Executing step {i+1}/{len(plan.steps)}, iteration {iteration+1}")
                result = self.executor.execute_tool(current_call, context)
                last_result = result
                
                step = AgentStep(
                    thought=AgentThought(f"Executing: {current_call.description or current_call.tool_name}"),
                    action=current_call,
                    result=result,
                    iteration=iteration,
                )
                
                if self.on_step:
                    self.on_step(step)
                
                # 2. Observe & Critique
                if result.success:
                    if self.critic:
                        critique = self.critic.evaluate(
                            result, 
                            context,
                            expected_outcome=current_call.expected_outcome,
                            step_num=i + 1,
                            total_steps=len(plan.steps)
                        )
                        if critique.approved:
                            success = True
                            logger.info(f"ReAct: Step approved by critic")
                        elif not current_call.parameters:
                            # No-param tools (execute_cad, validate_model) that
                            # succeeded cannot be meaningfully retried — approve.
                            success = True
                            logger.info(f"ReAct: Auto-approving no-param tool that succeeded")
                        else:
                            # Self-correct based on critique
                            logger.info(f"ReAct: Critic rejected - {critique.issues}")
                            if self.on_retry:
                                self.on_retry(f"Critique: {critique.issues}", current_call)
                                
                            if critique.retry_with and critique.retry_with.steps:
                                current_call = critique.retry_with.steps[0]
                            else:
                                # Try to fix based on suggestions
                                fixed_call = self.executor.apply_fix(
                                    current_call, 
                                    critique.suggestions,
                                    context
                                )
                                if fixed_call:
                                    current_call = fixed_call
                            iteration += 1
                    else:
                        success = True
                else:
                    # Error occurred - try to recover
                    logger.warning(f"ReAct: Tool failed - {result.error}")
                    if self.on_retry:
                        self.on_retry(f"Error: {result.error}", current_call)
                    
                    # Pass available tools to recovery if possible
                    if hasattr(self.executor, 'tools'):
                         tool_names = [t.name for t in self.executor.tools.all_tools]
                         context['available_tools'] = tool_names
                        
                    recovery = self.executor.suggest_recovery(result, current_call, context)
                    if recovery:
                        current_call = recovery
                    iteration += 1
                    
            results.append(last_result)
            
            # Update context with new state
            context['last_result'] = last_result.to_dict() if last_result else None
            context['step_index'] = i + 1
            
            # If a step failed after all retries, stop execution
            if not success and last_result and not last_result.success:
                logger.error(f"ReAct: Step {i+1} failed after {iteration} attempts")
                break
            
        return results


# Forward declarations for type hints
class ExecutorAgent(BaseAgent):
    """Placeholder - actual implementation in agent_orchestrator.py"""
    pass


class CriticAgent(BaseAgent):
    """Placeholder - actual implementation in agent_orchestrator.py"""
    pass
