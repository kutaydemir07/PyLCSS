# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Workflow Recorder - Records and replays successful action sequences.

Features:
- Automatic recording of successful agent runs
- Named workflows with natural language triggers
- Replay by voice command ("do the gear workflow")
- Success tracking for workflow quality
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .agent_orchestrator import ExecutorAgent
    from .agent_core import ToolCall, ToolResult

logger = logging.getLogger(__name__)

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    tool_name: str
    parameters: Dict[str, Any]
    description: str = ""
    expected_outcome: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowStep':
        return cls(
            tool_name=data.get("tool_name", ""),
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )
    
    
@dataclass
class Workflow:
    """A recorded workflow."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)
    trigger_phrases: List[str] = field(default_factory=list)  # Natural language triggers
    success_count: int = 0  # How many times this was used successfully
    failure_count: int = 0
    source_request: str = ""  # Original user request that created this
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.trigger_phrases:
            # Auto-generate trigger phrases from name
            self.trigger_phrases = [
                self.name.lower(),
                f"do {self.name.lower()}",
                f"run {self.name.lower()}",
                f"{self.name.lower()} workflow",
            ]
            
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "trigger_phrases": self.trigger_phrases,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "source_request": self.source_request,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            tags=data.get("tags", []),
            trigger_phrases=data.get("trigger_phrases", []),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            source_request=data.get("source_request", ""),
        )
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class WorkflowLibrary:
    """Stores and manages workflows."""
    
    def __init__(self, storage_dir: Path = WORKFLOWS_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, Workflow] = {}
        self._load_all()
        
    def _load_all(self):
        """Load all workflows from storage."""
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    wf = Workflow.from_dict(data)
                    self._workflows[wf.id] = wf
                    logger.debug(f"Loaded workflow: {wf.name}")
            except Exception as e:
                logger.warning(f"Failed to load workflow {path}: {e}")
                
        logger.info(f"WorkflowLibrary: Loaded {len(self._workflows)} workflows")
                
    def _save(self, workflow: Workflow):
        """Save a workflow to storage."""
        workflow.updated_at = datetime.now().isoformat()
        path = self.storage_dir / f"{workflow.id}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(workflow.to_dict(), f, indent=2)
        logger.debug(f"Saved workflow: {workflow.name}")
            
    def add(self, workflow: Workflow) -> None:
        """Add a new workflow."""
        self._workflows[workflow.id] = workflow
        self._save(workflow)
        logger.info(f"Added workflow: {workflow.name} ({len(workflow.steps)} steps)")
        
    def update(self, workflow: Workflow) -> None:
        """Update an existing workflow."""
        if workflow.id in self._workflows:
            self._workflows[workflow.id] = workflow
            self._save(workflow)
            
    def get(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)
        
    def get_by_name(self, name: str) -> Optional[Workflow]:
        """Get a workflow by name (case-insensitive)."""
        name_lower = name.lower()
        for wf in self._workflows.values():
            if wf.name.lower() == name_lower:
                return wf
        return None
        
    def find_by_phrase(self, phrase: str) -> Optional[Workflow]:
        """Find a workflow matching a trigger phrase."""
        phrase_lower = phrase.lower().strip()
        
        best_match = None
        best_score = 0
        
        for wf in self._workflows.values():
            # Check trigger phrases
            for trigger in wf.trigger_phrases:
                trigger_lower = trigger.lower()
                
                # Exact match
                if trigger_lower == phrase_lower:
                    return wf
                    
                # Contains match (prefer longer matches)
                if trigger_lower in phrase_lower or phrase_lower in trigger_lower:
                    score = len(trigger_lower)
                    if score > best_score:
                        best_match = wf
                        best_score = score
                        
            # Check name
            if wf.name.lower() in phrase_lower:
                score = len(wf.name)
                if score > best_score:
                    best_match = wf
                    best_score = score
                    
        return best_match
        
    def list_all(self) -> List[Workflow]:
        """List all workflows sorted by success rate."""
        return sorted(
            self._workflows.values(),
            key=lambda w: (w.success_rate, w.success_count),
            reverse=True
        )
        
    def list_by_tag(self, tag: str) -> List[Workflow]:
        """List workflows with a specific tag."""
        return [wf for wf in self._workflows.values() if tag in wf.tags]
        
    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            path = self.storage_dir / f"{workflow_id}.json"
            if path.exists():
                path.unlink()
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False
        
    def record_success(self, workflow_id: str) -> None:
        """Record a successful workflow execution."""
        wf = self.get(workflow_id)
        if wf:
            wf.success_count += 1
            self._save(wf)
            
    def record_failure(self, workflow_id: str) -> None:
        """Record a failed workflow execution."""
        wf = self.get(workflow_id)
        if wf:
            wf.failure_count += 1
            self._save(wf)
            
    @property
    def count(self) -> int:
        return len(self._workflows)


class WorkflowRecorder:
    """Records actions into workflows."""
    
    def __init__(self, library: WorkflowLibrary, auto_save: bool = True):
        self.library = library
        self.auto_save = auto_save
        self._recording = False
        self._current_steps: List[WorkflowStep] = []
        self._recording_name = ""
        self._source_request = ""
        
    def start_recording(self, name: str = "", source_request: str = "") -> None:
        """Start recording a new workflow."""
        self._recording = True
        self._current_steps = []
        self._recording_name = name or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._source_request = source_request
        logger.info(f"Started recording workflow: {self._recording_name}")
        
    def record_step(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        description: str = "",
        expected_outcome: str = "",
    ) -> None:
        """Record a step if recording is active."""
        if self._recording:
            step = WorkflowStep(
                tool_name=tool_name,
                parameters=parameters,
                description=description,
                expected_outcome=expected_outcome,
            )
            self._current_steps.append(step)
            logger.debug(f"Recorded step: {tool_name}")
            
    def stop_recording(
        self,
        save: bool = True,
        description: str = "",
        tags: List[str] = None,
        trigger_phrases: List[str] = None,
    ) -> Optional[Workflow]:
        """Stop recording and optionally save the workflow."""
        if not self._recording:
            return None
            
        self._recording = False
        
        if not save or not self._current_steps:
            logger.info("Recording stopped without saving")
            self._current_steps = []
            return None
            
        workflow = Workflow(
            id=str(uuid.uuid4())[:8],
            name=self._recording_name,
            description=description or f"Auto-recorded from: {self._source_request[:100]}",
            steps=self._current_steps.copy(),
            tags=tags or ["auto-recorded"],
            trigger_phrases=trigger_phrases or [],
            source_request=self._source_request,
        )
        
        self.library.add(workflow)
        self._current_steps = []
        
        logger.info(f"Saved workflow: {workflow.name} ({len(workflow.steps)} steps)")
        return workflow
        
    def cancel_recording(self) -> None:
        """Cancel the current recording without saving."""
        self._recording = False
        self._current_steps = []
        logger.info("Recording cancelled")
        
    def is_recording(self) -> bool:
        return self._recording
        
    @property
    def current_step_count(self) -> int:
        return len(self._current_steps)
        
    def auto_record_from_plan(
        self,
        plan: 'ExecutionPlan',
        results: List['ToolResult'],
        user_request: str,
    ) -> Optional[Workflow]:
        """
        Automatically create a workflow from a successful plan execution.
        
        Only saves if auto_save is enabled and all steps succeeded.
        """
        if not self.auto_save:
            return None
            
        # Only auto-save fully successful executions with 2+ steps
        if len(results) < 2 or not all(r.success for r in results):
            return None
            
        from .agent_core import ExecutionPlan
        
        # Create workflow from plan
        steps = []
        for tool_call in plan.steps:
            steps.append(WorkflowStep(
                tool_name=tool_call.tool_name,
                parameters=tool_call.parameters,
                description=tool_call.description,
                expected_outcome=tool_call.expected_outcome,
            ))
            
        # Generate a name from the goal
        name = self._generate_name_from_goal(plan.goal)
        
        workflow = Workflow(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=plan.goal,
            steps=steps,
            tags=["auto-saved"],
            source_request=user_request,
        )
        
        self.library.add(workflow)
        logger.info(f"Auto-saved workflow: {name}")
        return workflow
        
    def _generate_name_from_goal(self, goal: str) -> str:
        """Generate a workflow name from a goal description."""
        # Simple extraction of key words
        words = goal.lower().split()
        
        # Remove common words
        skip_words = {"a", "the", "an", "with", "and", "or", "to", "for", "of", "in", "on"}
        key_words = [w for w in words if w not in skip_words][:3]
        
        if key_words:
            return "_".join(key_words)
        return f"workflow_{datetime.now().strftime('%H%M%S')}"


class WorkflowPlayer:
    """Plays back recorded workflows."""
    
    def __init__(self, executor_agent: 'ExecutorAgent', library: WorkflowLibrary):
        self.executor = executor_agent
        self.library = library
        
    def play(
        self, 
        workflow: Workflow, 
        context: Dict[str, Any],
        on_step: Optional[callable] = None,
    ) -> List['ToolResult']:
        """
        Play back a workflow.
        
        Args:
            workflow: The workflow to play
            context: Execution context (should include get_graph_state)
            on_step: Optional callback for each step
            
        Returns:
            List of ToolResults from execution
        """
        from .agent_core import ToolCall, ToolResult
        
        results = []
        logger.info(f"Playing workflow: {workflow.name} ({len(workflow.steps)} steps)")
        
        for i, step in enumerate(workflow.steps):
            tool_call = ToolCall(
                tool_name=step.tool_name,
                parameters=step.parameters,
                description=step.description,
                expected_outcome=step.expected_outcome,
            )
            
            if on_step:
                on_step(i, step, None)
                
            result = self.executor.execute_tool(tool_call, context)
            results.append(result)
            
            if on_step:
                on_step(i, step, result)
            
            if not result.success:
                logger.warning(f"Workflow step {i+1} failed: {step.tool_name}")
                break
                
            # Update context with new state
            get_state = context.get("get_graph_state")
            if get_state:
                context['graph_state'] = get_state()
            
        # Update workflow statistics
        if all(r.success for r in results) and len(results) == len(workflow.steps):
            self.library.record_success(workflow.id)
            logger.info(f"Workflow completed successfully: {workflow.name}")
        else:
            self.library.record_failure(workflow.id)
            logger.warning(f"Workflow failed: {workflow.name}")
            
        return results
        
    def play_by_name(
        self, 
        name: str, 
        context: Dict[str, Any],
        on_step: Optional[callable] = None,
    ) -> Optional[List['ToolResult']]:
        """Play a workflow by name."""
        workflow = self.library.get_by_name(name)
        if workflow:
            return self.play(workflow, context, on_step)
        logger.warning(f"Workflow not found: {name}")
        return None
        
    def play_by_phrase(
        self, 
        phrase: str, 
        context: Dict[str, Any],
        on_step: Optional[callable] = None,
    ) -> Optional[List['ToolResult']]:
        """Play a workflow matching a trigger phrase."""
        workflow = self.library.find_by_phrase(phrase)
        if workflow:
            return self.play(workflow, context, on_step)
        return None
