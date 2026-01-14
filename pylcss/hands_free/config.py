# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Configuration and settings for the Hands-Free Control System.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
HANDS_FREE_DIR = Path(__file__).parent
MODELS_DIR = HANDS_FREE_DIR.parent.parent / "models"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_PATH = MODELS_DIR / VOSK_MODEL_NAME
CONFIG_FILE = HANDS_FREE_DIR / "settings.json"


@dataclass
class HeadTrackingConfig:
    """Configuration for head tracking system (deprecated - camera tracking removed)."""
    enabled: bool = False  # Disabled - camera head tracking removed
    camera_index: int = 0
    sensitivity_x: float = 1.0  # Multiplier for horizontal movement (was 2.5)
    sensitivity_y: float = 0.8  # Multiplier for vertical movement (was 2.0)
    deadzone: float = 0.03  # Minimum movement threshold (0-1)
    smoothing: float = 0.5  # Smoothing factor (0 = no smoothing, 1 = max)
    invert_x: bool = False
    invert_y: bool = False
    blink_threshold: float = 0.21  # Eye aspect ratio threshold for blink
    blink_frames: int = 3  # Consecutive frames to trigger click
    

@dataclass
class VoiceControlConfig:
    """Configuration for voice control system."""
    enabled: bool = True
    model_path: str = str(VOSK_MODEL_PATH)
    sample_rate: int = 16000
    hotword: str = "hey computer"  # Wake word
    hotword_enabled: bool = False  # Always listen by default
    command_timeout: float = 5.0  # Seconds to wait for command after hotword
    feedback_enabled: bool = True  # Audio feedback on command recognition
    

@dataclass
class HandsFreeConfig:
    """Main configuration container for hands-free system."""
    head_tracking: HeadTrackingConfig = field(default_factory=HeadTrackingConfig)
    voice_control: VoiceControlConfig = field(default_factory=VoiceControlConfig)
    
    # General settings
    startup_enabled: bool = False  # Auto-start on PyLCSS launch
    overlay_enabled: bool = True  # Show status overlay
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        save_path = path or CONFIG_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved hands-free config to {save_path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HandsFreeConfig":
        """Load configuration from JSON file."""
        load_path = path or CONFIG_FILE
        
        if not load_path.exists():
            logger.info("No config file found, using defaults")
            return cls()
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            config = cls(
                head_tracking=HeadTrackingConfig(**data.get('head_tracking', {})),
                voice_control=VoiceControlConfig(**data.get('voice_control', {})),
                startup_enabled=data.get('startup_enabled', False),
                overlay_enabled=data.get('overlay_enabled', True),
            )
            logger.info(f"Loaded hands-free config from {load_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return cls()


# Voice command definitions for PyLCSS
VOICE_COMMANDS: Dict[str, Dict] = {
    # Mouse actions
    "click": {"action": "mouse_click", "button": "left"},
    "left click": {"action": "mouse_click", "button": "left"},
    "right click": {"action": "mouse_click", "button": "right"},
    "double click": {"action": "mouse_double_click"},
    "drag": {"action": "mouse_drag_toggle"},
    "drop": {"action": "mouse_drag_toggle"},
    
    # Scrolling
    "scroll up": {"action": "scroll", "direction": 3},
    "scroll down": {"action": "scroll", "direction": -3},
    "scroll left": {"action": "scroll_horizontal", "direction": -3},
    "scroll right": {"action": "scroll_horizontal", "direction": 3},
    "page up": {"action": "scroll", "direction": 10},
    "page down": {"action": "scroll", "direction": -10},
    
    # Navigation - Tab switching (FIXED indices to match MainWindow)
    # Tab 0: Modeling, Tab 1: CAD, Tab 2: Surrogate, Tab 3: Solution Space
    # Tab 4: Optimization, Tab 5: Sensitivity, Tab 6: Help
    "go to modeling": {"action": "switch_tab", "tab": 0},
    "go to cad": {"action": "switch_tab", "tab": 1},
    "go to surrogate": {"action": "switch_tab", "tab": 2},
    "go to solution space": {"action": "switch_tab", "tab": 3},
    "go to optimization": {"action": "switch_tab", "tab": 4},
    "go to sensitivity": {"action": "switch_tab", "tab": 5},
    "go to help": {"action": "switch_tab", "tab": 6},
    
    # Tab navigation shortcuts
    "next tab": {"action": "next_tab"},
    "previous tab": {"action": "previous_tab"},
    
    # Keyboard shortcuts
    "save": {"action": "keyboard", "keys": ["ctrl", "s"]},
    "save project": {"action": "keyboard", "keys": ["ctrl", "s"]},
    "load project": {"action": "keyboard", "keys": ["ctrl", "o"]},
    "undo": {"action": "keyboard", "keys": ["ctrl", "z"]},
    "redo": {"action": "keyboard", "keys": ["ctrl", "y"]},
    "copy": {"action": "keyboard", "keys": ["ctrl", "c"]},
    "paste": {"action": "keyboard", "keys": ["ctrl", "v"]},
    "cut": {"action": "keyboard", "keys": ["ctrl", "x"]},
    "select all": {"action": "keyboard", "keys": ["ctrl", "a"]},
    "delete": {"action": "keyboard", "keys": ["delete"]},
    "escape": {"action": "keyboard", "keys": ["escape"]},
    "enter": {"action": "keyboard", "keys": ["enter"]},
    "tab": {"action": "keyboard", "keys": ["tab"]},
    
    # PyLCSS specific actions
    "build model": {"action": "pylcss_action", "command": "build_model"},
    "compute solution space": {"action": "pylcss_action", "command": "generate_samples"},
    "run optimization": {"action": "pylcss_action", "command": "run_optimization"},
    "stop optimization": {"action": "pylcss_action", "command": "stop_optimization"},
    "generate samples": {"action": "pylcss_action", "command": "generate_samples"},
    "train surrogate": {"action": "pylcss_action", "command": "train_surrogate"},
    "run sensitivity": {"action": "pylcss_action", "command": "run_sensitivity"},
    "new project": {"action": "pylcss_action", "command": "new_project"},
    "open project": {"action": "pylcss_action", "command": "open_project"},
    "export results": {"action": "pylcss_action", "command": "export_results"},
    
    # ============================================
    # MODELING ENVIRONMENT COMMANDS
    # ============================================
    
    # Node creation
    "add input": {"action": "pylcss_action", "command": "add_input"},
    "create input": {"action": "pylcss_action", "command": "add_input"},
    "new input": {"action": "pylcss_action", "command": "add_input"},
    "add output": {"action": "pylcss_action", "command": "add_output"},
    "create output": {"action": "pylcss_action", "command": "add_output"},
    "new output": {"action": "pylcss_action", "command": "add_output"},
    "add function": {"action": "pylcss_action", "command": "add_function"},
    "create function": {"action": "pylcss_action", "command": "add_function"},
    "new function": {"action": "pylcss_action", "command": "add_function"},
    "add intermediate": {"action": "pylcss_action", "command": "add_intermediate"},
    "create intermediate": {"action": "pylcss_action", "command": "add_intermediate"},
    "new intermediate": {"action": "pylcss_action", "command": "add_intermediate"},
    "add design variable": {"action": "pylcss_action", "command": "add_input"},
    "add parameter": {"action": "pylcss_action", "command": "add_input"},
    "add quantity of interest": {"action": "pylcss_action", "command": "add_output"},
    "add qoi": {"action": "pylcss_action", "command": "add_output"},
    
    # System management
    "add system": {"action": "pylcss_action", "command": "add_system"},
    "create system": {"action": "pylcss_action", "command": "add_system"},
    "new system": {"action": "pylcss_action", "command": "add_system"},
    "remove system": {"action": "pylcss_action", "command": "remove_system"},
    "delete system": {"action": "pylcss_action", "command": "remove_system"},
    "rename system": {"action": "pylcss_action", "command": "rename_system"},
    "next system": {"action": "pylcss_action", "command": "next_system"},
    "previous system": {"action": "pylcss_action", "command": "previous_system"},
    "switch system": {"action": "pylcss_action", "command": "next_system"},
    
    # Graph operations
    "validate": {"action": "pylcss_action", "command": "validate_graph"},
    "validate graph": {"action": "pylcss_action", "command": "validate_graph"},
    "check graph": {"action": "pylcss_action", "command": "validate_graph"},
    "connect nodes": {"action": "pylcss_action", "command": "auto_connect"},
    "auto connect": {"action": "pylcss_action", "command": "auto_connect"},
    "clear graph": {"action": "pylcss_action", "command": "clear_graph"},
    "delete all nodes": {"action": "pylcss_action", "command": "clear_graph"},
    "select all nodes": {"action": "pylcss_action", "command": "select_all_nodes"},
    "delete selected": {"action": "pylcss_action", "command": "delete_selected"},
    
    # ============================================
    # CAD ENVIRONMENT COMMANDS
    # ============================================
    
    # Primitive creation
    "add box": {"action": "pylcss_action", "command": "cad_add_box"},
    "create box": {"action": "pylcss_action", "command": "cad_add_box"},
    "add cube": {"action": "pylcss_action", "command": "cad_add_box"},
    "add cylinder": {"action": "pylcss_action", "command": "cad_add_cylinder"},
    "create cylinder": {"action": "pylcss_action", "command": "cad_add_cylinder"},
    "add sphere": {"action": "pylcss_action", "command": "cad_add_sphere"},
    "create sphere": {"action": "pylcss_action", "command": "cad_add_sphere"},
    "add cone": {"action": "pylcss_action", "command": "cad_add_cone"},
    "create cone": {"action": "pylcss_action", "command": "cad_add_cone"},
    "add torus": {"action": "pylcss_action", "command": "cad_add_torus"},
    "create torus": {"action": "pylcss_action", "command": "cad_add_torus"},
    "add donut": {"action": "pylcss_action", "command": "cad_add_torus"},
    
    # CAD operations
    "add extrude": {"action": "pylcss_action", "command": "cad_add_extrude"},
    "extrude": {"action": "pylcss_action", "command": "cad_add_extrude"},
    "add fillet": {"action": "pylcss_action", "command": "cad_add_fillet"},
    "fillet": {"action": "pylcss_action", "command": "cad_add_fillet"},
    "add chamfer": {"action": "pylcss_action", "command": "cad_add_chamfer"},
    "chamfer": {"action": "pylcss_action", "command": "cad_add_chamfer"},
    "add boolean": {"action": "pylcss_action", "command": "cad_add_boolean"},
    "boolean": {"action": "pylcss_action", "command": "cad_add_boolean"},
    "add union": {"action": "pylcss_action", "command": "cad_add_union"},
    "union": {"action": "pylcss_action", "command": "cad_add_union"},
    "add cut": {"action": "pylcss_action", "command": "cad_add_cut"},
    "cut": {"action": "pylcss_action", "command": "cad_add_cut"},
    "add revolve": {"action": "pylcss_action", "command": "cad_add_revolve"},
    "revolve": {"action": "pylcss_action", "command": "cad_add_revolve"},
    
    # CAD execution
    "run cad": {"action": "pylcss_action", "command": "cad_execute"},
    "execute cad": {"action": "pylcss_action", "command": "cad_execute"},
    "build cad": {"action": "pylcss_action", "command": "cad_execute"},
    "update cad": {"action": "pylcss_action", "command": "cad_execute"},
    "export cad": {"action": "pylcss_action", "command": "cad_export"},
    "export stl": {"action": "pylcss_action", "command": "cad_export"},
    
    # ============================================
    # SOLUTION SPACE COMMANDS
    # ============================================
    
    "resample": {"action": "pylcss_action", "command": "resample"},
    "refine samples": {"action": "pylcss_action", "command": "resample"},
    "add plot": {"action": "pylcss_action", "command": "add_plot"},
    "new plot": {"action": "pylcss_action", "command": "add_plot"},
    "clear plots": {"action": "pylcss_action", "command": "clear_plots"},
    "remove plots": {"action": "pylcss_action", "command": "clear_plots"},
    "save plots": {"action": "pylcss_action", "command": "save_plots"},
    "export plots": {"action": "pylcss_action", "command": "save_plots"},
    "configure colors": {"action": "pylcss_action", "command": "configure_colors"},
    "change colors": {"action": "pylcss_action", "command": "configure_colors"},
    "view code": {"action": "pylcss_action", "command": "view_code"},
    "show code": {"action": "pylcss_action", "command": "view_code"},
    
    # Product family
    "compute product family": {"action": "pylcss_action", "command": "compute_family"},
    "add variant": {"action": "pylcss_action", "command": "add_variant"},
    "remove variant": {"action": "pylcss_action", "command": "remove_variant"},
    "edit variant": {"action": "pylcss_action", "command": "edit_variant"},
    
    # ADG (Attribute Dependency Graph)
    "generate graph": {"action": "pylcss_action", "command": "compute_adg"},
    "generate dependency graph": {"action": "pylcss_action", "command": "compute_adg"},
    "show dependencies": {"action": "pylcss_action", "command": "compute_adg"},
    
    # ============================================
    # SURROGATE TRAINING COMMANDS
    # ============================================
    
    "refresh nodes": {"action": "pylcss_action", "command": "refresh_nodes"},
    "update nodes": {"action": "pylcss_action", "command": "refresh_nodes"},
    "generate data": {"action": "pylcss_action", "command": "generate_training_data"},
    "create training data": {"action": "pylcss_action", "command": "generate_training_data"},
    "browse file": {"action": "pylcss_action", "command": "browse_data_file"},
    "load data file": {"action": "pylcss_action", "command": "browse_data_file"},
    "save surrogate": {"action": "pylcss_action", "command": "save_surrogate"},
    "attach surrogate": {"action": "pylcss_action", "command": "save_surrogate"},
    "stop training": {"action": "pylcss_action", "command": "stop_training"},
    "cancel training": {"action": "pylcss_action", "command": "stop_training"},
    "adaptive training": {"action": "pylcss_action", "command": "adaptive_training"},
    "active learning": {"action": "pylcss_action", "command": "adaptive_training"},
    
    # ============================================
    # OPTIMIZATION COMMANDS
    # ============================================
    
    "optimization settings": {"action": "pylcss_action", "command": "optimization_settings"},
    "advanced settings": {"action": "pylcss_action", "command": "optimization_settings"},
    "algorithm settings": {"action": "pylcss_action", "command": "optimization_settings"},
    
    # ============================================
    # SENSITIVITY ANALYSIS COMMANDS
    # ============================================
    
    "refresh outputs": {"action": "pylcss_action", "command": "refresh_outputs"},
    "update outputs": {"action": "pylcss_action", "command": "refresh_outputs"},
    "export sensitivity": {"action": "pylcss_action", "command": "export_sensitivity"},
    "save sensitivity": {"action": "pylcss_action", "command": "export_sensitivity"},
    
    # ============================================
    # CONTROL COMMANDS
    # ============================================
    
    "stop": {"action": "control", "command": "pause_tracking"},
    "pause": {"action": "control", "command": "pause_tracking"},
    "resume": {"action": "control", "command": "resume_tracking"},
    "start": {"action": "control", "command": "resume_tracking"},
    "center": {"action": "control", "command": "center_cursor"},
    
    # Dictation mode
    "start dictation": {"action": "control", "command": "start_dictation"},
    "stop dictation": {"action": "control", "command": "stop_dictation"},
    
    # Window control
    "minimize": {"action": "window", "command": "minimize"},
    "maximize": {"action": "window", "command": "maximize"},
    "close window": {"action": "window", "command": "close"},
}


# Fuzzy matching alternatives for better recognition
COMMAND_ALIASES: Dict[str, str] = {
    # Click variations
    "select": "click",
    "press": "click",
    "tap": "click",
    "pick": "click",
    "clicking": "click",
    "pressed": "click",
    
    # Right click variations
    "right": "right click",
    "context menu": "right click",
    "right clicking": "right click",
    
    # Double click variations
    "double": "double click",
    "twice": "double click",
    "double clicking": "double click",
    
    # Scroll variations
    "up": "scroll up",
    "down": "scroll down",
    "scrolling up": "scroll up",
    "scrolling down": "scroll down",
    "go up": "scroll up",
    "go down": "scroll down",
    
    # Tab navigation variations
    "modeling tab": "go to modeling",
    "cad tab": "go to cad",
    "solution tab": "go to solution space",
    "surrogate tab": "go to surrogate",
    "optimization tab": "go to optimization",
    "sensitivity tab": "go to sensitivity",
    "model": "go to modeling",
    "design": "go to cad",
    "samples": "go to solution space",
    "training": "go to surrogate",
    "optimize": "go to optimization",
    "analysis": "go to sensitivity",
    
    # Tab navigation
    "forward": "next tab",
    "backward": "previous tab",
    "back": "previous tab",
    "prev": "previous tab",
    
    # Control variations  
    "hold": "pause",
    "wait": "pause",
    "freeze": "pause",
    "stop tracking": "pause",
    "continue": "resume",
    "go": "resume",
    "unfreeze": "resume",
    "recenter": "calibrate",
    "reset": "calibrate",
    
    # Keyboard variations
    "saving": "save",
    "saved": "save",
    "undoing": "undo",
    "redoing": "redo",
    "copying": "copy",
    "pasting": "paste",
    "deleting": "delete",
    "removing": "delete",
    "cancel": "escape",
    "close": "escape",
    "confirm": "enter",
    "okay": "enter",
    "ok": "enter",
    "submit": "enter",
    
    # PyLCSS action variations
    "build": "build model",
    "compile": "build model",
    "transfer": "build model",
    "help": "go to help",
    "documentation": "go to help",
    "load": "load project",
    "open": "open project",
}
