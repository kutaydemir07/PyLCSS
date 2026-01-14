# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Model compilation utilities for PyLCSS.

This module handles the compilation of system graphs into executable Python
code models. It uses the GraphBuilder to convert node-based system models
into callable functions with proper input/output interfaces.
"""

from .model_builder import GraphBuilder
import logging

logger = logging.getLogger(__name__)

def get_compiled_code(widget):
    """
    Compile all system graphs in the widget into executable Python models.

    Iterates through all systems managed by the system manager, converts each
    graph into Python code using GraphBuilder, and returns a list of model
    dictionaries ready for execution or merging.

    Args:
        widget: Main application widget containing system_manager with systems

    Returns:
        list: List of model dictionaries with keys:
            - 'name': System name (str)
            - 'code': Generated Python function code (str)
            - 'inputs': List of input variable dictionaries
            - 'outputs': List of output variable dictionaries

    Error Handling:
        - Catches exceptions during compilation and creates error models
        - Error models contain minimal code with error message as comment
        - Empty inputs/outputs lists for failed compilations
        - Prints error messages to console for debugging

    Model Structure:
        Each model contains a function named 'system_function_{i}' where i
        is the system index. The function takes keyword arguments for inputs
        and returns a dictionary of outputs.
    """
    models = []
    product_name = widget.system_manager.product_name.text()
    
    # Track global names to prevent collisions between systems
    global_names = set()
    # Pre-populate with system function names
    for i in range(len(widget.system_manager.systems)):
        global_names.add(f"system_function_{i}")

    # Compile each system graph into a model
    for i, sys in enumerate(widget.system_manager.systems):
        try:
            # Use GraphBuilder to convert graph to executable code
            builder = GraphBuilder(sys['graph'])
            code, inputs, outputs = builder.build_system_model(
                function_name=f"system_function_{i}",
                global_reserved_names=global_names
            )

            # Create model dictionary
            model = {
                'name': sys['name'],
                'code': code,
                'inputs': inputs,
                'outputs': outputs
            }
            models.append(model)

        except Exception as e:
            # Handle compilation errors gracefully
            logger.exception("Error building model for system '%s'", sys.get('name', '<unknown>'))
            # Return a basic error model to maintain list structure
            model = {
                'name': sys['name'],
                'code': f"# Error building model: {e}",
                'inputs': [],
                'outputs': []
            }
            models.append(model)

    return models






