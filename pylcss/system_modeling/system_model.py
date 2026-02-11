# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import tempfile
import importlib.util
import os
import logging
from typing import List, Dict, Any, Optional, Callable
from pylcss.config import TEMP_MODELS_DIR

logger = logging.getLogger(__name__)

class SystemModel:
    """
    Represents a compiled system model that can be safely passed between components.
    Holds the callable function and metadata instead of raw code strings.
    """

    def __init__(self, name: str, system_function: Callable[..., Dict[str, Any]], 
                 inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]], 
                 source_code: Optional[str] = None) -> None:
        """
        Initialize a SystemModel.

        Args:
            name: Name of the model
            system_function: Callable function that takes **kwargs and returns dict
            inputs: List of input variable dicts [{'name', 'unit', 'min', 'max'}, ...]
            outputs: List of output variable dicts [{'name', 'unit', 'req_min', 'req_max'}, ...]
            source_code: Optional source code string for debugging/serialization
        """
        self.name = name
        self.system_function = system_function
        self.inputs = inputs
        self.outputs = outputs
        self.source_code = source_code
        
        # Use the system function directly (no JIT compilation)
        self.fast_function = self.system_function
        
        # Try to use Numba for JIT compilation if available
        try:
            from numba import jit
            # Attempt to compile the function
            # Note: This requires the system_function to be compatible with Numba's nopython mode
            # or at least object mode. Since generated code uses numpy, it has a good chance.
            # We use forceobj=True to allow python objects (like dicts) which are returned by system_function
            self.fast_function = jit(forceobj=True)(self.system_function)
        except ImportError:
            pass # Numba not installed, fallback to standard python
        except Exception as e:
            logger.warning(f"Numba JIT compilation failed: {e}. Using standard Python execution.")
            pass

    @classmethod
    def from_code_string(cls, name: str, code_string: str, 
                        inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]) -> 'SystemModel':
        """
        Create a SystemModel from a code string by safely compiling it.
        Persists the code to a temporary file to enable multiprocessing pickling.

        Args:
            name: Name of the model
            code_string: Python code string containing system_function
            inputs: List of input dicts
            outputs: List of output dicts

        Returns:
            SystemModel instance
        """
        # Use centralized temp directory
        temp_dir = TEMP_MODELS_DIR
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a persistent file with unique name
        import uuid
        filename = f"model_{uuid.uuid4().hex}.py"
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(code_string)

        # Basic security check
        dangerous_terms = ['os.system', 'subprocess', 'shutil', 'sys.modules', 'eval(', 'exec(']
        if any(term in code_string for term in dangerous_terms):
            logger.warning(f"Potential security risk detected in model code for {name}. Proceeding with caution.")

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("temp_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the system_function
            system_function = None
            for attr_name in dir(module):
                if attr_name.startswith('system_function') and callable(getattr(module, attr_name)):
                    system_function = getattr(module, attr_name)
                    break
            if system_function is None:
                raise AttributeError("system_function not found in generated code")

            return cls(name, system_function, inputs, outputs, code_string)

        except Exception as e:
            # Only clean up if compilation FAILED
            try:
                os.unlink(filepath)
            except:
                pass
            raise e

    @classmethod
    def from_models(cls, models: List[Dict[str, Any]], merged_name: str = "Merged") -> 'SystemModel':
        """
        Create a merged SystemModel from multiple individual models.

        Args:
            models: List of model dicts [{'name', 'code', 'inputs', 'outputs'}, ...]
            merged_name: Name for the merged model

        Returns:
            SystemModel instance
        """
        if len(models) == 1:
            # Single model, no merging needed
            model = models[0]
            return cls.from_code_string(model['name'], model['code'], model['inputs'], model['outputs'])

        # Multiple models - delegate to robust merge logic
        from pylcss.system_modeling.model_merge import create_merged_model
        
        try:
            merged_dict = create_merged_model(models)
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            raise e
            
        return cls.from_code_string(
            merged_name,
            merged_dict['code'],
            merged_dict['inputs'],
            merged_dict['outputs']
        )

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        """Call the system function with the given arguments."""
        try:
            return self.fast_function(**kwargs)
        except Exception:
            # If JIT compiled version fails, fall back to original function
            return self.system_function(**kwargs)

    def get_input_names(self) -> List[str]:
        """Get list of input variable names."""
        return [inp['name'] for inp in self.inputs]

    def get_output_names(self) -> List[str]:
        """Get list of output variable names."""
        return [out['name'] for out in self.outputs]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'source_code': self.source_code
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemModel':
        """Create from dictionary (requires recompilation)."""
        return cls.from_code_string(
            data['name'],
            data['source_code'],
            data['inputs'],
            data['outputs']
        )






