# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

NODE_REGISTRY = {}

def register_node(cls):
    """Decorator to register a node class."""
    NODE_REGISTRY[cls.__identifier__] = cls
    return cls
