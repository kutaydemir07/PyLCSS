# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
# Main entry point for the PyLCSS application.

This script serves as a compatibility wrapper that calls the main function
from the pylcss package. For new installations, use 'pylcss' command directly.
"""

import os
import sys

# Add the parent directory to Python path so we can import pylcss
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pylcss.main import main

if __name__ == "__main__":
    main()







