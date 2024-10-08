# programs/__init__.py
import os
import sys

# Get the current directory (where this __init__.py is located)
current_dir = os.path.dirname(__file__)

# Construct the path to the root of the project
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)
