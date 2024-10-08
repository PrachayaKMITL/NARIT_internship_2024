# src/__init__.py

import os
import sys

# Get the directory of the src folder
src_path = os.path.abspath(os.path.dirname(__file__))  # This will get the absolute path of the src folder

# Add the src directory to the sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

# Optionally, you can import your modules
from .ClassPrediction import *
from .ConstructDataset import *
from .feature_extraction import *
from .ModelTraining import *
from .preprocessing import *
from .TotalCalculation import *

# Specify what is available when the package is imported
__all__ = [
    "ClassPrediction",
    "ConstructDataset",
    "feature_extraction",
    "ModelTraining",
    "preprocessing",
    "TotalCalculation",
]
