# src/__init__.py
import os
import sys
import cv2

# Get the directory of the src folder
current_dir = os.path.dirname(__file__)

# Construct the path to the src directory
src_path = os.path.abspath(current_dir)

# Ensure the src path is added to sys.path only if it's not already there
if src_path not in sys.path:
    sys.path.append(src_path)

# Now you can import your modules from the src directory
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
