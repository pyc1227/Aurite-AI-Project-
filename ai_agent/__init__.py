"""
AI Macro Analysis Agent Package
A comprehensive AI agent for quarterly market predictions using macro analysis.
"""

__version__ = "1.0.0"
__author__ = "Macro Analysis AI Team"

from .agent import MacroAnalysisAgent
from .config import Config
from .database import NeonDBManager
from .feature_engineer import FeatureEngineer
from .model_manager import ModelManager
from .openai_client import OpenAIClient

__all__ = [
    "MacroAnalysisAgent",
    "Config",
    "NeonDBManager",
    "FeatureEngineer",
    "ModelManager",
    "OpenAIClient",
]
