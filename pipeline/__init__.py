"""Gram-positive antibacterial drug-target discovery pipeline."""

__version__ = "2.1.0"
__author__ = "Advait Suratran"

from pipeline.config import Config
from pipeline.cli import main, parse_args, run_pipeline

__all__ = ["Config", "main", "parse_args", "run_pipeline"]
