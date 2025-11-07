"""PEAX: Pursuer-Evader Environment for Reinforcement Learning in JAX."""

from peax.environment import PursuerEvaderEnv
from peax.types import AgentState, EnvState, EnvParams, Observation
from peax.boundaries import (
    SquareBoundary,
    TriangleBoundary,
    CircleBoundary,
    create_boundary,
)

__version__ = "0.1.0"

__all__ = [
    "PursuerEvaderEnv",
    "AgentState",
    "EnvState",
    "EnvParams",
    "Observation",
    "SquareBoundary",
    "TriangleBoundary",
    "CircleBoundary",
    "create_boundary",
]
