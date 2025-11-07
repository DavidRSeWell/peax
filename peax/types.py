"""Type definitions for the pursuer-evader environment."""

from typing import NamedTuple, Dict
import jax.numpy as jnp
from chex import Array, PRNGKey


class AgentState(NamedTuple):
    """State of a single agent with double integrator dynamics.

    Attributes:
        position: (x, y) position
        velocity: (vx, vy) velocity
    """
    position: Array  # shape (2,)
    velocity: Array  # shape (2,)


class EnvState(NamedTuple):
    """Complete state of the pursuer-evader environment.

    Attributes:
        pursuer: State of the pursuer agent
        evader: State of the evader agent
        time: Current timestep
    """
    pursuer: AgentState
    evader: AgentState
    time: int


class EnvParams(NamedTuple):
    """Parameters for the pursuer-evader environment.

    Attributes:
        max_steps: Maximum number of timesteps
        dt: Timestep duration
        capture_radius: Distance threshold for capture
        pursuer_mass: Mass of pursuer
        evader_mass: Mass of evader
        max_force: Maximum force that can be applied
        boundary_size: Size of the boundary
    """
    max_steps: int = 200
    dt: float = 0.1
    capture_radius: float = 0.5
    pursuer_mass: float = 1.0
    evader_mass: float = 1.0
    max_force: float = 10.0
    boundary_size: float = 10.0


class Observation(NamedTuple):
    """Observation for each agent.

    For the pursuer, this includes its own state and the evader's state.
    For the evader, this includes its own state and the pursuer's state.
    """
    own_position: Array  # shape (2,)
    own_velocity: Array  # shape (2,)
    other_position: Array  # shape (2,)
    other_velocity: Array  # shape (2,)
    time_remaining: float
