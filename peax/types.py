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
        wall_penalty_coef: Coefficient for wall proximity penalty (0.0 = disabled)
        velocity_reward_coef: Coefficient for velocity reward (0.0 = disabled)
    """
    max_steps: int = 200
    dt: float = 0.1
    capture_radius: float = 0.5
    pursuer_mass: float = 1.0
    evader_mass: float = 1.0
    max_force: float = 5.0
    boundary_size: float = 10.0
    wall_penalty_coef: float = 0.0
    velocity_reward_coef: float = 0.0


class Observation(NamedTuple):
    """Observation for each agent using relative coordinates.

    This representation is translation and rotation invariant, making
    learning easier and more focused on the pursuit-evasion dynamics
    rather than absolute positions.

    For each agent, observations are relative to itself:
    - relative_position: opponent position - own position (vector pointing to opponent)
    - relative_velocity: opponent velocity - own velocity (closing velocity)
    - own_velocity: own velocity (needed for control)
    - time_remaining: normalized time left in episode
    - agent_id: 0.0 for pursuer, 1.0 for evader (CRITICAL for shared policy self-play!)
    """
    relative_position: Array  # shape (2,) - vector from self to opponent
    relative_velocity: Array  # shape (2,) - opponent_vel - own_vel
    own_velocity: Array  # shape (2,) - own velocity
    time_remaining: float  # normalized [0, 1]
    agent_id: float  # 0.0 = pursuer, 1.0 = evader
