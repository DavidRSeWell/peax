"""Double integrator dynamics for point mass agents."""

import jax.numpy as jnp
from chex import Array

from peax.types import AgentState


def double_integrator_step(
    state: AgentState,
    force: Array,
    mass: float,
    dt: float,
    max_force: float,
) -> AgentState:
    """Update agent state using double integrator dynamics.

    Dynamics:
        acceleration = force / mass
        velocity = velocity + acceleration * dt
        position = position + velocity * dt

    Args:
        state: Current agent state
        force: Control force (fx, fy)
        mass: Agent mass
        dt: Timestep duration
        max_force: Maximum force magnitude (for clipping)

    Returns:
        Updated agent state
    """
    # Clip force to maximum magnitude
    force_magnitude = jnp.sqrt(jnp.sum(force ** 2))
    force = jnp.where(
        force_magnitude > max_force,
        force * (max_force / force_magnitude),
        force
    )

    # Compute acceleration
    acceleration = force / mass

    # Update velocity and position using Euler integration
    new_velocity = state.velocity + acceleration * dt
    new_position = state.position + new_velocity * dt

    return AgentState(position=new_position, velocity=new_velocity)


def compute_distance(state1: AgentState, state2: AgentState) -> Array:
    """Compute Euclidean distance between two agents.

    Args:
        state1: First agent state
        state2: Second agent state

    Returns:
        Scalar distance between agents
    """
    return jnp.sqrt(jnp.sum((state1.position - state2.position) ** 2))


def check_capture(
    pursuer_state: AgentState,
    evader_state: AgentState,
    capture_radius: float
) -> Array:
    """Check if pursuer has captured evader.

    Args:
        pursuer_state: State of pursuer
        evader_state: State of evader
        capture_radius: Distance threshold for capture

    Returns:
        Boolean indicating if capture occurred
    """
    distance = compute_distance(pursuer_state, evader_state)
    return distance <= capture_radius


def clip_state_to_boundary(state: AgentState, boundary) -> AgentState:
    """Clip agent state to remain within boundary.

    When an agent hits the boundary, its position is clipped and
    velocity is set to zero.

    Args:
        state: Agent state
        boundary: Boundary object with clip_to_boundary method

    Returns:
        Clipped agent state
    """
    clipped_position = boundary.clip_to_boundary(state.position)

    # Check if position was clipped (agent hit boundary)
    hit_boundary = jnp.any(clipped_position != state.position)

    # Zero out velocity if hit boundary
    new_velocity = jnp.where(hit_boundary, jnp.zeros_like(state.velocity), state.velocity)

    return AgentState(position=clipped_position, velocity=new_velocity)
