"""Main pursuer-evader environment with Gym-style interface."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from peax.types import AgentState, EnvState, EnvParams, Observation
from peax.boundaries import create_boundary, Boundary
from peax.dynamics import (
    double_integrator_step,
    check_capture,
    clip_state_to_boundary,
    compute_distance,
)


class PursuerEvaderEnv:
    """Pursuer-Evader environment for reinforcement learning.

    This environment implements a 2-player zero-sum game where a pursuer
    tries to catch an evader within a time limit. Both agents are point
    masses with double integrator dynamics.

    The environment follows a functional style compatible with JAX - all
    methods are stateless and take/return state explicitly.
    """

    def __init__(
        self,
        boundary_type: str = "square",
        boundary_size: float = 10.0,
        max_steps: int = 200,
        dt: float = 0.1,
        capture_radius: float = 0.5,
        pursuer_mass: float = 1.0,
        evader_mass: float = 1.0,
        max_force: float = 10.0,
    ):
        """Initialize the pursuer-evader environment.

        Args:
            boundary_type: Type of boundary ("square", "triangle", or "circle")
            boundary_size: Size of the boundary
            max_steps: Maximum number of timesteps
            dt: Timestep duration
            capture_radius: Distance threshold for capture
            pursuer_mass: Mass of pursuer agent
            evader_mass: Mass of evader agent
            max_force: Maximum force magnitude
        """
        self.boundary = create_boundary(boundary_type, boundary_size)
        self.params = EnvParams(
            max_steps=max_steps,
            dt=dt,
            capture_radius=capture_radius,
            pursuer_mass=pursuer_mass,
            evader_mass=evader_mass,
            max_force=max_force,
            boundary_size=boundary_size,
        )

    def reset(self, key: PRNGKey) -> Tuple[EnvState, Dict[str, Observation]]:
        """Reset the environment to initial state.

        Args:
            key: JAX random key

        Returns:
            Tuple of (initial state, initial observations)
        """
        key1, key2 = jax.random.split(key)

        # Sample random positions for pursuer and evader
        pursuer_pos = self.boundary.sample_position(key1)
        evader_pos = self.boundary.sample_position(key2)

        # Initialize with zero velocity
        pursuer_state = AgentState(
            position=pursuer_pos,
            velocity=jnp.zeros(2)
        )
        evader_state = AgentState(
            position=evader_pos,
            velocity=jnp.zeros(2)
        )

        state = EnvState(
            pursuer=pursuer_state,
            evader=evader_state,
            time=0
        )

        obs = self._get_observations(state)

        return state, obs

    def step(
        self,
        state: EnvState,
        actions: Dict[str, Array]
    ) -> Tuple[EnvState, Dict[str, Observation], Dict[str, float], bool, Dict]:
        """Take a step in the environment.

        Args:
            state: Current environment state
            actions: Dictionary with "pursuer" and "evader" force actions

        Returns:
            Tuple of (next_state, observations, rewards, done, info)
        """
        pursuer_force = actions["pursuer"]
        evader_force = actions["evader"]

        # Update agent states using double integrator dynamics
        new_pursuer_state = double_integrator_step(
            state.pursuer,
            pursuer_force,
            self.params.pursuer_mass,
            self.params.dt,
            self.params.max_force,
        )
        new_evader_state = double_integrator_step(
            state.evader,
            evader_force,
            self.params.evader_mass,
            self.params.dt,
            self.params.max_force,
        )

        # Clip positions to boundary and handle collisions
        new_pursuer_state = clip_state_to_boundary(new_pursuer_state, self.boundary)
        new_evader_state = clip_state_to_boundary(new_evader_state, self.boundary)

        # Update time
        new_time = state.time + 1

        # Create new state
        new_state = EnvState(
            pursuer=new_pursuer_state,
            evader=new_evader_state,
            time=new_time
        )

        # Check termination conditions
        captured = check_capture(
            new_pursuer_state,
            new_evader_state,
            self.params.capture_radius
        )
        timeout = new_time >= self.params.max_steps
        done = captured | timeout

        # Compute rewards (zero-sum game)
        rewards = self._compute_rewards(new_state, captured, timeout)

        # Get observations
        obs = self._get_observations(new_state)

        # Info dictionary
        info = {
            "captured": bool(captured),
            "timeout": bool(timeout),
            "distance": float(compute_distance(new_pursuer_state, new_evader_state)),
            "time": int(new_time),
        }

        return new_state, obs, rewards, bool(done), info

    def _get_observations(self, state: EnvState) -> Dict[str, Observation]:
        """Get observations for both agents.

        Args:
            state: Current environment state

        Returns:
            Dictionary with observations for "pursuer" and "evader"
        """
        time_remaining = float(self.params.max_steps - state.time) / self.params.max_steps

        pursuer_obs = Observation(
            own_position=state.pursuer.position,
            own_velocity=state.pursuer.velocity,
            other_position=state.evader.position,
            other_velocity=state.evader.velocity,
            time_remaining=time_remaining,
        )

        evader_obs = Observation(
            own_position=state.evader.position,
            own_velocity=state.evader.velocity,
            other_position=state.pursuer.position,
            other_velocity=state.pursuer.velocity,
            time_remaining=time_remaining,
        )

        return {
            "pursuer": pursuer_obs,
            "evader": evader_obs,
        }

    def _compute_rewards(
        self,
        state: EnvState,
        captured: Array,
        timeout: Array
    ) -> Dict[str, float]:
        """Compute rewards for both agents.

        This is a zero-sum game:
        - Pursuer gets +1 for capture, -1 for timeout
        - Evader gets -1 for capture, +1 for timeout
        - Otherwise, sparse reward of 0

        Args:
            state: Current environment state
            captured: Whether evader was captured
            timeout: Whether time limit was reached

        Returns:
            Dictionary with rewards for "pursuer" and "evader"
        """
        dist_ = jnp.linalg.norm(state.pursuer.position - state.evader.position)

        r_immediate = dist_ / self.boundary.max_dist

        # Sparse rewards: only at episode end
        pursuer_reward = jnp.where(
            captured,
            1.0,  # Pursuer wins
            jnp.where(timeout, -1.0, 0.0)  # Evader wins or game continues
        )*self.boundary.max_dist + -r_immediate

        # Zero-sum: evader reward is negative of pursuer reward
        evader_reward = -pursuer_reward 

        return {
            "pursuer": float(pursuer_reward),
            "evader": float(evader_reward),
        }

    @property
    def observation_space_dim(self) -> int:
        """Dimension of observation space.

        Observation includes:
        - Own position (2)
        - Own velocity (2)
        - Other position (2)
        - Other velocity (2)
        - Time remaining (1)
        Total: 9
        """
        return 9

    @property
    def action_space_dim(self) -> int:
        """Dimension of action space (force in x and y)."""
        return 2
