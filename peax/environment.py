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
        wall_penalty_coef: float = 0.0,
        velocity_reward_coef: float = 0.0,
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
            wall_penalty_coef: Coefficient for wall proximity penalty (0.0 = disabled)
            velocity_reward_coef: Coefficient for velocity reward (0.0 = disabled)
        """
        self.boundary_type = boundary_type
        self.boundary = create_boundary(boundary_type, boundary_size)
        self.params = EnvParams(
            max_steps=max_steps,
            dt=dt,
            capture_radius=capture_radius,
            pursuer_mass=pursuer_mass,
            evader_mass=evader_mass,
            max_force=max_force,
            boundary_size=boundary_size,
            wall_penalty_coef=wall_penalty_coef,
            velocity_reward_coef=velocity_reward_coef,
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

        This is a zero-sum game with shaped rewards:
        - Terminal: Pursuer gets +1 for capture, -1 for timeout
        - Immediate: Distance-based shaping (±0.005 per step)
        - Wall penalty: Optional penalty for being near walls
        - Velocity reward: Optional reward for movement
        - Cumulative shaping over 200 steps ≈ ±1.0, so terminal rewards dominate

        Args:
            state: Current environment state
            captured: Whether evader was captured
            timeout: Whether time limit was reached

        Returns:
            Dictionary with rewards for "pursuer" and "evader"
        """
        # Normalize distance to [0, 1]
        dist = jnp.linalg.norm(state.pursuer.position - state.evader.position)
        normalized_dist = dist / self.boundary.max_dist

        # Small immediate reward scaled to not dominate terminal reward
        # 0.005 * 200 steps = 1.0 max cumulative, matching terminal reward magnitude
        distance_reward = -0.005 * normalized_dist  # Pursuer wants to minimize distance

        # Wall proximity penalty (for both agents to discourage wall-sitting)
        wall_penalty_pursuer = 0.0
        wall_penalty_evader = 0.0
        if self.params.wall_penalty_coef > 0.0:
            # Distance to nearest wall (for square boundary)
            half_size = self.params.boundary_size / 2
            pursuer_wall_dist = jnp.min(jnp.array([
                half_size - jnp.abs(state.pursuer.position[0]),
                half_size - jnp.abs(state.pursuer.position[1])
            ]))
            evader_wall_dist = jnp.min(jnp.array([
                half_size - jnp.abs(state.evader.position[0]),
                half_size - jnp.abs(state.evader.position[1])
            ]))

            # Penalty inversely proportional to wall distance (0 at center, -1 at wall)
            # Normalized by half_size so max penalty is -wall_penalty_coef
            wall_penalty_pursuer = -self.params.wall_penalty_coef * (1.0 - pursuer_wall_dist / half_size)
            wall_penalty_evader = -self.params.wall_penalty_coef * (1.0 - evader_wall_dist / half_size)

        # Velocity reward (encourage movement)
        velocity_reward_pursuer = 0.0
        velocity_reward_evader = 0.0
        if self.params.velocity_reward_coef > 0.0:
            # Reward based on velocity magnitude (normalized by expected max velocity)
            max_expected_velocity = 20.0  # Approximate max velocity
            pursuer_speed = jnp.linalg.norm(state.pursuer.velocity)
            evader_speed = jnp.linalg.norm(state.evader.velocity)

            velocity_reward_pursuer = self.params.velocity_reward_coef * (pursuer_speed / max_expected_velocity)
            velocity_reward_evader = self.params.velocity_reward_coef * (evader_speed / max_expected_velocity)

        # Terminal rewards
        terminal_reward = jnp.where(
            captured,
            1.0,      # Pursuer wins
            jnp.where(timeout, -1.0, 0.0)  # Evader wins or game continues
        )

        # Combine all rewards
        pursuer_reward = (terminal_reward + distance_reward +
                         wall_penalty_pursuer + velocity_reward_pursuer)
        evader_reward = (-terminal_reward - distance_reward +
                        wall_penalty_evader + velocity_reward_evader)

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
