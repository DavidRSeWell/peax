"""Tests for the pursuer-evader environment."""

import jax
import jax.numpy as jnp
import pytest

from peax import PursuerEvaderEnv, AgentState, EnvState


class TestPursuerEvaderEnv:
    """Test suite for PursuerEvaderEnv."""

    def test_environment_creation(self):
        """Test that environment can be created with different boundaries."""
        for boundary_type in ["square", "triangle", "circle"]:
            env = PursuerEvaderEnv(boundary_type=boundary_type)
            assert env is not None
            assert env.params.max_steps == 200

    def test_reset(self):
        """Test environment reset."""
        env = PursuerEvaderEnv(boundary_type="square", boundary_size=10.0)
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)

        # Check state structure
        assert isinstance(state, EnvState)
        assert isinstance(state.pursuer, AgentState)
        assert isinstance(state.evader, AgentState)
        assert state.time == 0

        # Check observations
        assert "pursuer" in obs
        assert "evader" in obs
        assert obs["pursuer"].own_position.shape == (2,)
        assert obs["pursuer"].time_remaining == 1.0

    def test_step(self):
        """Test environment step."""
        env = PursuerEvaderEnv(boundary_type="square", boundary_size=10.0)
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)

        # Take a step
        actions = {
            "pursuer": jnp.array([1.0, 0.0]),
            "evader": jnp.array([-1.0, 0.0]),
        }
        new_state, new_obs, rewards, done, info = env.step(state, actions)

        # Check state updated
        assert new_state.time == 1
        assert not jnp.allclose(new_state.pursuer.position, state.pursuer.position)
        assert not jnp.allclose(new_state.evader.position, state.evader.position)

        # Check rewards are zero-sum
        assert rewards["pursuer"] + rewards["evader"] == 0.0

    def test_capture(self):
        """Test that capture is detected correctly."""
        env = PursuerEvaderEnv(boundary_type="square", capture_radius=1.0)

        # Create state where agents are close
        pursuer_state = AgentState(
            position=jnp.array([0.0, 0.0]),
            velocity=jnp.zeros(2)
        )
        evader_state = AgentState(
            position=jnp.array([0.5, 0.0]),  # Within capture radius
            velocity=jnp.zeros(2)
        )
        state = EnvState(pursuer=pursuer_state, evader=evader_state, time=0)

        # Take a step with zero actions
        actions = {
            "pursuer": jnp.zeros(2),
            "evader": jnp.zeros(2),
        }
        new_state, obs, rewards, done, info = env.step(state, actions)

        # Should be captured
        assert done
        assert info["captured"]
        assert rewards["pursuer"] == 1.0
        assert rewards["evader"] == -1.0

    def test_timeout(self):
        """Test that timeout is handled correctly."""
        env = PursuerEvaderEnv(max_steps=10)
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)

        # Run until timeout
        actions = {
            "pursuer": jnp.zeros(2),
            "evader": jnp.zeros(2),
        }

        for _ in range(10):
            state, obs, rewards, done, info = env.step(state, actions)

        # Should timeout
        assert done
        assert info["timeout"]
        assert rewards["pursuer"] == -1.0
        assert rewards["evader"] == 1.0

    def test_boundary_clipping(self):
        """Test that agents are clipped to boundary."""
        env = PursuerEvaderEnv(boundary_type="square", boundary_size=2.0)

        # Create state where agent is at boundary
        pursuer_state = AgentState(
            position=jnp.array([0.9, 0.9]),
            velocity=jnp.array([10.0, 10.0])  # Large velocity towards boundary
        )
        evader_state = AgentState(
            position=jnp.array([0.0, 0.0]),
            velocity=jnp.zeros(2)
        )
        state = EnvState(pursuer=pursuer_state, evader=evader_state, time=0)

        # Apply large force towards boundary
        actions = {
            "pursuer": jnp.array([10.0, 10.0]),
            "evader": jnp.zeros(2),
        }
        new_state, obs, rewards, done, info = env.step(state, actions)

        # Check position is within boundary (size/2 = 1.0)
        assert jnp.all(jnp.abs(new_state.pursuer.position) <= 1.0)

    def test_observation_dimensions(self):
        """Test observation space dimensions."""
        env = PursuerEvaderEnv()
        assert env.observation_space_dim == 9
        assert env.action_space_dim == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
