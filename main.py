"""Example usage of the Pursuer-Evader environment."""

import jax
import jax.numpy as jnp
from peax import PursuerEvaderEnv


def random_policy(key, obs, max_force=10.0):
    """Simple random policy for demonstration."""
    return jax.random.uniform(key, shape=(2,), minval=-max_force, maxval=max_force)


def main():
    print("=" * 60)
    print("PEAX: Pursuer-Evader Environment Demo")
    print("=" * 60)

    # Create environment with different boundary types
    boundary_types = ["square", "triangle", "circle"]

    for boundary_type in boundary_types:
        print(f"\n--- Testing with {boundary_type} boundary ---")

        env = PursuerEvaderEnv(
            boundary_type=boundary_type,
            boundary_size=10.0,
            max_steps=200,
            capture_radius=0.5,
        )

        # Initialize environment
        key = jax.random.PRNGKey(42)
        key, reset_key = jax.random.split(key)
        state, obs = env.reset(reset_key)

        print(f"Initial pursuer position: {state.pursuer.position}")
        print(f"Initial evader position: {state.evader.position}")
        print(f"Initial distance: {jnp.sqrt(jnp.sum((state.pursuer.position - state.evader.position)**2)):.2f}")

        # Run episode with random actions
        episode_length = 0
        total_pursuer_reward = 0.0
        total_evader_reward = 0.0

        for step in range(env.params.max_steps):
            # Sample random actions
            key, key1, key2 = jax.random.split(key, 3)
            pursuer_action = random_policy(key1, obs["pursuer"])
            evader_action = random_policy(key2, obs["evader"])

            actions = {
                "pursuer": pursuer_action,
                "evader": evader_action,
            }

            # Step environment
            state, obs, rewards, done, info = env.step(state, actions)

            episode_length += 1
            total_pursuer_reward += rewards["pursuer"]
            total_evader_reward += rewards["evader"]

            if done:
                print(f"\nEpisode finished after {episode_length} steps")
                print(f"Captured: {info['captured']}")
                print(f"Timeout: {info['timeout']}")
                print(f"Final distance: {info['distance']:.2f}")
                print(f"Pursuer total reward: {total_pursuer_reward}")
                print(f"Evader total reward: {total_evader_reward}")
                break

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
