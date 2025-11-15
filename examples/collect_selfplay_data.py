"""Collect self-play data from checkpointed DQN agents using Flashbax.

This script loads a trained agent checkpoint and runs self-play episodes,
storing all transitions in a Flashbax replay buffer for offline RL training.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import flashbax as fbx
from tqdm import tqdm

# Import from cleanrl_dqn
import sys
sys.path.insert(0, os.path.dirname(__file__))
from cleanrl_dqn import QNetwork, DQNConfig, observation_to_array, discretize_action

from peax import PursuerEvaderEnv


def load_checkpoint(checkpoint_path: str) -> Tuple[QNetwork, Dict, DQNConfig]:
    """Load a trained agent checkpoint.

    Args:
        checkpoint_path: Path to checkpoint pickle file

    Returns:
        Tuple of (q_network, params, config)
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = DQNConfig(**checkpoint['config'])
    params = checkpoint['params']

    # Create Q-network
    num_actions = config.num_actions_per_dim ** 2
    q_network = QNetwork(action_dim=num_actions)

    return q_network, params, config


def select_greedy_action(
    q_network: QNetwork,
    params: Dict,
    obs: jnp.ndarray,
    num_actions_per_dim: int,
    max_force: float
) -> Tuple[int, jnp.ndarray]:
    """Select greedy action from Q-network.

    Args:
        q_network: Q-network
        params: Network parameters
        obs: Observation array
        num_actions_per_dim: Number of discrete actions per dimension
        max_force: Maximum force magnitude

    Returns:
        Tuple of (action_index, force_vector)
    """
    q_values = q_network.apply(params, obs)
    action_idx = int(jnp.argmax(q_values))
    force = discretize_action(action_idx, num_actions_per_dim, max_force)
    return action_idx, force


def run_selfplay_episode(
    env: PursuerEvaderEnv,
    q_network: QNetwork,
    params: Dict,
    num_actions_per_dim: int,
    key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Run a single self-play episode and collect transitions.

    Args:
        env: PursuerEvaderEnv environment
        q_network: Q-network
        params: Network parameters
        num_actions_per_dim: Number of discrete actions per dimension
        key: JAX random key

    Returns:
        Tuple of (observations, actions, rewards, next_observations, dones, agent_ids, info)
        Arrays are collected from both agents' perspectives
    """
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    # Lists to collect data
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones_list = []
    agent_ids = []  # 0 = pursuer, 1 = evader

    for step in range(env.params.max_steps):
        # Get observations
        pursuer_obs = observation_to_array(obs_dict["pursuer"])
        evader_obs = observation_to_array(obs_dict["evader"])

        # Both agents use the same network (self-play)
        pursuer_action_idx, pursuer_force = select_greedy_action(
            q_network, params, pursuer_obs, num_actions_per_dim, env.params.max_force
        )
        evader_action_idx, evader_force = select_greedy_action(
            q_network, params, evader_obs, num_actions_per_dim, env.params.max_force
        )

        # Step environment
        actions_dict = {"pursuer": pursuer_force, "evader": evader_force}
        next_env_state, next_obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)

        # Get next observations
        next_pursuer_obs = observation_to_array(next_obs_dict["pursuer"])
        next_evader_obs = observation_to_array(next_obs_dict["evader"])

        # Store pursuer transition
        observations.append(pursuer_obs)
        actions.append(pursuer_action_idx)
        rewards.append(rewards_dict["pursuer"])
        next_observations.append(next_pursuer_obs)
        dones_list.append(done)
        agent_ids.append(0)
        
        # Store evader transition
        observations.append(evader_obs)
        actions.append(evader_action_idx)
        rewards.append(rewards_dict["evader"])
        next_observations.append(next_evader_obs)
        dones_list.append(done)
        agent_ids.append(1)

        # Update state
        env_state = next_env_state
        obs_dict = next_obs_dict

        if done:
            break

    # Convert to JAX arrays
    observations = jnp.array(observations, dtype=jnp.float32)
    actions = jnp.array(actions, dtype=jnp.int32)
    rewards = jnp.array(rewards, dtype=jnp.float32)
    next_observations = jnp.array(next_observations, dtype=jnp.float32)
    dones_array = jnp.array(dones_list, dtype=jnp.float32)
    agent_ids = jnp.array(agent_ids, dtype=jnp.int32)

    return observations, actions, rewards, next_observations, dones_array, agent_ids, info


def collect_selfplay_data(
    checkpoint_path: str,
    num_episodes: int,
    output_dir: str,
    seed: int = 0
):
    """Collect self-play data and save to Flashbax buffer.

    Args:
        checkpoint_path: Path to agent checkpoint
        num_episodes: Number of episodes to collect
        output_dir: Directory to save buffer data
        seed: Random seed
    """
    print("=" * 70)
    print("Self-Play Data Collection")
    print("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    q_network, params, config = load_checkpoint(checkpoint_path)
    print(f"Loaded agent trained for {config.total_timesteps} timesteps")

    # Create environment
    env = PursuerEvaderEnv(
        boundary_type=config.boundary_type,
        boundary_size=config.boundary_size,
        max_steps=config.max_steps,
        capture_radius=config.capture_radius,
    )

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Calculate buffer size (estimate: 2 * max_steps * num_episodes)
    estimated_size = 2 * config.max_steps * num_episodes

    # Create Flashbax buffer
    # Using item buffer since we're storing independent transitions
    print(f"Creating Flashbax buffer (estimated size: {estimated_size})")

    # Create example transition for buffer initialization
    example_transition = {
        "observation": jnp.zeros(env.observation_space_dim, dtype=jnp.float32),
        "action": jnp.array(0, dtype=jnp.int32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "next_observation": jnp.zeros(env.observation_space_dim, dtype=jnp.float32),
        "done": jnp.array(0.0, dtype=jnp.float32),
        "agent_id": jnp.array(0, dtype=jnp.int32),
    }

    # Create item buffer (for independent transitions)
    buffer = fbx.make_item_buffer(
        max_length=estimated_size,
        min_length=1,
        sample_batch_size=256,
        add_batches=False,  # Add single items
    )

    buffer_state = buffer.init(example_transition)

    # Collect episodes
    print(f"Collecting {num_episodes} self-play episodes...")
    total_transitions = 0
    total_captures = 0
    total_timeouts = 0
    episode_rewards_pursuer = []
    episode_rewards_evader = []

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        key, episode_key = jax.random.split(key)

        # Run episode
        observations, actions, rewards, next_observations, dones, agent_ids, info = run_selfplay_episode(
            env, q_network, params, config.num_actions_per_dim, episode_key
        )

        # Add transitions to buffer
        for i in range(len(observations)):
            transition = {
                "observation": observations[i],
                "action": actions[i],
                "reward": rewards[i],
                "next_observation": next_observations[i],
                "done": dones[i],
                "agent_id": agent_ids[i],
            }
            buffer_state = buffer.add(buffer_state, transition)

        total_transitions += len(observations)

        # Track statistics
        if info["captured"]:
            total_captures += 1
        if info["timeout"]:
            total_timeouts += 1

        # Calculate episode rewards for each agent
        pursuer_reward = float(jnp.sum(rewards[agent_ids == 0]))
        evader_reward = float(jnp.sum(rewards[agent_ids == 1]))
        episode_rewards_pursuer.append(pursuer_reward)
        episode_rewards_evader.append(evader_reward)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_pursuer_reward = np.mean(episode_rewards_pursuer[-10:])
            avg_evader_reward = np.mean(episode_rewards_evader[-10:])
            capture_rate = total_captures / (episode + 1) * 100
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Transitions collected: {total_transitions}")
            print(f"  Avg Pursuer Reward: {avg_pursuer_reward:.2f}")
            print(f"  Avg Evader Reward: {avg_evader_reward:.2f}")
            print(f"  Capture Rate: {capture_rate:.1f}%")

    print("\n" + "=" * 70)
    print("Data Collection Summary")
    print("=" * 70)
    print(f"Total episodes: {num_episodes}")
    print(f"Total transitions: {total_transitions}")
    print(f"Captures: {total_captures} ({total_captures/num_episodes*100:.1f}%)")
    print(f"Timeouts: {total_timeouts} ({total_timeouts/num_episodes*100:.1f}%)")
    print(f"Avg Pursuer Reward: {np.mean(episode_rewards_pursuer):.2f}")
    print(f"Avg Evader Reward: {np.mean(episode_rewards_evader):.2f}")
    print("=" * 70)

    # Save buffer state and metadata
    buffer_save_path = output_path / "buffer_state.pkl"
    metadata_save_path = output_path / "metadata.pkl"

    print(f"\nSaving buffer to {buffer_save_path}")
    with open(buffer_save_path, 'wb') as f:
        pickle.dump(buffer_state, f)

    # Save metadata
    metadata = {
        "num_episodes": num_episodes,
        "total_transitions": total_transitions,
        "config": vars(config),
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "captures": total_captures,
        "timeouts": total_timeouts,
        "episode_rewards_pursuer": episode_rewards_pursuer,
        "episode_rewards_evader": episode_rewards_evader,
        "buffer_shape": {
            "observation": observations[0].shape,
            "action": (),
            "reward": (),
            "next_observation": next_observations[0].shape,
            "done": (),
            "agent_id": (),
        }
    }

    print(f"Saving metadata to {metadata_save_path}")
    with open(metadata_save_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✅ Data collection complete!")
    print(f"   Buffer saved to: {buffer_save_path}")
    print(f"   Metadata saved to: {metadata_save_path}")

    return buffer_state, metadata


def main():
    parser = argparse.ArgumentParser(description="Collect self-play data from checkpointed agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint (.pkl file)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of self-play episodes to collect (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for buffer data (default: data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    args = parser.parse_args()

    # Collect data
    collect_selfplay_data(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
