"""Collect data from checkpointed Minimax-Q DQN agents using Flashbax.

This script loads a trained minimax-Q agent checkpoint and runs episodes,
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

# Import from minimax_dqn
import sys
sys.path.insert(0, os.path.dirname(__file__))
from minimax_dqn import (
    JointQNetwork, MinimaxDQNConfig, get_global_state,
    discretize_action, get_minimax_action
)

from peax import PursuerEvaderEnv


def load_checkpoint(checkpoint_path: str) -> Tuple[JointQNetwork, Dict, MinimaxDQNConfig]:
    """Load a trained minimax-Q agent checkpoint.

    Args:
        checkpoint_path: Path to checkpoint pickle file

    Returns:
        Tuple of (q_network, params, config)
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = MinimaxDQNConfig(**checkpoint['config'])
    params = checkpoint['params']

    # Create joint Q-network
    num_actions = config.num_actions_per_dim ** 2
    q_network = JointQNetwork(
        pursuer_action_dim=num_actions,
        evader_action_dim=num_actions
    )

    return q_network, params, config


def select_minimax_actions(
    q_network: JointQNetwork,
    params: Dict,
    global_state: jnp.ndarray,
    num_actions_per_dim: int,
    max_force: float
) -> Tuple[int, int, jnp.ndarray, jnp.ndarray]:
    """Select actions using minimax strategy.

    Args:
        q_network: Joint Q-network
        params: Network parameters
        global_state: Global state array
        num_actions_per_dim: Number of discrete actions per dimension
        max_force: Maximum force magnitude

    Returns:
        Tuple of (pursuer_action_idx, evader_action_idx, pursuer_force, evader_force)
    """
    # Get Q-matrix for all joint actions
    q_matrix = q_network.apply(params, global_state[None, :])[0]

    # Select minimax actions
    pursuer_action_idx = get_minimax_action(q_matrix, is_pursuer=True)
    evader_action_idx = get_minimax_action(q_matrix, is_pursuer=False)

    # Convert to forces
    pursuer_force = discretize_action(pursuer_action_idx, num_actions_per_dim, max_force)
    evader_force = discretize_action(evader_action_idx, num_actions_per_dim, max_force)

    return pursuer_action_idx, evader_action_idx, pursuer_force, evader_force


def run_episode(
    env: PursuerEvaderEnv,
    q_network: JointQNetwork,
    params: Dict,
    num_actions_per_dim: int,
    key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Run a single episode and collect joint state-action transitions.

    Args:
        env: PursuerEvaderEnv environment
        q_network: Joint Q-network
        params: Network parameters
        num_actions_per_dim: Number of discrete actions per dimension
        key: JAX random key

    Returns:
        Tuple of (states, pursuer_actions, evader_actions, rewards, next_states, dones, info)
    """
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    # Lists to collect data
    states = []
    pursuer_actions = []
    evader_actions = []
    rewards = []
    next_states = []
    dones_list = []

    for step in range(env.params.max_steps):
        # Get normalized global state
        global_state = get_global_state(env_state, env)

        # Select actions using minimax strategy
        pursuer_action_idx, evader_action_idx, pursuer_force, evader_force = select_minimax_actions(
            q_network, params, global_state, num_actions_per_dim, env.params.max_force
        )

        # Step environment
        actions_dict = {"pursuer": pursuer_force, "evader": evader_force}
        next_env_state, next_obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)

        # Get next global state
        next_global_state = get_global_state(next_env_state, env)

        # Store joint transition
        states.append(global_state)
        pursuer_actions.append(pursuer_action_idx)
        evader_actions.append(evader_action_idx)
        rewards.append(rewards_dict["pursuer"])  # Zero-sum: pursuer's perspective
        next_states.append(next_global_state)
        dones_list.append(done)

        # Update state
        env_state = next_env_state
        obs_dict = next_obs_dict

        if done:
            break

    # Convert to JAX arrays
    states = jnp.array(states, dtype=jnp.float32)
    pursuer_actions = jnp.array(pursuer_actions, dtype=jnp.int32)
    evader_actions = jnp.array(evader_actions, dtype=jnp.int32)
    rewards = jnp.array(rewards, dtype=jnp.float32)
    next_states = jnp.array(next_states, dtype=jnp.float32)
    dones_array = jnp.array(dones_list, dtype=jnp.float32)

    return states, pursuer_actions, evader_actions, rewards, next_states, dones_array, info


def collect_minimax_data(
    checkpoint_path: str,
    num_episodes: int,
    output_dir: str,
    seed: int = 0
):
    """Collect minimax-Q data and save to Flashbax buffer.

    Args:
        checkpoint_path: Path to agent checkpoint
        num_episodes: Number of episodes to collect
        output_dir: Directory to save buffer data
        seed: Random seed
    """
    print("=" * 70)
    print("Minimax-Q Data Collection")
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
        wall_penalty_coef=config.wall_penalty_coef,
        velocity_reward_coef=config.velocity_reward_coef,
    )

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Calculate buffer size (estimate: max_steps * num_episodes)
    estimated_size = config.max_steps * num_episodes

    # Create Flashbax buffer
    print(f"Creating Flashbax buffer (estimated size: {estimated_size})")

    # Create example transition for buffer initialization
    # Global state: [p_pos(2), p_vel(2), e_pos(2), e_vel(2), time(1)] = 9D
    example_transition = {
        "state": jnp.zeros(9, dtype=jnp.float32),
        "pursuer_action": jnp.array(0, dtype=jnp.int32),
        "evader_action": jnp.array(0, dtype=jnp.int32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "next_state": jnp.zeros(9, dtype=jnp.float32),
        "done": jnp.array(0.0, dtype=jnp.float32),
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
    print(f"Collecting {num_episodes} episodes...")
    total_transitions = 0
    total_captures = 0
    total_timeouts = 0
    episode_rewards = []

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        key, episode_key = jax.random.split(key)

        # Run episode
        states, pursuer_actions, evader_actions, rewards, next_states, dones, info = run_episode(
            env, q_network, params, config.num_actions_per_dim, episode_key
        )

        # Add transitions to buffer
        for i in range(len(states)):
            transition = {
                "state": states[i],
                "pursuer_action": pursuer_actions[i],
                "evader_action": evader_actions[i],
                "reward": rewards[i],
                "next_state": next_states[i],
                "done": dones[i],
            }
            buffer_state = buffer.add(buffer_state, transition)

        total_transitions += len(states)

        # Track statistics
        if info["captured"]:
            total_captures += 1
        if info["timeout"]:
            total_timeouts += 1

        # Calculate episode reward
        episode_reward = float(jnp.sum(rewards))
        episode_rewards.append(episode_reward)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            capture_rate = total_captures / (episode + 1) * 100
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Transitions collected: {total_transitions}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Capture Rate: {capture_rate:.1f}%")

    print("\n" + "=" * 70)
    print("Data Collection Summary")
    print("=" * 70)
    print(f"Total episodes: {num_episodes}")
    print(f"Total transitions: {total_transitions}")
    print(f"Captures: {total_captures} ({total_captures/num_episodes*100:.1f}%)")
    print(f"Timeouts: {total_timeouts} ({total_timeouts/num_episodes*100:.1f}%)")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f}")
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
        "episode_rewards": episode_rewards,
        "buffer_shape": {
            "state": states[0].shape,
            "pursuer_action": (),
            "evader_action": (),
            "reward": (),
            "next_state": next_states[0].shape,
            "done": (),
        },
        "note": "Joint state-action transitions from minimax-Q agent"
    }

    print(f"Saving metadata to {metadata_save_path}")
    with open(metadata_save_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✅ Data collection complete!")
    print(f"   Buffer saved to: {buffer_save_path}")
    print(f"   Metadata saved to: {metadata_save_path}")

    return buffer_state, metadata


def main():
    parser = argparse.ArgumentParser(description="Collect data from checkpointed minimax-Q agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default="minimax_checkpoint_step_500.pkl",
        help="Path to minimax-Q agent checkpoint (.pkl file)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="minimax_data",
        help="Output directory for buffer data (default: minimax_data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    args = parser.parse_args()

    # Collect data
    collect_minimax_data(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
