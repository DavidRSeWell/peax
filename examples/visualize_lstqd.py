"""Visualize trained LSTQD agent by loading a C matrix and rendering episodes as GIF.

This script loads a trained C matrix from run_lstqd_end_end.py and creates
a GIF visualization showing the agent's gameplay.

Usage:
    python visualize_lstqd.py                    # Uses default C_old.npy
    python visualize_lstqd.py --c_matrix_path my_matrix.npy
    python visualize_lstqd.py --num_episodes 10  # Render 10 episodes
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np

from peax.fpta import LSTQD
from peax.basis import simple_pursuer_evader_basis
from peax.environment import PursuerEvaderEnv, discretize_action
from run_lstqd_end_end import render_episode_to_gif


def main():
    parser = argparse.ArgumentParser(description="Visualize trained LSTQD agent")
    parser.add_argument(
        "--c_matrix_path",
        type=str,
        default="C_old.npy",
        help="Path to the saved C matrix (.npy file)"
    )
    parser.add_argument(
        "--output_gif",
        type=str,
        default="lstqd_visualization.gif",
        help="Output GIF filename"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to render in the GIF"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    # Environment parameters
    parser.add_argument("--boundary_type", type=str, default="square")
    parser.add_argument("--boundary_size", type=float, default=10.0)
    parser.add_argument("--capture_radius", type=float, default=0.5)
    parser.add_argument("--num_actions_per_dim", type=int, default=5)
    parser.add_argument("--wall_penalty_coef", type=float, default=0.01)
    parser.add_argument("--velocity_reward_coef", type=float, default=0.005)

    args = parser.parse_args()

    print("=" * 70)
    print("LSTQD Agent Visualization")
    print("=" * 70)
    print(f"Loading C matrix from: {args.c_matrix_path}")
    print(f"Output GIF: {args.output_gif}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print("=" * 70)

    # Load C matrix
    C = np.load(args.c_matrix_path)
    print(f"Loaded C matrix with shape: {C.shape}")
    C = jnp.array(C)

    # Set random seed
    key = jax.random.PRNGKey(args.seed)

    # Environment setup
    env = PursuerEvaderEnv(
        boundary_type=args.boundary_type,
        boundary_size=args.boundary_size,
        max_steps=args.max_steps,
        capture_radius=args.capture_radius,
        wall_penalty_coef=args.wall_penalty_coef,
        velocity_reward_coef=args.velocity_reward_coef,
    )

    obs_dim = env.observation_space_dim
    act_dim = 2

    print(f"Environment observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    # Create discrete action space
    possible_actions = [
        discretize_action(i, args.num_actions_per_dim, env.params.max_force)
        for i in range(args.num_actions_per_dim ** 2)
    ]
    print(f"Number of discrete actions: {len(possible_actions)}")

    # Basis setup
    basis_fn = simple_pursuer_evader_basis(
        alpha=0.2,
        num_trait=obs_dim,
        num_actions=act_dim,
    )

    # LSTQD setup
    lstqd = LSTQD(basis=basis_fn, num_actions=len(possible_actions))
    print(f"LSTQD initialized with {lstqd.m} basis functions")

    # Verify C matrix shape matches
    if C.shape[0] != lstqd.m or C.shape[1] != lstqd.m:
        print(f"WARNING: C matrix shape {C.shape} doesn't match LSTQD basis size ({lstqd.m}, {lstqd.m})")
        print("This may cause errors or incorrect behavior.")

    print("=" * 70)
    print("Rendering episodes...")
    print("=" * 70)

    # Render episodes to GIF
    episode_info = render_episode_to_gif(
        env=env,
        fpta=lstqd,
        C=C,
        possible_actions=possible_actions,
        key=key,
        filename=args.output_gif,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes
    )

    print("=" * 70)
    print("Rendering complete!")
    print("=" * 70)
    print(f"GIF saved to: {args.output_gif}")
    print(f"Average reward: {episode_info['reward']:.2f}")
    print(f"Last episode captured: {episode_info['captured']}")
    print(f"Last episode timeout: {episode_info['timeout']}")
    print(f"Total frames: {episode_info['length']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
