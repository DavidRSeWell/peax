"""Utility script to plot DQN training results with visualization."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon, Circle as MPLCircle

from peax import PursuerEvaderEnv
from dqn_pursuer import QNetwork, discretize_action, observation_to_array


def plot_training_curves(episode_rewards, episode_lengths, losses, save_path=None):
    """Plot training curves.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Smooth curves with moving average
    window = 10

    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Episode rewards
    axes[0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(episode_rewards) >= window:
        smoothed = moving_average(episode_rewards, window)
        axes[0].plot(range(window-1, len(episode_rewards)), smoothed,
                     color='blue', linewidth=2, label='Smoothed')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode lengths
    axes[1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
    if len(episode_lengths) >= window:
        smoothed = moving_average(episode_lengths, window)
        axes[1].plot(range(window-1, len(episode_lengths)), smoothed,
                     color='green', linewidth=2, label='Smoothed')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Training loss
    if losses:
        axes[2].plot(losses, alpha=0.3, color='red', label='Raw')
        if len(losses) >= window:
            smoothed = moving_average(losses, window)
            axes[2].plot(range(window-1, len(losses)), smoothed,
                         color='red', linewidth=2, label='Smoothed')
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('TD Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def visualize_episode(
    env: PursuerEvaderEnv,
    q_network: QNetwork,
    num_actions_per_dim: int = 5,
    seed: int = 42,
    save_path: None = None
):
    """Visualize a single episode with agent trajectories.

    Args:
        env: Environment
        q_network: Trained Q-network
        num_actions_per_dim: Number of discrete actions per dimension
        seed: Random seed
        save_path: Optional path to save figure
    """
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    state, obs_dict = env.reset(reset_key)

    # Store trajectories
    pursuer_positions = [state.pursuer.position]
    evader_positions = [state.evader.position]

    num_actions = num_actions_per_dim ** 2

    # Run episode
    for step in range(env.params.max_steps):
        # Pursuer action (greedy)
        pursuer_obs = observation_to_array(obs_dict["pursuer"])
        q_values = q_network(pursuer_obs)
        action_idx = int(jnp.argmax(q_values))
        pursuer_force = discretize_action(action_idx, num_actions_per_dim, env.params.max_force)

        # Random evader
        key, evader_key = jax.random.split(key)
        evader_force = jax.random.uniform(
            evader_key,
            shape=(2,),
            minval=-env.params.max_force,
            maxval=env.params.max_force
        )

        # Step
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        state, obs_dict, rewards, done, info = env.step(state, actions)

        pursuer_positions.append(state.pursuer.position)
        evader_positions.append(state.evader.position)

        if done:
            break

    # Convert to numpy arrays
    pursuer_positions = np.array(pursuer_positions)
    evader_positions = np.array(evader_positions)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw boundary
    boundary_size = env.params.boundary_size
    if hasattr(env.boundary, 'half_size'):  # Square
        half_size = env.boundary.half_size
        rect = Rectangle(
            (-half_size, -half_size),
            boundary_size,
            boundary_size,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.set_xlim(-half_size - 1, half_size + 1)
        ax.set_ylim(-half_size - 1, half_size + 1)
    elif hasattr(env.boundary, 'vertices'):  # Triangle
        vertices = np.array(env.boundary.vertices)
        polygon = Polygon(
            vertices,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(polygon)
        margin = boundary_size * 0.1
        ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
        ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    elif hasattr(env.boundary, 'radius'):  # Circle
        radius = env.boundary.radius
        circle = MPLCircle(
            (0, 0),
            radius,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(circle)
        ax.set_xlim(-radius - 1, radius + 1)
        ax.set_ylim(-radius - 1, radius + 1)

    # Plot trajectories
    ax.plot(pursuer_positions[:, 0], pursuer_positions[:, 1],
            'b-', linewidth=2, label='Pursuer', alpha=0.7)
    ax.plot(evader_positions[:, 0], evader_positions[:, 1],
            'r-', linewidth=2, label='Evader', alpha=0.7)

    # Plot start positions
    ax.scatter(pursuer_positions[0, 0], pursuer_positions[0, 1],
               c='blue', s=200, marker='o', edgecolors='black',
               linewidth=2, label='Pursuer Start', zorder=5)
    ax.scatter(evader_positions[0, 0], evader_positions[0, 1],
               c='red', s=200, marker='o', edgecolors='black',
               linewidth=2, label='Evader Start', zorder=5)

    # Plot end positions
    ax.scatter(pursuer_positions[-1, 0], pursuer_positions[-1, 1],
               c='blue', s=200, marker='X', edgecolors='black',
               linewidth=2, label='Pursuer End', zorder=5)
    ax.scatter(evader_positions[-1, 0], evader_positions[-1, 1],
               c='red', s=200, marker='X', edgecolors='black',
               linewidth=2, label='Evader End', zorder=5)

    # Draw capture radius at final position
    capture_circle = MPLCircle(
        pursuer_positions[-1],
        env.params.capture_radius,
        fill=False,
        edgecolor='blue',
        linestyle='--',
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(capture_circle)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    result = "CAPTURED" if info["captured"] else "TIMEOUT"
    ax.set_title(f'Episode Trajectory ({result} after {len(pursuer_positions)-1} steps)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {save_path}")

    plt.show()


def visualize_selfplay_episode(
    env: PursuerEvaderEnv,
    pursuer_q_network: QNetwork,
    evader_q_network: QNetwork,
    num_actions_per_dim: int = 5,
    seed: int = 42,
    save_path: None = None
):
    """Visualize a self-play episode with both trained agents.

    Args:
        env: Environment
        pursuer_q_network: Trained pursuer Q-network
        evader_q_network: Trained evader Q-network
        num_actions_per_dim: Number of discrete actions per dimension
        seed: Random seed
        save_path: Optional path to save figure
    """
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    state, obs_dict = env.reset(reset_key)

    # Store trajectories
    pursuer_positions = [state.pursuer.position]
    evader_positions = [state.evader.position]

    num_actions = num_actions_per_dim ** 2

    # Run episode with both trained agents
    for step in range(env.params.max_steps):
        # Pursuer action (greedy)
        pursuer_obs = observation_to_array(obs_dict["pursuer"])
        pursuer_q_values = pursuer_q_network(pursuer_obs)
        pursuer_action_idx = int(jnp.argmax(pursuer_q_values))
        pursuer_force = discretize_action(pursuer_action_idx, num_actions_per_dim, env.params.max_force)

        # Evader action (greedy)
        evader_obs = observation_to_array(obs_dict["evader"])
        evader_q_values = evader_q_network(evader_obs)
        evader_action_idx = int(jnp.argmax(evader_q_values))
        evader_force = discretize_action(evader_action_idx, num_actions_per_dim, env.params.max_force)

        # Step
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        state, obs_dict, rewards, done, info = env.step(state, actions)

        pursuer_positions.append(state.pursuer.position)
        evader_positions.append(state.evader.position)

        if done:
            break

    # Convert to numpy arrays
    pursuer_positions = np.array(pursuer_positions)
    evader_positions = np.array(evader_positions)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw boundary
    boundary_size = env.params.boundary_size
    if hasattr(env.boundary, 'half_size'):  # Square
        half_size = env.boundary.half_size
        rect = Rectangle(
            (-half_size, -half_size),
            boundary_size,
            boundary_size,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.set_xlim(-half_size - 1, half_size + 1)
        ax.set_ylim(-half_size - 1, half_size + 1)
    elif hasattr(env.boundary, 'vertices'):  # Triangle
        vertices = np.array(env.boundary.vertices)
        polygon = Polygon(
            vertices,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(polygon)
        margin = boundary_size * 0.1
        ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
        ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    elif hasattr(env.boundary, 'radius'):  # Circle
        radius = env.boundary.radius
        circle = MPLCircle(
            (0, 0),
            radius,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(circle)
        ax.set_xlim(-radius - 1, radius + 1)
        ax.set_ylim(-radius - 1, radius + 1)

    # Plot trajectories
    ax.plot(pursuer_positions[:, 0], pursuer_positions[:, 1],
            'b-', linewidth=2, label='Pursuer (Trained)', alpha=0.7)
    ax.plot(evader_positions[:, 0], evader_positions[:, 1],
            'r-', linewidth=2, label='Evader (Trained)', alpha=0.7)

    # Plot start positions
    ax.scatter(pursuer_positions[0, 0], pursuer_positions[0, 1],
               c='blue', s=200, marker='o', edgecolors='black',
               linewidth=2, label='Pursuer Start', zorder=5)
    ax.scatter(evader_positions[0, 0], evader_positions[0, 1],
               c='red', s=200, marker='o', edgecolors='black',
               linewidth=2, label='Evader Start', zorder=5)

    # Plot end positions
    ax.scatter(pursuer_positions[-1, 0], pursuer_positions[-1, 1],
               c='blue', s=200, marker='X', edgecolors='black',
               linewidth=2, label='Pursuer End', zorder=5)
    ax.scatter(evader_positions[-1, 0], evader_positions[-1, 1],
               c='red', s=200, marker='X', edgecolors='black',
               linewidth=2, label='Evader End', zorder=5)

    # Draw capture radius at final position
    capture_circle = MPLCircle(
        pursuer_positions[-1],
        env.params.capture_radius,
        fill=False,
        edgecolor='blue',
        linestyle='--',
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(capture_circle)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    result = "CAPTURED" if info["captured"] else "TIMEOUT"
    ax.set_title(f'Self-Play Episode ({result} after {len(pursuer_positions)-1} steps)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved self-play trajectory visualization to {save_path}")

    plt.show()


def main():
    """Example usage of plotting utilities."""
    # This is a placeholder - in practice, you'd load saved results
    # or run the training and pass the results here

    print("Plotting utilities loaded.")
    print("\nUsage:")
    print("  from plot_results import plot_training_curves, visualize_episode")
    print("  plot_training_curves(episode_rewards, episode_lengths, losses)")
    print("  visualize_episode(env, q_network)")
    print("  visualize_selfplay_episode(env, pursuer_q_network, evader_q_network)")


if __name__ == "__main__":
    main()
