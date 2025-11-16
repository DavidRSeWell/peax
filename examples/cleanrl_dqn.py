"""CleanRL-style DQN implementation with self-play for PursuerEvaderEnv.

This adapts the CleanRL JAX DQN implementation to work with the PEAX environment
using self-play. A single Q-network is trained to play optimally from both the
pursuer and evader perspectives. Both agents use the same network and both agents'
experiences contribute to training.

Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence, Dict
from pathlib import Path

import flax
import flax.linen as nn
import hydra
from hydra.core.config_store import ConfigStore
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle

from peax import PursuerEvaderEnv, Observation


@dataclass
class DQNConfig:
    """Hyperparameters for DQN training."""

    exp_name: str = "cleanrl_dqn_peax_selfplay"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer (reduced for stability)"""
    buffer_size: int = 50000
    """the replay memory buffer size (increased to reduce overfitting)"""
    gamma: float = 0.9
    """the discount factor gamma (reduced to limit value accumulation)"""
    tau: float = 1.0
    """the target network update rate (1.0 = hard update)"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration (increased for more exploration)"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_every: int = 10000
    """evaluate and log metrics every N timesteps"""
    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping (reduced for stability)"""
    reward_clip: float = 2.0
    """clip rewards to [-reward_clip, reward_clip] (reduced for stability)"""

    # Environment specific arguments
    boundary_type: str = "square"
    """boundary type for the environment"""
    boundary_size: float = 10.0
    """size of the boundary"""
    max_steps: int = 200
    """maximum steps per episode"""
    capture_radius: float = 0.5
    """capture radius"""
    num_actions_per_dim: int = 5
    """number of discrete actions per dimension (total actions = this squared)"""
    wall_penalty_coef: float = 0.001
    """coefficient for wall proximity penalty (0.0 = disabled)"""
    velocity_reward_coef: float = 0.001
    """coefficient for velocity reward (0.0 = disabled)"""


# Register the config with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=DQNConfig)


class QNetwork(nn.Module):
    """Q-Network using Flax."""

    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    """Extended TrainState to include target network parameters."""

    target_params: flax.core.FrozenDict


class ReplayBuffer:
    """Simple replay buffer for storing transitions."""

    def __init__(self, buffer_size: int, obs_shape: Sequence[int], action_shape: Sequence[int]):
        self.buffer_size = buffer_size
        self.obs_buf = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions_buf = np.zeros((buffer_size, *action_shape), dtype=np.int32)
        self.rewards_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.dones_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        """Add a transition to the buffer."""
        self.obs_buf[self.pos] = np.array(obs)
        self.next_obs_buf[self.pos] = np.array(next_obs)
        self.actions_buf[self.pos] = np.array(action)
        self.rewards_buf[self.pos] = np.array(reward)
        self.dones_buf[self.pos] = np.array(done)
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            observations=self.obs_buf[idxs],
            next_observations=self.next_obs_buf[idxs],
            actions=self.actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            dones=self.dones_buf[idxs],
        )


def observation_to_array(obs: Observation, boundary_size: float = 10.0, max_velocity: float = 20.0) -> np.ndarray:
    """Convert Observation NamedTuple to normalized flat array.

    Args:
        obs: Observation namedtuple
        boundary_size: Size of the boundary for normalizing positions
        max_velocity: Maximum expected velocity for normalization

    Returns:
        Normalized observation array with values roughly in [-1, 1]
    """
    return np.concatenate([
        np.array(obs.own_position) / (boundary_size / 2),  # Normalize to ~[-1, 1]
        np.array(obs.own_velocity) / max_velocity,  # Normalize velocities
        np.array(obs.other_position) / (boundary_size / 2),  # Normalize to ~[-1, 1]
        np.array(obs.other_velocity) / max_velocity,  # Normalize velocities
        np.array([obs.time_remaining])  # Already in [0, 1]
    ])


def discretize_action(action_idx: int, num_actions_per_dim: int, max_force: float) -> jnp.ndarray:
    """Convert discrete action index to continuous force.

    Args:
        action_idx: Discrete action index
        num_actions_per_dim: Number of discrete actions per dimension
        max_force: Maximum force magnitude

    Returns:
        2D force vector
    """
    # Convert 1D index to 2D grid coordinates
    fx_idx = action_idx // num_actions_per_dim
    fy_idx = action_idx % num_actions_per_dim

    # Map to force values
    force_values = np.linspace(-max_force, max_force, num_actions_per_dim)
    fx = force_values[fx_idx]
    fy = force_values[fy_idx]

    return jnp.array([fx, fy])


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon schedule for exploration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def render_episode_to_gif(
    env: PursuerEvaderEnv,
    q_network: QNetwork,
    q_state: TrainState,
    num_actions_per_dim: int,
    key: jax.Array,
    filename: str,
    max_steps: int = 200
) -> Dict:
    """Render an episode and save as GIF.

    Args:
        env: Environment
        q_network: Q-network
        q_state: Training state
        num_actions_per_dim: Number of actions per dimension
        key: JAX random key
        filename: Output GIF filename
        max_steps: Maximum steps to render

    Returns:
        Dict with episode info (reward, captured, timeout, length)
    """
    # Run episode and collect states
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    states = [env_state]
    episode_reward = 0.0

    for step in range(max_steps):
        # Greedy action for pursuer
        pursuer_obs = observation_to_array(obs_dict["pursuer"], env.params.boundary_size)
        q_values = q_network.apply(q_state.params, pursuer_obs)
        action = int(jnp.argmax(q_values))
        pursuer_force = discretize_action(action, num_actions_per_dim, env.params.max_force)

        # Greedy action for evader
        evader_obs = observation_to_array(obs_dict["evader"], env.params.boundary_size)
        q_values_evader = q_network.apply(q_state.params, evader_obs)
        action_evader = int(jnp.argmax(q_values_evader))
        evader_force = discretize_action(action_evader, num_actions_per_dim, env.params.max_force)

        # Step environment
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        env_state, obs_dict, rewards, done, info = env.step(env_state, actions)

        states.append(env_state)
        episode_reward += rewards["pursuer"]

        if done:
            break

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))

    def init():
        ax.clear()
        ax.set_xlim(-env.params.boundary_size * 0.6, env.params.boundary_size * 0.6)
        ax.set_ylim(-env.params.boundary_size * 0.6, env.params.boundary_size * 0.6)
        ax.set_aspect('equal')
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(-env.params.boundary_size * 0.6, env.params.boundary_size * 0.6)
        ax.set_ylim(-env.params.boundary_size * 0.6, env.params.boundary_size * 0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        state = states[frame]

        # Draw boundary
        boundary_type = type(env.boundary).__name__
        if "Square" in boundary_type:
            size = env.params.boundary_size / 2
            rect = patches.Rectangle(
                (-size, -size), size * 2, size * 2,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(rect)
        elif "Circle" in boundary_type:
            circle = patches.Circle(
                (0, 0), env.params.boundary_size / 2,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(circle)

        # Draw agents
        pursuer_pos = state.pursuer.position
        evader_pos = state.evader.position

        # Pursuer (red)
        ax.plot(pursuer_pos[0], pursuer_pos[1], 'ro', markersize=12, label='Pursuer')
        # Draw velocity arrow
        if np.linalg.norm(state.pursuer.velocity) > 0.1:
            ax.arrow(pursuer_pos[0], pursuer_pos[1],
                    state.pursuer.velocity[0] * 0.5, state.pursuer.velocity[1] * 0.5,
                    head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)

        # Evader (blue)
        ax.plot(evader_pos[0], evader_pos[1], 'bo', markersize=12, label='Evader')
        # Draw velocity arrow
        if np.linalg.norm(state.evader.velocity) > 0.1:
            ax.arrow(evader_pos[0], evader_pos[1],
                    state.evader.velocity[0] * 0.5, state.evader.velocity[1] * 0.5,
                    head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.6)

        # Draw capture radius
        capture_circle = patches.Circle(
            pursuer_pos, env.params.capture_radius,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.2
        )
        ax.add_patch(capture_circle)

        # Title with info
        time_remaining = env.params.max_steps - state.time
        ax.set_title(f'Step {state.time}/{env.params.max_steps} | Time Remaining: {time_remaining}')
        ax.legend(loc='upper right')

        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(states),
                        interval=100, blit=True, repeat=True)

    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)

    return {
        "reward": episode_reward,
        "captured": info["captured"],
        "timeout": info["timeout"],
        "length": len(states) - 1
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DQNConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object
    """
    # Print the configuration
    print("=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    key, q_key = jax.random.split(key, 2)

    # Environment setup
    env = PursuerEvaderEnv(
        boundary_type=cfg.boundary_type,
        boundary_size=cfg.boundary_size,
        max_steps=cfg.max_steps,
        capture_radius=cfg.capture_radius,
        wall_penalty_coef=cfg.wall_penalty_coef,
        velocity_reward_coef=cfg.velocity_reward_coef,
    )

    obs_dim = env.observation_space_dim
    num_actions = cfg.num_actions_per_dim ** 2

    print("=" * 70)
    print(f"CleanRL DQN Self-Play Training on PursuerEvaderEnv")
    print("=" * 70)
    print(f"Training mode: Self-play (single network for both agents)")
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of discrete actions: {num_actions} ({cfg.num_actions_per_dim}x{cfg.num_actions_per_dim} grid)")
    print(f"Total timesteps: {cfg.total_timesteps}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Buffer size: {cfg.buffer_size}")
    print(f"Batch size: {cfg.batch_size}")
    print("=" * 70)

    # Q-Network
    q_network = QNetwork(action_dim=num_actions)
    dummy_obs = jnp.zeros((obs_dim,))
    q_params = q_network.init(q_key, dummy_obs)

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate),
    )

    # Training state
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        target_params=jax.tree.map(lambda x: x, q_params),
        tx=optimizer,
    )

    # Replay buffer
    rb = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(),
    )

    # TensorBoard writer
    writer = SummaryWriter(f"runs/{cfg.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()]),
    )

    @jax.jit
    def update(q_state: TrainState, observations, actions, next_observations, rewards, dones):
        """Update Q-network with Double DQN, Huber loss, and reward clipping."""
        def loss_fn(params):
            # Clip rewards for stability
            clipped_rewards = jnp.clip(rewards, -cfg.reward_clip, cfg.reward_clip)

            # Current Q-values
            q_pred = q_network.apply(params, observations)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]

            # Double DQN: use online network to select actions, target network to evaluate
            # This reduces overestimation bias
            q_next_online = q_network.apply(params, next_observations)
            next_actions = jnp.argmax(q_next_online, axis=-1)  # Select with online network

            q_next_target = q_network.apply(q_state.target_params, next_observations)
            q_next = q_next_target[jnp.arange(q_next_target.shape[0]), next_actions]  # Evaluate with target network

            q_target = clipped_rewards + cfg.gamma * q_next * (1 - dones)

            # Huber loss (more robust to outliers than MSE)
            td_error = q_pred - jax.lax.stop_gradient(q_target)
            huber_loss = optax.huber_loss(td_error, delta=1.0).mean()

            return huber_loss, (q_pred.mean(), q_target.mean(), q_next.max(), q_next.min())

        (loss, (q_mean, q_target_mean, q_max, q_min)), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)

        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads)

        q_state = q_state.apply_gradients(grads=grads)
        return q_state, loss, q_mean, q_target_mean, grad_norm, q_max, q_min

    # Training loop
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)
    pursuer_obs = observation_to_array(obs_dict["pursuer"], cfg.boundary_size)
    evader_obs = observation_to_array(obs_dict["evader"], cfg.boundary_size)

    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0
    num_episodes = 0

    start_time = time.time()

    for global_step in range(cfg.total_timesteps):
        # Epsilon schedule
        epsilon = linear_schedule(
            cfg.start_e,
            cfg.end_e,
            cfg.exploration_fraction * cfg.total_timesteps,
            global_step,
        )

        # Select action for pursuer
        if random.random() < epsilon:
            pursuer_action = random.randint(0, num_actions - 1)
        else:
            q_values = q_network.apply(q_state.params, pursuer_obs)
            pursuer_action = int(jnp.argmax(q_values))

        # Select action for evader (using same Q-network)
        if random.random() < epsilon:
            evader_action = random.randint(0, num_actions - 1)
        else:
            q_values_evader = q_network.apply(q_state.params, evader_obs)
            evader_action = int(jnp.argmax(q_values_evader))

        # Convert discrete actions to continuous forces
        pursuer_force = discretize_action(pursuer_action, cfg.num_actions_per_dim, env.params.max_force)
        evader_force = discretize_action(evader_action, cfg.num_actions_per_dim, env.params.max_force)

        # Step environment
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        next_env_state, next_obs_dict, rewards, done, info = env.step(env_state, actions)

        next_pursuer_obs = observation_to_array(next_obs_dict["pursuer"], cfg.boundary_size)
        next_evader_obs = observation_to_array(next_obs_dict["evader"], cfg.boundary_size)
        pursuer_reward = rewards["pursuer"]
        evader_reward = rewards["evader"]

        # Store both agents' experiences in replay buffer
        rb.add(pursuer_obs, next_pursuer_obs, pursuer_action, pursuer_reward, done)
        rb.add(evader_obs, next_evader_obs, evader_action, evader_reward, done)

        # Update state
        pursuer_obs = next_pursuer_obs
        evader_obs = next_evader_obs
        env_state = next_env_state

        episode_reward += pursuer_reward
        episode_length += 1

        # Handle episode termination
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            num_episodes += 1

            # Log episode metrics to TensorBoard
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            writer.add_scalar("charts/episode_number", num_episodes, global_step)

            if num_episodes % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                sps = int(global_step / (time.time() - start_time))
                print(f"Step {global_step:6d} | Episode {num_episodes:4d} | "
                      f"Reward: {avg_reward:6.2f} | Length: {avg_length:5.1f} | "
                      f"Epsilon: {epsilon:.3f} | SPS: {sps}")

            # Reset environment
            key, reset_key = jax.random.split(key)
            env_state, obs_dict = env.reset(reset_key)
            pursuer_obs = observation_to_array(obs_dict["pursuer"], cfg.boundary_size)
            evader_obs = observation_to_array(obs_dict["evader"], cfg.boundary_size)
            episode_reward = 0.0
            episode_length = 0

        # Training step
        if global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0:
            batch = rb.sample(cfg.batch_size)
            q_state, loss, q_mean, q_target_mean, grad_norm, q_max, q_min = update(
                q_state,
                batch["observations"],
                batch["actions"],
                batch["next_observations"],
                batch["rewards"],
                batch["dones"],
            )

            # Log training metrics
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", float(loss), global_step)
                writer.add_scalar("losses/q_values", float(q_mean), global_step)
                writer.add_scalar("losses/q_targets", float(q_target_mean), global_step)
                writer.add_scalar("losses/grad_norm", float(grad_norm), global_step)
                writer.add_scalar("losses/q_max", float(q_max), global_step)
                writer.add_scalar("losses/q_min", float(q_min), global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)

        # Update target network
        if global_step % cfg.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=jax.tree.map(lambda x: x, q_state.params)
            )

        # Evaluation with GIF generation and checkpoint saving
        if global_step > 0 and global_step % cfg.eval_every == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at step {global_step}")
            print(f"{'='*70}")

            # Run evaluation episodes
            eval_episodes = 10
            eval_rewards = []
            eval_captures = 0
            eval_timeouts = 0

            for eval_ep in range(eval_episodes):
                key, eval_key = jax.random.split(key)
                eval_state, eval_obs_dict = env.reset(eval_key)
                eval_pursuer_obs = observation_to_array(eval_obs_dict["pursuer"], cfg.boundary_size)
                eval_evader_obs = observation_to_array(eval_obs_dict["evader"], cfg.boundary_size)
                eval_reward = 0.0

                for eval_step in range(cfg.max_steps):
                    # Greedy action for pursuer
                    q_values = q_network.apply(q_state.params, eval_pursuer_obs)
                    eval_pursuer_action = int(jnp.argmax(q_values))
                    eval_pursuer_force = discretize_action(eval_pursuer_action, cfg.num_actions_per_dim, env.params.max_force)

                    # Greedy action for evader
                    q_values_evader = q_network.apply(q_state.params, eval_evader_obs)
                    eval_evader_action = int(jnp.argmax(q_values_evader))
                    eval_evader_force = discretize_action(eval_evader_action, cfg.num_actions_per_dim, env.params.max_force)

                    eval_actions = {"pursuer": eval_pursuer_force, "evader": eval_evader_force}
                    eval_state, eval_obs_dict, eval_rewards_dict, eval_done, eval_info = env.step(eval_state, eval_actions)

                    eval_pursuer_obs = observation_to_array(eval_obs_dict["pursuer"], cfg.boundary_size)
                    eval_evader_obs = observation_to_array(eval_obs_dict["evader"], cfg.boundary_size)
                    eval_reward += eval_rewards_dict["pursuer"]

                    if eval_done:
                        if eval_info["captured"]:
                            eval_captures += 1
                        if eval_info["timeout"]:
                            eval_timeouts += 1
                        break

                eval_rewards.append(eval_reward)

            avg_eval_reward = np.mean(eval_rewards)
            eval_capture_rate = eval_captures / eval_episodes
            eval_timeout_rate = eval_timeouts / eval_episodes

            print(f"Average eval reward: {avg_eval_reward:.2f}")
            print(f"Capture rate: {eval_capture_rate*100:.1f}%")
            print(f"Timeout rate: {eval_timeout_rate*100:.1f}%")

            # Log evaluation metrics
            writer.add_scalar("eval/average_return", avg_eval_reward, global_step)
            writer.add_scalar("eval/capture_rate", eval_capture_rate, global_step)
            writer.add_scalar("eval/timeout_rate", eval_timeout_rate, global_step)

            # Generate and log GIF
            print("Generating evaluation GIF...")
            gif_path = f"eval_episode_step_{global_step}.gif"
            key, gif_key = jax.random.split(key)
            episode_info = render_episode_to_gif(
                env, q_network, q_state, cfg.num_actions_per_dim,
                gif_key, gif_path, cfg.max_steps
            )
            print(f"GIF saved to {gif_path}")
            print(f"  Episode reward: {episode_info['reward']:.2f}")
            print(f"  Captured: {episode_info['captured']}")
            print(f"  Timeout: {episode_info['timeout']}")
            print(f"  Length: {episode_info['length']}")

            # Log GIF to TensorBoard
            try:
                from PIL import Image
                gif_image = Image.open(gif_path)
                frames = []
                try:
                    while True:
                        frames.append(np.array(gif_image.convert('RGB')))
                        gif_image.seek(gif_image.tell() + 1)
                except EOFError:
                    pass
                # Log first frame as image
                if frames:
                    writer.add_image("eval/episode_visualization", frames[0], global_step, dataformats='HWC')
            except Exception as e:
                print(f"Warning: Could not log GIF to TensorBoard: {e}")

            # Save checkpoint
            checkpoint_path = f"checkpoint_step_{global_step}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': q_state.params,
                    'target_params': q_state.target_params,
                    'step': global_step,
                    'episode': num_episodes,
                    'config': dict(cfg)
                }, f)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(f"{'='*70}\n")

    print("=" * 70)
    print("Training completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 70)

    # Evaluation
    print("\nEvaluating agent...")
    eval_episodes = 20
    eval_rewards = []
    captures = 0
    timeouts = 0

    for ep in range(eval_episodes):
        key, reset_key = jax.random.split(key)
        env_state, obs_dict = env.reset(reset_key)
        pursuer_obs = observation_to_array(obs_dict["pursuer"], cfg.boundary_size)
        evader_obs = observation_to_array(obs_dict["evader"], cfg.boundary_size)

        ep_reward = 0.0

        for step in range(cfg.max_steps):
            # Greedy action for pursuer
            q_values = q_network.apply(q_state.params, pursuer_obs)
            pursuer_action = int(jnp.argmax(q_values))
            pursuer_force = discretize_action(pursuer_action, cfg.num_actions_per_dim, env.params.max_force)

            # Greedy action for evader
            q_values_evader = q_network.apply(q_state.params, evader_obs)
            evader_action = int(jnp.argmax(q_values_evader))
            evader_force = discretize_action(evader_action, cfg.num_actions_per_dim, env.params.max_force)

            actions = {"pursuer": pursuer_force, "evader": evader_force}
            env_state, obs_dict, rewards, done, info = env.step(env_state, actions)

            pursuer_obs = observation_to_array(obs_dict["pursuer"], cfg.boundary_size)
            evader_obs = observation_to_array(obs_dict["evader"], cfg.boundary_size)
            ep_reward += rewards["pursuer"]

            if done:
                if info["captured"]:
                    captures += 1
                if info["timeout"]:
                    timeouts += 1
                break

        eval_rewards.append(ep_reward)
        result = "CAPTURED" if info["captured"] else "TIMEOUT"
        print(f"  Episode {ep+1:2d}: Reward = {ep_reward:6.2f} | {result}")

    print("=" * 70)
    print(f"Evaluation Results ({eval_episodes} episodes)")
    print(f"Average reward: {np.mean(eval_rewards):.2f}")
    print(f"Capture rate: {captures/eval_episodes*100:.1f}%")
    print(f"Timeout rate: {timeouts/eval_episodes*100:.1f}%")
    print("=" * 70)

    # Close TensorBoard writer
    writer.close()

    return q_state, episode_rewards, episode_lengths


if __name__ == "__main__":
    main()
