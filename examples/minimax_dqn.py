"""Minimax-Q DQN for Markov Games (Pursuer-Evader).

This implements a centralized Q-learning approach for two-player zero-sum games
based on Littman's Minimax-Q algorithm (ICML 1994). The Q-network takes the global
state and outputs Q-values for all joint action pairs.

Key differences from standard DQN:
1. Q(s, a_pursuer, a_evader) instead of Q(obs, a)
2. Global state: both agents' positions, velocities, and time
3. Action selection via minimax/maximin for zero-sum games
4. Backup: Q(s, a1, a2) = r + γ * V(s') where V(s') is the minimax value

Based on: "Markov Games as a Framework for Multi-Agent Reinforcement Learning"
by Michael L. Littman, ICML 1994
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
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
class MinimaxDQNConfig:
    """Hyperparameters for Minimax-Q DQN training."""

    exp_name: str = "minimax_dqn_peax"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.2
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_every: int = 10000
    """evaluate and log metrics every N timesteps"""
    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping"""
    reward_clip: float = 2.0
    """clip rewards to [-reward_clip, reward_clip]"""

    # Environment specific arguments
    boundary_type: str = "square"
    """boundary type for the environment"""
    boundary_size: float = 10.0
    """size of the boundary"""
    max_steps: int = 200
    """maximum steps per episode"""
    capture_radius: float = 0.5
    """capture radius"""
    num_actions_per_dim: int = 3
    """number of discrete actions per dimension"""
    wall_penalty_coef: float = 0.01
    """coefficient for wall proximity penalty"""
    velocity_reward_coef: float = 0.005
    """coefficient for velocity reward"""


cs = ConfigStore.instance()
cs.store(name="config", node=MinimaxDQNConfig)


class JointQNetwork(nn.Module):
    """Joint Q-Network that outputs Q(s, a_pursuer, a_evader)."""

    pursuer_action_dim: int
    evader_action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass.

        Args:
            x: Global state [batch, state_dim]

        Returns:
            Q-values of shape [batch, pursuer_actions, evader_actions]
        """
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        # Output joint Q-values
        x = nn.Dense(self.pursuer_action_dim * self.evader_action_dim)(x)
        # Reshape to [batch, pursuer_actions, evader_actions]
        return x.reshape(-1, self.pursuer_action_dim, self.evader_action_dim)


class TrainState(TrainState):
    """Extended TrainState to include target network parameters."""
    target_params: flax.core.FrozenDict


class ReplayBuffer:
    """Replay buffer for joint state-action transitions."""

    def __init__(self, buffer_size: int, state_shape: Sequence[int],
                 action_shape: Tuple[int, int]):
        self.buffer_size = buffer_size
        self.state_buf = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.next_state_buf = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        # Store joint actions as two separate indices
        self.pursuer_actions_buf = np.zeros((buffer_size,), dtype=np.int32)
        self.evader_actions_buf = np.zeros((buffer_size,), dtype=np.int32)
        self.rewards_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.dones_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, next_state, pursuer_action, evader_action, reward, done):
        """Add a transition to the buffer."""
        self.state_buf[self.pos] = np.array(state)
        self.next_state_buf[self.pos] = np.array(next_state)
        self.pursuer_actions_buf[self.pos] = pursuer_action
        self.evader_actions_buf[self.pos] = evader_action
        self.rewards_buf[self.pos] = reward
        self.dones_buf[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            states=self.state_buf[idxs],
            next_states=self.next_state_buf[idxs],
            pursuer_actions=self.pursuer_actions_buf[idxs],
            evader_actions=self.evader_actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            dones=self.dones_buf[idxs],
        )


def get_global_state(env_state, env: PursuerEvaderEnv) -> np.ndarray:
    """Convert environment state to global state vector.

    Args:
        env_state: Environment state
        env: Environment instance

    Returns:
        Global state: [pursuer_pos, pursuer_vel, evader_pos, evader_vel, time]
    """
    return np.concatenate([
        np.array(env_state.pursuer.position),
        np.array(env_state.pursuer.velocity),
        np.array(env_state.evader.position),
        np.array(env_state.evader.velocity),
        np.array([env_state.time / env.params.max_steps])
    ])


def discretize_action(action_idx: int, num_actions_per_dim: int, max_force: float) -> jnp.ndarray:
    """Convert discrete action index to continuous force."""
    fx_idx = action_idx // num_actions_per_dim
    fy_idx = action_idx % num_actions_per_dim
    force_values = np.linspace(-max_force, max_force, num_actions_per_dim)
    fx = force_values[fx_idx]
    fy = force_values[fy_idx]
    return jnp.array([fx, fy])


def compute_minimax_value(q_matrix: jnp.ndarray) -> float:
    """Compute minimax value for pursuer (row player maximizes, column player minimizes).

    For a zero-sum game matrix, this finds max_a1 min_a2 Q(a1, a2).

    Args:
        q_matrix: Q-values of shape [pursuer_actions, evader_actions]

    Returns:
        Minimax value
    """
    # For each pursuer action, find worst-case (minimum) over evader actions
    worst_case_values = jnp.min(q_matrix, axis=1)
    # Pursuer chooses action that maximizes worst-case value
    minimax_value = jnp.max(worst_case_values)
    return minimax_value


def get_minimax_action(q_matrix: jnp.ndarray, is_pursuer: bool) -> int:
    """Get greedy action using minimax for pursuer or maximin for evader.

    Args:
        q_matrix: Q-values of shape [pursuer_actions, evader_actions]
        is_pursuer: If True, find pursuer's maximin action; else evader's minimax action

    Returns:
        Action index
    """
    if is_pursuer:
        # Pursuer: max_a1 min_a2 Q(a1, a2)
        worst_case_values = jnp.min(q_matrix, axis=1)
        action = jnp.argmax(worst_case_values)
    else:
        # Evader: min_a2 max_a1 Q(a1, a2)
        best_case_values = jnp.max(q_matrix, axis=0)
        action = jnp.argmin(best_case_values)
    return int(action)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon schedule for exploration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def render_episode_to_gif(
    env: PursuerEvaderEnv,
    q_network: JointQNetwork,
    q_state: TrainState,
    num_actions_per_dim: int,
    key: jax.Array,
    filename: str,
    max_steps: int = 200,
    num_episodes: int = 5
) -> Dict:
    """Render episodes and save as GIF."""
    states = []
    total_reward = 0.0
    last_info = {}

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        env_state, obs_dict = env.reset(reset_key)
        states.append(env_state)

        episode_reward = 0.0

        for step in range(max_steps):
            # Get global state
            global_state = get_global_state(env_state, env)

            # Get Q-matrix
            q_matrix = q_network.apply(q_state.params, global_state[None, :])[0]

            # Get greedy actions
            pursuer_action = get_minimax_action(q_matrix, is_pursuer=True)
            evader_action = get_minimax_action(q_matrix, is_pursuer=False)

            pursuer_force = discretize_action(pursuer_action, num_actions_per_dim, env.params.max_force)
            evader_force = discretize_action(evader_action, num_actions_per_dim, env.params.max_force)

            actions_dict = {"pursuer": pursuer_force, "evader": evader_force}
            env_state, obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)

            states.append(env_state)
            episode_reward += rewards_dict["pursuer"]

            if done:
                last_info = info
                break

        total_reward += episode_reward

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

        pursuer_pos = state.pursuer.position
        evader_pos = state.evader.position

        ax.plot(pursuer_pos[0], pursuer_pos[1], 'ro', markersize=12, label='Pursuer')
        if np.linalg.norm(state.pursuer.velocity) > 0.1:
            ax.arrow(pursuer_pos[0], pursuer_pos[1],
                    state.pursuer.velocity[0] * 0.5, state.pursuer.velocity[1] * 0.5,
                    head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)

        ax.plot(evader_pos[0], evader_pos[1], 'bo', markersize=12, label='Evader')
        if np.linalg.norm(state.evader.velocity) > 0.1:
            ax.arrow(evader_pos[0], evader_pos[1],
                    state.evader.velocity[0] * 0.5, state.evader.velocity[1] * 0.5,
                    head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.6)

        capture_circle = patches.Circle(
            pursuer_pos, env.params.capture_radius,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.2
        )
        ax.add_patch(capture_circle)

        time_remaining = env.params.max_steps - state.time
        ax.set_title(f'Step {state.time}/{env.params.max_steps} | Time Remaining: {time_remaining}')
        ax.legend(loc='upper right')

        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(states),
                        interval=100, blit=True, repeat=True)

    writer = PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)

    return {
        "reward": total_reward / num_episodes,
        "captured": last_info.get("captured", False),
        "timeout": last_info.get("timeout", False),
        "length": len(states) - num_episodes
    }


@hydra.main(version_base=None, config_name="config")
def main(cfg: MinimaxDQNConfig) -> None:
    """Main training function."""
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

    # Global state: [pursuer_pos(2), pursuer_vel(2), evader_pos(2), evader_vel(2), time(1)] = 9D
    state_dim = 9
    num_actions = cfg.num_actions_per_dim ** 2

    print("=" * 70)
    print(f"Minimax-Q DQN Training on PursuerEvaderEnv")
    print("=" * 70)
    print(f"Global state dimension: {state_dim}")
    print(f"Number of discrete actions per agent: {num_actions}")
    print(f"Joint action space size: {num_actions * num_actions}")
    print(f"Total timesteps: {cfg.total_timesteps}")
    print("=" * 70)

    # Joint Q-Network
    q_network = JointQNetwork(pursuer_action_dim=num_actions, evader_action_dim=num_actions)
    dummy_state = jnp.zeros((1, state_dim))
    q_params = q_network.init(q_key, dummy_state)

    # Optimizer
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
        state_shape=(state_dim,),
        action_shape=(num_actions, num_actions),
    )

    # TensorBoard writer
    writer = SummaryWriter(f"runs/{cfg.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()]),
    )

    @jax.jit
    def update(q_state: TrainState, states, pursuer_actions, evader_actions,
               next_states, rewards, dones):
        """Update Q-network with Minimax-Q backup."""
        def loss_fn(params):
            # Clip rewards
            clipped_rewards = jnp.clip(rewards, -cfg.reward_clip, cfg.reward_clip)

            # Current Q-values for the taken actions
            q_pred_all = q_network.apply(params, states)  # [batch, num_p_actions, num_e_actions]
            q_pred = q_pred_all[jnp.arange(q_pred_all.shape[0]), pursuer_actions, evader_actions]

            # Minimax value for next states
            q_next_all = q_network.apply(q_state.target_params, next_states)
            # Compute minimax value for each next state
            minimax_values = jax.vmap(compute_minimax_value)(q_next_all)

            # Bellman target
            q_target = clipped_rewards + cfg.gamma * minimax_values * (1 - dones)

            # Huber loss
            td_error = q_pred - jax.lax.stop_gradient(q_target)
            huber_loss = optax.huber_loss(td_error, delta=1.0).mean()

            return huber_loss, (q_pred.mean(), q_target.mean(), minimax_values.max(), minimax_values.min())

        (loss, (q_mean, q_target_mean, q_max, q_min)), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
        grad_norm = optax.global_norm(grads)
        q_state = q_state.apply_gradients(grads=grads)
        return q_state, loss, q_mean, q_target_mean, grad_norm, q_max, q_min

    # Training loop
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)
    global_state = get_global_state(env_state, env)

    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0
    num_episodes = 0

    start_time = time.time()

    for global_step in range(cfg.total_timesteps):
        epsilon = linear_schedule(
            cfg.start_e,
            cfg.end_e,
            cfg.exploration_fraction * cfg.total_timesteps,
            global_step,
        )

        # Select actions
        if random.random() < epsilon:
            pursuer_action = random.randint(0, num_actions - 1)
            evader_action = random.randint(0, num_actions - 1)
        else:
            q_matrix = q_network.apply(q_state.params, global_state[None, :])[0]
            pursuer_action = get_minimax_action(q_matrix, is_pursuer=True)
            evader_action = get_minimax_action(q_matrix, is_pursuer=False)

        # Convert to forces
        pursuer_force = discretize_action(pursuer_action, cfg.num_actions_per_dim, env.params.max_force)
        evader_force = discretize_action(evader_action, cfg.num_actions_per_dim, env.params.max_force)

        # Step environment
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        next_env_state, next_obs_dict, rewards, done, info = env.step(env_state, actions)
        next_global_state = get_global_state(next_env_state, env)

        pursuer_reward = rewards["pursuer"]

        # Store transition
        rb.add(global_state, next_global_state, pursuer_action, evader_action, pursuer_reward, done)

        # Update state
        global_state = next_global_state
        env_state = next_env_state

        episode_reward += pursuer_reward
        episode_length += 1

        # Handle episode termination
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            num_episodes += 1

            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

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
            global_state = get_global_state(env_state, env)
            episode_reward = 0.0
            episode_length = 0

        # Training step
        if global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0:
            batch = rb.sample(cfg.batch_size)
            q_state, loss, q_mean, q_target_mean, grad_norm, q_max, q_min = update(
                q_state,
                batch["states"],
                batch["pursuer_actions"],
                batch["evader_actions"],
                batch["next_states"],
                batch["rewards"],
                batch["dones"],
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", float(loss), global_step)
                writer.add_scalar("losses/q_values", float(q_mean), global_step)
                writer.add_scalar("losses/q_targets", float(q_target_mean), global_step)
                writer.add_scalar("losses/grad_norm", float(grad_norm), global_step)
                writer.add_scalar("losses/q_max", float(q_max), global_step)
                writer.add_scalar("losses/q_min", float(q_min), global_step)

        # Update target network
        if global_step % cfg.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=jax.tree.map(lambda x: x, q_state.params)
            )

        # Evaluation
        if global_step > 0 and global_step % cfg.eval_every == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at step {global_step}")
            print(f"{'='*70}")

            eval_episodes = 10
            eval_rewards = []
            eval_captures = 0
            eval_timeouts = 0

            for eval_ep in range(eval_episodes):
                key, eval_key = jax.random.split(key)
                eval_state, eval_obs_dict = env.reset(eval_key)
                eval_global_state = get_global_state(eval_state, env)
                eval_reward = 0.0

                for eval_step in range(cfg.max_steps):
                    q_matrix = q_network.apply(q_state.params, eval_global_state[None, :])[0]
                    eval_pursuer_action = get_minimax_action(q_matrix, is_pursuer=True)
                    eval_evader_action = get_minimax_action(q_matrix, is_pursuer=False)

                    eval_pursuer_force = discretize_action(eval_pursuer_action, cfg.num_actions_per_dim, env.params.max_force)
                    eval_evader_force = discretize_action(eval_evader_action, cfg.num_actions_per_dim, env.params.max_force)

                    eval_actions = {"pursuer": eval_pursuer_force, "evader": eval_evader_force}
                    eval_state, eval_obs_dict, eval_rewards_dict, eval_done, eval_info = env.step(eval_state, eval_actions)
                    eval_global_state = get_global_state(eval_state, env)
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

            writer.add_scalar("eval/average_return", avg_eval_reward, global_step)
            writer.add_scalar("eval/capture_rate", eval_capture_rate, global_step)
            writer.add_scalar("eval/timeout_rate", eval_timeout_rate, global_step)

            # Generate GIF
            print("Generating evaluation GIF...")
            gif_path = f"minimax_eval_step_{global_step}.gif"
            key, gif_key = jax.random.split(key)
            episode_info = render_episode_to_gif(
                env, q_network, q_state, cfg.num_actions_per_dim,
                gif_key, gif_path, cfg.max_steps
            )
            print(f"GIF saved to {gif_path}")

            # Save checkpoint
            checkpoint_path = f"minimax_checkpoint_step_{global_step}.pkl"
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

    writer.close()
    return q_state, episode_rewards, episode_lengths


if __name__ == "__main__":
    main()
