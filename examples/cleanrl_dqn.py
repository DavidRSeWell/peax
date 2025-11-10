"""CleanRL-style DQN implementation for PursuerEvaderEnv.

This adapts the CleanRL JAX DQN implementation to work with the PEAX environment.
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from peax import PursuerEvaderEnv, Observation


@dataclass
class Args:
    """Hyperparameters for DQN training."""

    exp_name: str = "cleanrl_dqn_peax"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate (1.0 = hard update)"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

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


def observation_to_array(obs: Observation) -> np.ndarray:
    """Convert Observation NamedTuple to flat array."""
    return np.concatenate([
        np.array(obs.own_position),
        np.array(obs.own_velocity),
        np.array(obs.other_position),
        np.array(obs.other_velocity),
        np.array([obs.time_remaining])
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


def main():
    args = Args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # Environment setup
    env = PursuerEvaderEnv(
        boundary_type=args.boundary_type,
        boundary_size=args.boundary_size,
        max_steps=args.max_steps,
        capture_radius=args.capture_radius,
    )

    obs_dim = env.observation_space_dim
    num_actions = args.num_actions_per_dim ** 2

    print("=" * 70)
    print(f"CleanRL DQN Training on PursuerEvaderEnv")
    print("=" * 70)
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of discrete actions: {num_actions} ({args.num_actions_per_dim}x{args.num_actions_per_dim} grid)")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # Q-Network
    q_network = QNetwork(action_dim=num_actions)
    dummy_obs = jnp.zeros((obs_dim,))
    q_params = q_network.init(q_key, dummy_obs)

    # Optimizer
    optimizer = optax.adam(learning_rate=args.learning_rate)

    # Training state
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        target_params=jax.tree.map(lambda x: x, q_params),
        tx=optimizer,
    )

    # Replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(),
    )

    @jax.jit
    def update(q_state: TrainState, observations, actions, next_observations, rewards, dones):
        """Update Q-network."""
        def loss_fn(params):
            # Current Q-values
            q_pred = q_network.apply(params, observations)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]

            # Target Q-values
            q_next = q_network.apply(q_state.target_params, next_observations)
            q_next = jnp.max(q_next, axis=-1)
            q_target = rewards + args.gamma * q_next * (1 - dones)

            # MSE loss
            loss = ((q_pred - jax.lax.stop_gradient(q_target)) ** 2).mean()
            return loss, q_pred.mean()

        (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return q_state, loss, q_mean

    # Training loop
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)
    obs = observation_to_array(obs_dict["pursuer"])

    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0
    num_episodes = 0

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Epsilon schedule
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        # Select action
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            q_values = q_network.apply(q_state.params, obs)
            action = int(jnp.argmax(q_values))

        # Convert discrete action to continuous force
        pursuer_force = discretize_action(action, args.num_actions_per_dim, env.params.max_force)

        # Random evader action
        key, evader_key = jax.random.split(key)
        evader_force = jax.random.uniform(
            evader_key,
            shape=(2,),
            minval=-env.params.max_force,
            maxval=env.params.max_force
        )

        # Step environment
        actions = {"pursuer": pursuer_force, "evader": evader_force}
        next_env_state, next_obs_dict, rewards, done, info = env.step(env_state, actions)

        next_obs = observation_to_array(next_obs_dict["pursuer"])
        reward = rewards["pursuer"]

        # Store in replay buffer
        rb.add(obs, next_obs, action, reward, done)

        # Update state
        obs = next_obs
        env_state = next_env_state

        episode_reward += reward
        episode_length += 1

        # Handle episode termination
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            num_episodes += 1

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
            obs = observation_to_array(obs_dict["pursuer"])
            episode_reward = 0.0
            episode_length = 0

        # Training step
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch = rb.sample(args.batch_size)
            q_state, loss, q_mean = update(
                q_state,
                batch["observations"],
                batch["actions"],
                batch["next_observations"],
                batch["rewards"],
                batch["dones"],
            )

        # Update target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=jax.tree.map(lambda x: x, q_state.params)
            )

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
        obs = observation_to_array(obs_dict["pursuer"])

        ep_reward = 0.0

        for step in range(args.max_steps):
            # Greedy action
            q_values = q_network.apply(q_state.params, obs)
            action = int(jnp.argmax(q_values))
            pursuer_force = discretize_action(action, args.num_actions_per_dim, env.params.max_force)

            # Random evader
            key, evader_key = jax.random.split(key)
            evader_force = jax.random.uniform(
                evader_key,
                shape=(2,),
                minval=-env.params.max_force,
                maxval=env.params.max_force
            )

            actions = {"pursuer": pursuer_force, "evader": evader_force}
            env_state, obs_dict, rewards, done, info = env.step(env_state, actions)

            obs = observation_to_array(obs_dict["pursuer"])
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

    return q_state, episode_rewards, episode_lengths


if __name__ == "__main__":
    main()
