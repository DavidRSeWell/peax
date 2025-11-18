"""Deep Q-Network (DQN) for training a pursuer agent.

This example demonstrates how to train a pursuer agent using DQN with the
Equinox library. The pursuer learns to catch a random evader.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import NamedTuple, Tuple
from collections import deque
import numpy as np

from peax import PursuerEvaderEnv, EnvState, Observation


class ReplayBuffer:
    """Experience replay buffer for DQN.

    Stores transitions and samples random batches for training.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Dimension of observations
            action_dim: Dimension of actions (for discrete actions, this is 1)
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Pre-allocate arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.size = 0
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, key: jax.Array):
        """Sample a random batch of transitions."""
        indices = jax.random.choice(
            key,
            self.size,
            shape=(batch_size,),
            replace=False
        )

        return (
            jnp.array(self.observations[indices]),
            jnp.array(self.actions[indices]),
            jnp.array(self.rewards[indices]),
            jnp.array(self.next_observations[indices]),
            jnp.array(self.dones[indices]),
        )

    def __len__(self):
        """Return current size of buffer."""
        return self.size


class QNetwork(eqx.Module):
    """Q-Network for discrete action DQN using Equinox MLP.

    Maps observations to Q-values for each discrete action.
    """

    mlp: eqx.nn.MLP

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int, key: jax.Array):
        """Initialize Q-Network.

        Args:
            obs_dim: Dimension of observation space
            num_actions: Number of discrete actions
            hidden_size: Size of hidden layers
            key: JAX random key
        """
        self.mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=num_actions,
            width_size=hidden_size,
            depth=3,  # 2 hidden layers + output layer
            activation=jax.nn.relu,
            key=key
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            obs: Observation array

        Returns:
            Q-values for each action
        """
        return self.mlp(obs)


def observation_to_array(obs: Observation) -> jax.Array:
    """Convert Observation NamedTuple to flat array.

    Args:
        obs: Observation from environment

    Returns:
        10D array: [own_pos(2), own_vel(2), other_pos(2), other_vel(2), time(1), agent_id(1)]
        agent_id is critical for self-play with shared policies
    """
    return jnp.concatenate([
        obs.own_position,
        obs.own_velocity,
        obs.other_position,
        obs.other_velocity,
        jnp.array([obs.time_remaining]),
        jnp.array([obs.agent_id])  # CRITICAL: tells network if it's pursuer or evader!
    ])


def discretize_action(action_idx: int, num_actions_per_dim: int, max_force: float) -> jax.Array:
    """Convert discrete action index to continuous force.

    We discretize the 2D action space into a grid.

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
    force_values = jnp.linspace(-max_force, max_force, num_actions_per_dim)
    fx = force_values[fx_idx]
    fy = force_values[fy_idx]

    return jnp.array([fx, fy])


def select_action(
    q_network: QNetwork,
    obs: jax.Array,
    epsilon: float,
    num_actions: int,
    key: jax.Array
) -> int:
    """Select action using epsilon-greedy policy.

    Args:
        q_network: Q-network
        obs: Current observation
        epsilon: Exploration rate
        num_actions: Total number of discrete actions
        key: JAX random key

    Returns:
        Selected action index
    """
    key1, key2 = jax.random.split(key)

    # Epsilon-greedy
    explore = jax.random.uniform(key1) < epsilon

    # Random action
    random_action = jax.random.choice(key2, num_actions)

    # Greedy action
    q_values = q_network(obs)
    greedy_action = jnp.argmax(q_values)

    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


@eqx.filter_jit
def compute_td_loss(
    q_network: QNetwork,
    target_network: QNetwork,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    gamma: float
) -> jax.Array:
    """Compute temporal difference loss for DQN.

    Args:
        q_network: Current Q-network
        target_network: Target Q-network
        observations: Batch of observations
        actions: Batch of actions taken
        rewards: Batch of rewards
        next_observations: Batch of next observations
        dones: Batch of done flags
        gamma: Discount factor

    Returns:
        Mean TD loss
    """
    # Current Q-values
    q_values = jax.vmap(q_network)(observations)  # (batch_size, num_actions)
    q_values_taken = q_values[jnp.arange(len(actions)), actions]  # (batch_size,)

    # Target Q-values
    next_q_values = jax.vmap(target_network)(next_observations)  # (batch_size, num_actions)
    next_q_values_max = jnp.max(next_q_values, axis=1)  # (batch_size,)

    # Compute target
    targets = rewards + gamma * next_q_values_max * (1 - dones)

    # MSE loss
    loss = jnp.mean((q_values_taken - jax.lax.stop_gradient(targets)) ** 2)

    return loss


@eqx.filter_jit
def update_step(
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    gamma: float
) -> Tuple[QNetwork, optax.OptState, jax.Array]:
    """Perform one gradient update step.

    Args:
        q_network: Current Q-network
        target_network: Target Q-network
        optimizer: Optax optimizer
        opt_state: Optimizer state
        observations: Batch of observations
        actions: Batch of actions
        rewards: Batch of rewards
        next_observations: Batch of next observations
        dones: Batch of done flags
        gamma: Discount factor

    Returns:
        Tuple of (updated Q-network, updated optimizer state, loss value)
    """
    loss, grads = eqx.filter_value_and_grad(compute_td_loss)(
        q_network,
        target_network,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        gamma
    )

    updates, opt_state = optimizer.update(grads, opt_state, q_network)
    q_network = eqx.apply_updates(q_network, updates)

    return q_network, opt_state, loss


def train_dqn(
    env: PursuerEvaderEnv,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    buffer_size: int = 10000,
    target_update_freq: int = 10,
    num_actions_per_dim: int = 5,
    hidden_size: int = 128,
    seed: int = 0
):
    """Train a DQN agent to be a pursuer.

    Args:
        env: PursuerEvaderEnv environment
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        batch_size: Batch size for training
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate per episode
        buffer_size: Size of replay buffer
        target_update_freq: Frequency of target network updates (in episodes)
        num_actions_per_dim: Number of discrete actions per dimension
        hidden_size: Size of hidden layers in Q-network
        seed: Random seed
    """
    # Initialize
    key = jax.random.PRNGKey(seed)
    key, qnet_key = jax.random.split(key)

    obs_dim = env.observation_space_dim
    num_actions = num_actions_per_dim ** 2  # Grid of actions in 2D

    # Create Q-network and target network
    q_network = QNetwork(obs_dim, num_actions, hidden_size, qnet_key)
    target_network = q_network  # Initialize target network as copy

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(q_network, eqx.is_array))

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size, obs_dim, 1)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    epsilon = epsilon_start

    print("=" * 60)
    print("Training DQN Pursuer Agent")
    print("=" * 60)
    print(f"Observation dim: {obs_dim}")
    print(f"Number of discrete actions: {num_actions}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Get pursuer observation
            pursuer_obs = obs_dict["pursuer"]
            pursuer_obs_array = np.array(observation_to_array(pursuer_obs))

            # Select action for pursuer
            key, action_key = jax.random.split(key)
            action_idx = int(select_action(
                q_network,
                jnp.array(pursuer_obs_array),
                epsilon,
                num_actions,
                action_key
            ))
            pursuer_force = discretize_action(action_idx, num_actions_per_dim, env.params.max_force)

            # Random action for evader
            key, evader_key = jax.random.split(key)
            evader_force = jax.random.uniform(
                evader_key,
                shape=(2,),
                minval=-env.params.max_force,
                maxval=env.params.max_force
            )

            # Step environment
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            next_state, next_obs_dict, rewards, done, info = env.step(state, actions)

            # Store transition
            next_pursuer_obs_array = np.array(observation_to_array(next_obs_dict["pursuer"]))
            replay_buffer.add(
                pursuer_obs_array,
                action_idx,
                rewards["pursuer"],
                next_pursuer_obs_array,
                done
            )

            episode_reward += rewards["pursuer"]
            episode_length += 1

            # Update state
            state = next_state
            obs_dict = next_obs_dict

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                key, sample_key = jax.random.split(key)
                batch = replay_buffer.sample(batch_size, sample_key)

                q_network, opt_state, loss = update_step(
                    q_network,
                    target_network,
                    optimizer,
                    opt_state,
                    *batch,
                    gamma
                )
                losses.append(float(loss))

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update target network
        if episode % target_update_freq == 0:
            target_network = q_network

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            avg_loss = np.mean(losses[-100:]) if losses else 0

            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Loss: {avg_loss:8.4f} | "
                  f"Epsilon: {epsilon:.3f}")

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)

    return q_network, episode_rewards, episode_lengths, losses


def evaluate_agent(
    env: PursuerEvaderEnv,
    q_network: QNetwork,
    num_episodes: int = 10,
    num_actions_per_dim: int = 5,
    seed: int = 42
):
    """Evaluate trained DQN agent.

    Args:
        env: Environment
        q_network: Trained Q-network
        num_episodes: Number of evaluation episodes
        num_actions_per_dim: Number of discrete actions per dimension
        seed: Random seed
    """
    key = jax.random.PRNGKey(seed)
    num_actions = num_actions_per_dim ** 2

    episode_rewards = []
    captures = []
    timeouts = []

    print("\n" + "=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward = 0.0

        for step in range(env.params.max_steps):
            # Get pursuer observation (greedy action, no exploration)
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

            episode_reward += rewards["pursuer"]

            if done:
                captures.append(info["captured"])
                timeouts.append(info["timeout"])
                break

        episode_rewards.append(episode_reward)

        result = "CAPTURED" if info["captured"] else "TIMEOUT"
        print(f"Episode {episode+1}: Reward = {episode_reward:6.2f}, {result}")

    print("=" * 60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Capture Rate: {np.mean(captures)*100:.1f}%")
    print(f"Timeout Rate: {np.mean(timeouts)*100:.1f}%")
    print("=" * 60)


def main():
    """Main training script."""
    # Create environment
    env = PursuerEvaderEnv(
        boundary_type="square",
        boundary_size=10.0,
        max_steps=200,
        capture_radius=0.5,
    )

    # Train agent
    q_network, episode_rewards, episode_lengths, losses = train_dqn(
        env,
        num_episodes=500,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        target_update_freq=10,
        num_actions_per_dim=5,
        hidden_size=128,
        seed=0
    )

    # Evaluate agent
    evaluate_agent(env, q_network, num_episodes=20, num_actions_per_dim=5)

    # Optionally visualize results (requires matplotlib)
    try:
        from plot_results import plot_training_curves, visualize_episode

        print("\nGenerating visualizations...")
        plot_training_curves(episode_rewards, episode_lengths, losses)
        visualize_episode(env, q_network, num_actions_per_dim=5, seed=123)
    except ImportError:
        print("\nSkipping visualization (matplotlib not installed)")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
