"""Deep Q-Network (DQN) with Shared Network for Self-Play.

This example demonstrates self-play where both the pursuer and evader share
a single Q-network. This creates a symmetric learning scenario where the
network learns an optimal strategy that works for both roles.

This approach is similar to AlphaGo Zero where a single network plays both
sides of the game.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from collections import deque
from peax import PursuerEvaderEnv, EnvState, Observation
from tqdm import tqdm
from typing import NamedTuple, Tuple, Dict


class ReplayBuffer:
    """Experience replay buffer for DQN using JAX arrays.

    Stores transitions and samples random batches for training.
    Uses JAX arrays to avoid CPU synchronization and enable GPU storage.
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

        # Pre-allocate JAX arrays (will stay on GPU if available)
        self.observations = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity,), dtype=jnp.float32)
        self.next_observations = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.dones = jnp.zeros((capacity,), dtype=jnp.bool_)

        self.size = 0
        self.position = 0

    def add(
        self,
        obs: jax.Array,
        action: int,
        reward: float,
        next_obs: jax.Array,
        done: bool
    ):
        """Add a transition to the buffer.

        Args:
            obs: Observation (JAX array)
            action: Action index
            reward: Reward value
            next_obs: Next observation (JAX array)
            done: Done flag
        """
        # Update arrays in-place using JAX's at[] syntax
        self.observations = self.observations.at[self.position].set(obs)
        self.actions = self.actions.at[self.position].set(action)
        self.rewards = self.rewards.at[self.position].set(reward)
        self.next_observations = self.next_observations.at[self.position].set(next_obs)
        self.dones = self.dones.at[self.position].set(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, key: jax.Array):
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            key: JAX random key

        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = jax.random.choice(
            key,
            self.size,
            shape=(batch_size,),
            replace=False
        )

        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def __len__(self):
        """Return current size of buffer."""
        return self.size


class QNetwork(eqx.Module):
    """Q-Network for discrete action DQN using Equinox MLP.

    Maps observations to Q-values for each discrete action.
    This network is shared between pursuer and evader.
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


@eqx.filter_jit
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


def train_dqn_shared(
    env: PursuerEvaderEnv,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    buffer_size: int = 20000,  # Larger buffer since we collect 2x data
    target_update_freq: int = 10,
    num_actions_per_dim: int = 5,
    hidden_size: int = 128,
    seed: int = 0
):
    """Train DQN with a shared network for both agents.

    Both pursuer and evader use the same Q-network. The network learns
    a symmetric strategy that works for both roles.

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

    # Create single shared Q-network
    q_network = QNetwork(obs_dim, num_actions, hidden_size, qnet_key)
    target_network = q_network  # Initialize target network

    # Create single optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(q_network, eqx.is_array))

    # Create single replay buffer (stores experiences from both agents)
    replay_buffer = ReplayBuffer(buffer_size, obs_dim, 1)

    # Training metrics
    episode_rewards_pursuer = []
    episode_rewards_evader = []
    episode_lengths = []
    losses = []
    captures = []
    timeouts = []

    epsilon = epsilon_start

    print("=" * 60)
    print("Training DQN with Shared Network")
    print("=" * 60)
    print(f"Observation dim: {obs_dim}")
    print(f"Number of discrete actions: {num_actions}")
    print(f"Episodes: {num_episodes}")
    print(f"Network: SHARED (both agents use same network)")
    print("=" * 60)

    for episode in tqdm(range(num_episodes)):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward_pursuer = 0.0
        episode_reward_evader = 0.0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Get observations for both agents (keep as JAX arrays)
            pursuer_obs = obs_dict["pursuer"]
            evader_obs = obs_dict["evader"]
            pursuer_obs_array = observation_to_array(pursuer_obs)
            evader_obs_array = observation_to_array(evader_obs)

            # Both agents use the SAME network to select actions
            key, pursuer_key, evader_key = jax.random.split(key, 3)

            pursuer_action_idx = int(select_action(
                q_network,  # Same network
                pursuer_obs_array,
                epsilon,
                num_actions,
                pursuer_key
            ))
            pursuer_force = discretize_action(pursuer_action_idx, num_actions_per_dim, env.params.max_force)

            evader_action_idx = int(select_action(
                q_network,  # Same network
                evader_obs_array,
                epsilon,
                num_actions,
                evader_key
            ))
            evader_force = discretize_action(evader_action_idx, num_actions_per_dim, env.params.max_force)

            # Step environment
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            next_state, next_obs_dict, rewards, done, info = env.step(state, actions)

            # Get next observations for BOTH agents (keep as JAX arrays)
            next_pursuer_obs_array = observation_to_array(next_obs_dict["pursuer"])
            next_evader_obs_array = observation_to_array(next_obs_dict["evader"])

            # Add pursuer's experience (no conversion needed - stays as JAX arrays)
            replay_buffer.add(
                pursuer_obs_array,
                pursuer_action_idx,
                rewards["pursuer"],
                next_pursuer_obs_array,
                done
            )

            # Add evader's experience (no conversion needed - stays as JAX arrays)
            replay_buffer.add(
                evader_obs_array,
                evader_action_idx,
                rewards["evader"],
                next_evader_obs_array,
                done
            )

            episode_reward_pursuer += rewards["pursuer"]
            episode_reward_evader += rewards["evader"]
            episode_length += 1

            # Update state
            state = next_state
            obs_dict = next_obs_dict

            # Train the shared network if enough samples
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
                captures.append(info["captured"])
                timeouts.append(info["timeout"])
                break

        episode_rewards_pursuer.append(episode_reward_pursuer)
        episode_rewards_evader.append(episode_reward_evader)
        episode_lengths.append(episode_length)

        # Update target network
        if episode % target_update_freq == 0:
            target_network = q_network

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Logging
        if episode % 10 == 0:
            avg_reward_pursuer = np.mean(episode_rewards_pursuer[-10:]) if episode_rewards_pursuer else 0
            avg_reward_evader = np.mean(episode_rewards_evader[-10:]) if episode_rewards_evader else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            avg_loss = np.mean(losses[-100:]) if losses else 0
            capture_rate = np.mean(captures[-10:]) * 100 if len(captures) >= 10 else 0

            print(f"Episode {episode:4d} | "
                  f"P_Rew: {avg_reward_pursuer:6.2f} | "
                  f"E_Rew: {avg_reward_evader:6.2f} | "
                  f"Len: {avg_length:5.1f} | "
                  f"Cap%: {capture_rate:4.1f} | "
                  f"Loss: {avg_loss:8.4f} | "
                  f"Eps: {epsilon:.3f}")

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Final capture rate: {np.mean(captures[-50:]) * 100:.1f}%")
    print(f"Final timeout rate: {np.mean(timeouts[-50:]) * 100:.1f}%")
    print("=" * 60)

    return (q_network,
            episode_rewards_pursuer, episode_rewards_evader,
            episode_lengths, losses)


def evaluate_shared(
    env: PursuerEvaderEnv,
    q_network: QNetwork,
    num_episodes: int = 10,
    num_actions_per_dim: int = 5,
    seed: int = 42
):
    """Evaluate the shared network in self-play.

    Args:
        env: Environment
        q_network: Trained shared Q-network
        num_episodes: Number of evaluation episodes
        num_actions_per_dim: Number of discrete actions per dimension
        seed: Random seed
    """
    key = jax.random.PRNGKey(seed)
    num_actions = num_actions_per_dim ** 2

    episode_rewards_pursuer = []
    episode_rewards_evader = []
    captures = []
    timeouts = []
    episode_lengths = []

    print("\n" + "=" * 60)
    print("Evaluating Shared Network")
    print("=" * 60)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward_pursuer = 0.0
        episode_reward_evader = 0.0

        for step in range(env.params.max_steps):
            # Both agents use the SAME network (greedy actions)
            pursuer_obs = observation_to_array(obs_dict["pursuer"])
            evader_obs = observation_to_array(obs_dict["evader"])

            pursuer_q_values = q_network(pursuer_obs)
            evader_q_values = q_network(evader_obs)

            pursuer_action_idx = int(jnp.argmax(pursuer_q_values))
            evader_action_idx = int(jnp.argmax(evader_q_values))

            pursuer_force = discretize_action(pursuer_action_idx, num_actions_per_dim, env.params.max_force)
            evader_force = discretize_action(evader_action_idx, num_actions_per_dim, env.params.max_force)

            # Step
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            state, obs_dict, rewards, done, info = env.step(state, actions)

            episode_reward_pursuer += rewards["pursuer"]
            episode_reward_evader += rewards["evader"]

            if done:
                captures.append(info["captured"])
                timeouts.append(info["timeout"])
                episode_lengths.append(step + 1)
                break

        episode_rewards_pursuer.append(episode_reward_pursuer)
        episode_rewards_evader.append(episode_reward_evader)

        result = "CAPTURED" if info["captured"] else "TIMEOUT"
        print(f"Episode {episode+1:2d}: P_Rew={episode_reward_pursuer:6.2f}, "
              f"E_Rew={episode_reward_evader:6.2f}, Len={step+1:3d}, {result}")

    print("=" * 60)
    print(f"Average Pursuer Reward: {np.mean(episode_rewards_pursuer):6.2f}")
    print(f"Average Evader Reward: {np.mean(episode_rewards_evader):6.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):5.1f}")
    print(f"Capture Rate: {np.mean(captures)*100:.1f}%")
    print(f"Timeout Rate: {np.mean(timeouts)*100:.1f}%")
    print("=" * 60)
    print("\nNote: With a shared network, both agents use the same policy.")
    print("This creates symmetric play where the network learns optimal")
    print("strategy for both pursuit and evasion roles.")
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

    # Train with shared network
    results = train_dqn_shared(
        env,
        num_episodes=1000,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=20000,  # 2x larger since we collect from both agents
        target_update_freq=10,
        num_actions_per_dim=5,
        hidden_size=128,
        seed=0
    )

    q_network = results[0]
    episode_rewards_pursuer, episode_rewards_evader = results[1], results[2]
    episode_lengths, losses = results[3], results[4]

    # Evaluate
    evaluate_shared(env, q_network, num_episodes=20, num_actions_per_dim=5)

    # Optionally visualize results
    try:
        from plot_results import visualize_selfplay_episode
        import matplotlib.pyplot as plt

        print("\nGenerating visualizations...")

        # Plot training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Helper for smoothing
        window = 10
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        # Pursuer rewards
        axes[0].plot(episode_rewards_pursuer, alpha=0.3, color='blue', label='Pursuer (Raw)')
        axes[0].plot(episode_rewards_evader, alpha=0.3, color='red', label='Evader (Raw)')
        if len(episode_rewards_pursuer) >= window:
            smoothed_p = moving_average(episode_rewards_pursuer, window)
            smoothed_e = moving_average(episode_rewards_evader, window)
            axes[0].plot(range(window-1, len(episode_rewards_pursuer)), smoothed_p,
                        color='blue', linewidth=2, label='Pursuer (Smoothed)')
            axes[0].plot(range(window-1, len(episode_rewards_evader)), smoothed_e,
                        color='red', linewidth=2, label='Evader (Smoothed)')
        axes[0].set_title('Agent Rewards (Shared Network)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Episode lengths
        axes[1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
        if len(episode_lengths) >= window:
            smoothed = moving_average(episode_lengths, window)
            axes[1].plot(range(window-1, len(episode_lengths)), smoothed,
                        color='green', linewidth=2, label='Smoothed')
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Loss
        if losses:
            axes[2].plot(losses, alpha=0.3, color='purple', label='Raw')
            if len(losses) >= window:
                smoothed = moving_average(losses, window)
                axes[2].plot(range(window-1, len(losses)), smoothed,
                            color='purple', linewidth=2, label='Smoothed')
        axes[2].set_title('Training Loss (Shared Network)')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('TD Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Visualize episode (both agents use same network)
        print("\nVisualizing episode with shared network...")
        visualize_selfplay_episode(env, q_network, q_network,
                                   num_actions_per_dim=5, seed=123)

    except ImportError:
        print("\nSkipping visualization (matplotlib not installed)")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
