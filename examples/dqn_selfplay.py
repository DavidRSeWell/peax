"""Deep Q-Network (DQN) with Self-Play for Pursuer-Evader.

This example demonstrates self-play where both the pursuer and evader are
controlled by their own Q-networks that train against each other. This creates
a competitive learning scenario where both agents improve over time.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import NamedTuple, Tuple, Dict
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
        Flat array of observation values
    """
    return jnp.concatenate([
        obs.own_position,
        obs.own_velocity,
        obs.other_position,
        obs.other_velocity,
        jnp.array([obs.time_remaining])
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


def train_dqn_selfplay(
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
    """Train DQN agents using self-play.

    Both pursuer and evader have their own Q-networks that train against each other.

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
    key, pursuer_key, evader_key = jax.random.split(key, 3)

    obs_dim = env.observation_space_dim
    num_actions = num_actions_per_dim ** 2  # Grid of actions in 2D

    # Create Q-networks for both agents
    pursuer_q_network = QNetwork(obs_dim, num_actions, hidden_size, pursuer_key)
    evader_q_network = QNetwork(obs_dim, num_actions, hidden_size, evader_key)

    # Create target networks
    pursuer_target_network = pursuer_q_network
    evader_target_network = evader_q_network

    # Create optimizers for both agents
    pursuer_optimizer = optax.adam(learning_rate)
    evader_optimizer = optax.adam(learning_rate)

    pursuer_opt_state = pursuer_optimizer.init(eqx.filter(pursuer_q_network, eqx.is_array))
    evader_opt_state = evader_optimizer.init(eqx.filter(evader_q_network, eqx.is_array))

    # Create replay buffers for both agents
    pursuer_replay_buffer = ReplayBuffer(buffer_size, obs_dim, 1)
    evader_replay_buffer = ReplayBuffer(buffer_size, obs_dim, 1)

    # Training metrics
    episode_rewards_pursuer = []
    episode_rewards_evader = []
    episode_lengths = []
    pursuer_losses = []
    evader_losses = []
    captures = []
    timeouts = []

    epsilon = epsilon_start

    print("=" * 60)
    print("Training DQN with Self-Play")
    print("=" * 60)
    print(f"Observation dim: {obs_dim}")
    print(f"Number of discrete actions: {num_actions}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward_pursuer = 0.0
        episode_reward_evader = 0.0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Get observations
            pursuer_obs = obs_dict["pursuer"]
            evader_obs = obs_dict["evader"]
            pursuer_obs_array = np.array(observation_to_array(pursuer_obs))
            evader_obs_array = np.array(observation_to_array(evader_obs))

            # Select actions for both agents using epsilon-greedy
            key, pursuer_key, evader_key = jax.random.split(key, 3)

            pursuer_action_idx = int(select_action(
                pursuer_q_network,
                jnp.array(pursuer_obs_array),
                epsilon,
                num_actions,
                pursuer_key
            ))
            pursuer_force = discretize_action(pursuer_action_idx, num_actions_per_dim, env.params.max_force)

            evader_action_idx = int(select_action(
                evader_q_network,
                jnp.array(evader_obs_array),
                epsilon,
                num_actions,
                evader_key
            ))
            evader_force = discretize_action(evader_action_idx, num_actions_per_dim, env.params.max_force)

            # Step environment
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            next_state, next_obs_dict, rewards, done, info = env.step(state, actions)

            # Store transitions for both agents
            next_pursuer_obs_array = np.array(observation_to_array(next_obs_dict["pursuer"]))
            next_evader_obs_array = np.array(observation_to_array(next_obs_dict["evader"]))

            pursuer_replay_buffer.add(
                pursuer_obs_array,
                pursuer_action_idx,
                rewards["pursuer"],
                next_pursuer_obs_array,
                done
            )

            evader_replay_buffer.add(
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

            # Train both agents if enough samples
            if len(pursuer_replay_buffer) >= batch_size:
                # Train pursuer
                key, sample_key = jax.random.split(key)
                batch = pursuer_replay_buffer.sample(batch_size, sample_key)
                pursuer_q_network, pursuer_opt_state, pursuer_loss = update_step(
                    pursuer_q_network,
                    pursuer_target_network,
                    pursuer_optimizer,
                    pursuer_opt_state,
                    *batch,
                    gamma
                )
                pursuer_losses.append(float(pursuer_loss))

            if len(evader_replay_buffer) >= batch_size:
                # Train evader
                key, sample_key = jax.random.split(key)
                batch = evader_replay_buffer.sample(batch_size, sample_key)
                evader_q_network, evader_opt_state, evader_loss = update_step(
                    evader_q_network,
                    evader_target_network,
                    evader_optimizer,
                    evader_opt_state,
                    *batch,
                    gamma
                )
                evader_losses.append(float(evader_loss))

            if done:
                captures.append(info["captured"])
                timeouts.append(info["timeout"])
                break

        episode_rewards_pursuer.append(episode_reward_pursuer)
        episode_rewards_evader.append(episode_reward_evader)
        episode_lengths.append(episode_length)

        # Update target networks
        if episode % target_update_freq == 0:
            pursuer_target_network = pursuer_q_network
            evader_target_network = evader_q_network

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Logging
        if episode % 10 == 0:
            avg_reward_pursuer = np.mean(episode_rewards_pursuer[-10:]) if episode_rewards_pursuer else 0
            avg_reward_evader = np.mean(episode_rewards_evader[-10:]) if episode_rewards_evader else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            avg_loss_pursuer = np.mean(pursuer_losses[-100:]) if pursuer_losses else 0
            avg_loss_evader = np.mean(evader_losses[-100:]) if evader_losses else 0
            capture_rate = np.mean(captures[-10:]) * 100 if len(captures) >= 10 else 0

            print(f"Episode {episode:4d} | "
                  f"P_Rew: {avg_reward_pursuer:6.2f} | "
                  f"E_Rew: {avg_reward_evader:6.2f} | "
                  f"Len: {avg_length:5.1f} | "
                  f"Cap%: {capture_rate:4.1f} | "
                  f"Eps: {epsilon:.3f}")

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Final capture rate: {np.mean(captures[-50:]) * 100:.1f}%")
    print(f"Final timeout rate: {np.mean(timeouts[-50:]) * 100:.1f}%")
    print("=" * 60)

    return (pursuer_q_network, evader_q_network,
            episode_rewards_pursuer, episode_rewards_evader,
            episode_lengths, pursuer_losses, evader_losses)


def evaluate_selfplay(
    env: PursuerEvaderEnv,
    pursuer_q_network: QNetwork,
    evader_q_network: QNetwork,
    num_episodes: int = 10,
    num_actions_per_dim: int = 5,
    seed: int = 42
):
    """Evaluate trained agents in self-play.

    Args:
        env: Environment
        pursuer_q_network: Trained pursuer Q-network
        evader_q_network: Trained evader Q-network
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
    print("Evaluating Self-Play Agents")
    print("=" * 60)

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, obs_dict = env.reset(reset_key)

        episode_reward_pursuer = 0.0
        episode_reward_evader = 0.0

        for step in range(env.params.max_steps):
            # Get observations (greedy actions, no exploration)
            pursuer_obs = observation_to_array(obs_dict["pursuer"])
            evader_obs = observation_to_array(obs_dict["evader"])

            pursuer_q_values = pursuer_q_network(pursuer_obs)
            evader_q_values = evader_q_network(evader_obs)

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


def main():
    """Main training script."""
    # Create environment
    env = PursuerEvaderEnv(
        boundary_type="square",
        boundary_size=10.0,
        max_steps=200,
        capture_radius=0.5,
    )

    # Train agents with self-play
    results = train_dqn_selfplay(
        env,
        num_episodes=1000,
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

    pursuer_q_network, evader_q_network = results[0], results[1]
    episode_rewards_pursuer, episode_rewards_evader = results[2], results[3]
    episode_lengths, pursuer_losses, evader_losses = results[4], results[5], results[6]

    # Evaluate agents
    evaluate_selfplay(env, pursuer_q_network, evader_q_network,
                      num_episodes=20, num_actions_per_dim=5)

    # Optionally visualize results
    try:
        from plot_results import visualize_selfplay_episode
        import matplotlib.pyplot as plt

        print("\nGenerating visualizations...")

        # Plot training curves for both agents
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Helper for smoothing
        window = 10
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        # Pursuer rewards
        axes[0, 0].plot(episode_rewards_pursuer, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards_pursuer) >= window:
            smoothed = moving_average(episode_rewards_pursuer, window)
            axes[0, 0].plot(range(window-1, len(episode_rewards_pursuer)), smoothed,
                           color='blue', linewidth=2, label='Smoothed')
        axes[0, 0].set_title('Pursuer Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Evader rewards
        axes[1, 0].plot(episode_rewards_evader, alpha=0.3, color='red', label='Raw')
        if len(episode_rewards_evader) >= window:
            smoothed = moving_average(episode_rewards_evader, window)
            axes[1, 0].plot(range(window-1, len(episode_rewards_evader)), smoothed,
                           color='red', linewidth=2, label='Smoothed')
        axes[1, 0].set_title('Evader Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
        if len(episode_lengths) >= window:
            smoothed = moving_average(episode_lengths, window)
            axes[0, 1].plot(range(window-1, len(episode_lengths)), smoothed,
                           color='green', linewidth=2, label='Smoothed')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Win rate (empty for now)
        axes[1, 1].axis('off')

        # Pursuer loss
        if pursuer_losses:
            axes[0, 2].plot(pursuer_losses, alpha=0.3, color='blue', label='Raw')
            if len(pursuer_losses) >= window:
                smoothed = moving_average(pursuer_losses, window)
                axes[0, 2].plot(range(window-1, len(pursuer_losses)), smoothed,
                               color='blue', linewidth=2, label='Smoothed')
        axes[0, 2].set_title('Pursuer Loss')
        axes[0, 2].set_xlabel('Update Step')
        axes[0, 2].set_ylabel('TD Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Evader loss
        if evader_losses:
            axes[1, 2].plot(evader_losses, alpha=0.3, color='red', label='Raw')
            if len(evader_losses) >= window:
                smoothed = moving_average(evader_losses, window)
                axes[1, 2].plot(range(window-1, len(evader_losses)), smoothed,
                               color='red', linewidth=2, label='Smoothed')
        axes[1, 2].set_title('Evader Loss')
        axes[1, 2].set_xlabel('Update Step')
        axes[1, 2].set_ylabel('TD Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Visualize a self-play episode
        visualize_selfplay_episode(env, pursuer_q_network, evader_q_network,
                                   num_actions_per_dim=5, seed=123)

    except ImportError:
        print("\nSkipping visualization (matplotlib not installed)")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
