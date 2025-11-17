"""Tabular Minimax-Q Learning for Pursuit-Evasion Game.

This implementation follows the paper "Pursuit and evasion game between UVAs
based on multi-agent reinforcement learning" which uses:
1. Discretized state space (tabular Q-table)
2. Minimax-Q learning for zero-sum games
3. Relative observations

The key insight is that tabular methods with state discretization can provide
guaranteed convergence in zero-sum games, unlike deep RL which can be unstable.
"""

import os
import pickle
import time
from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog
from omegaconf import OmegaConf

from peax import PursuerEvaderEnv, Observation


@dataclass
class MinimaxQConfig:
    """Configuration for Minimax-Q learning."""

    exp_name: str = "minimax_q_peax"
    """experiment name"""
    seed: int = 1
    """random seed"""

    # Training parameters
    total_episodes: int = 50000
    """total number of training episodes"""
    learning_rate: float = 0.1
    """learning rate (alpha)"""
    gamma: float = 0.9
    """discount factor"""
    start_epsilon: float = 1.0
    """initial exploration rate"""
    end_epsilon: float = 0.05
    """final exploration rate"""
    epsilon_decay_episodes: int = 25000
    """episodes over which to decay epsilon"""

    # Evaluation
    eval_every: int = 1000
    """evaluate every N episodes"""
    eval_episodes: int = 20
    """number of episodes for evaluation"""

    # Environment
    boundary_type: str = "square"
    """boundary type"""
    boundary_size: float = 10.0
    """boundary size"""
    max_steps: int = 200
    """max steps per episode"""
    capture_radius: float = 0.5
    """capture radius"""
    max_force: float = 5.0
    """maximum force"""
    wall_penalty_coef: float = 0.01
    """wall proximity penalty"""
    velocity_reward_coef: float = 0.005
    """velocity reward coefficient"""

    # Action space
    num_actions_per_dim: int = 3
    """number of discrete actions per dimension"""

    # State discretization bins
    distance_bins: int = 6
    """number of bins for relative distance"""
    angle_bins: int = 8
    """number of bins for angles"""
    velocity_bins: int = 5
    """number of bins for velocities"""


# Register config
cs = ConfigStore.instance()
cs.store(name="config", node=MinimaxQConfig)


def discretize_state(obs: Observation, cfg: MinimaxQConfig) -> int:
    """Discretize continuous observation into discrete state index.

    Following the paper's approach, we use relative coordinates:
    - Relative distance (magnitude of relative position)
    - Relative angle (direction to opponent)
    - Relative velocity magnitude
    - Own velocity magnitude

    Args:
        obs: Observation with relative coordinates
        cfg: Configuration

    Returns:
        Discrete state index
    """
    # Relative distance (0 to ~14 for 10x10 boundary)
    rel_dist = np.linalg.norm(obs.relative_position)
    max_dist = cfg.boundary_size * np.sqrt(2)

    # Discretize distance into bins: [0,1), [1,2), [2,4), [4,6), [6,inf)
    if rel_dist < 1.0:
        dist_bin = 0
    elif rel_dist < 2.0:
        dist_bin = 1
    elif rel_dist < 4.0:
        dist_bin = 2
    elif rel_dist < 6.0:
        dist_bin = 3
    elif rel_dist < 10.0:
        dist_bin = 4
    else:
        dist_bin = 5

    # Relative angle (-π to π)
    rel_angle = np.arctan2(obs.relative_position[1], obs.relative_position[0])
    # Discretize into bins [0, 2π) with cfg.angle_bins bins
    angle_normalized = (rel_angle + np.pi) / (2 * np.pi)  # [0, 1)
    angle_bin = int(angle_normalized * cfg.angle_bins) % cfg.angle_bins

    # Relative velocity magnitude
    rel_vel_mag = np.linalg.norm(obs.relative_velocity)
    max_rel_vel = 20.0  # Approximate max relative velocity
    vel_normalized = np.clip(rel_vel_mag / max_rel_vel, 0, 1)
    rel_vel_bin = int(vel_normalized * (cfg.velocity_bins - 1))

    # Own velocity magnitude
    own_vel_mag = np.linalg.norm(obs.own_velocity)
    max_vel = 10.0  # Approximate max velocity
    own_vel_normalized = np.clip(own_vel_mag / max_vel, 0, 1)
    own_vel_bin = int(own_vel_normalized * (cfg.velocity_bins - 1))

    # Combine into single state index
    # Total states = distance_bins × angle_bins × rel_vel_bins × own_vel_bins
    state_idx = (
        dist_bin * (cfg.angle_bins * cfg.velocity_bins * cfg.velocity_bins) +
        angle_bin * (cfg.velocity_bins * cfg.velocity_bins) +
        rel_vel_bin * cfg.velocity_bins +
        own_vel_bin
    )

    return state_idx


def get_num_states(cfg: MinimaxQConfig) -> int:
    """Calculate total number of discrete states."""
    return cfg.distance_bins * cfg.angle_bins * cfg.velocity_bins * cfg.velocity_bins


def discretize_action(action_idx: int, num_actions_per_dim: int, max_force: float) -> np.ndarray:
    """Convert discrete action index to continuous force."""
    fx_idx = action_idx // num_actions_per_dim
    fy_idx = action_idx % num_actions_per_dim

    force_values = np.linspace(-max_force, max_force, num_actions_per_dim)
    fx = force_values[fx_idx]
    fy = force_values[fy_idx]

    return np.array([fx, fy])


def solve_minimax(q_values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve minimax optimization to find mixed strategy.

    For a zero-sum game, finds:
    max_π1 min_a2 Σ Q(s, a1, a2) π1(a1)

    This is equivalent to a linear program.

    Args:
        q_values: Q-values for current state [num_actions_pursuer, num_actions_evader]

    Returns:
        policy: Probability distribution over pursuer actions
        value: Minimax value
    """
    num_pursuer_actions, num_evader_actions = q_values.shape

    # Solve using linear programming
    # Variables: [π1(a1), ..., π1(an), V]
    # Maximize V subject to:
    #   Σ π1(a1) Q(a1, a2) >= V for all a2 (evader actions)
    #   Σ π1(a1) = 1
    #   π1(a1) >= 0

    # Convert to minimization: minimize -V
    c = np.zeros(num_pursuer_actions + 1)
    c[-1] = -1  # Maximize V = minimize -V

    # Inequality constraints: -Σ π1(a1) Q(a1, a2) + V <= 0
    A_ub = np.zeros((num_evader_actions, num_pursuer_actions + 1))
    for a2 in range(num_evader_actions):
        A_ub[a2, :-1] = -q_values[:, a2]  # -Q(a1, a2)
        A_ub[a2, -1] = 1  # +V
    b_ub = np.zeros(num_evader_actions)

    # Equality constraint: Σ π1(a1) = 1
    A_eq = np.zeros((1, num_pursuer_actions + 1))
    A_eq[0, :-1] = 1
    b_eq = np.array([1.0])

    # Bounds: π1(a1) >= 0, V unbounded
    bounds = [(0, None)] * num_pursuer_actions + [(None, None)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if result.success:
            policy = result.x[:-1]
            value = result.x[-1]
            # Normalize policy (numerical errors)
            policy = policy / policy.sum()
            return policy, value
        else:
            # Fallback: uniform distribution
            policy = np.ones(num_pursuer_actions) / num_pursuer_actions
            value = np.min(np.max(q_values, axis=0))
            return policy, value
    except:
        # Fallback on error
        policy = np.ones(num_pursuer_actions) / num_pursuer_actions
        value = np.min(np.max(q_values, axis=0))
        return policy, value


class MinimaxQLearning:
    """Tabular Minimax-Q Learning for zero-sum games."""

    def __init__(self, cfg: MinimaxQConfig):
        self.cfg = cfg
        self.num_states = get_num_states(cfg)
        self.num_actions = cfg.num_actions_per_dim ** 2

        # Q-table: Q(s, a_pursuer, a_evader)
        # Shape: [num_states, num_actions, num_actions]
        self.q_table = np.zeros((self.num_states, self.num_actions, self.num_actions))

        print(f"Initialized Minimax-Q Learning:")
        print(f"  State space size: {self.num_states}")
        print(f"  Action space size: {self.num_actions}")
        print(f"  Q-table shape: {self.q_table.shape}")
        print(f"  Q-table memory: {self.q_table.nbytes / 1024 / 1024:.2f} MB")

    def select_action(self, state_idx: int, epsilon: float, agent: str) -> int:
        """Select action using epsilon-greedy policy.

        For pursuer: select action with highest minimax value
        For evader: select action that minimizes pursuer's value
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit
            q_values = self.q_table[state_idx]  # [num_actions, num_actions]

            if agent == "pursuer":
                # Solve minimax to get policy
                policy, value = solve_minimax(q_values)
                # Sample from mixed strategy
                return np.random.choice(self.num_actions, p=policy)
            else:  # evader
                # Evader chooses action that minimizes pursuer's max value
                # For each evader action, compute max over pursuer actions
                pursuer_max_values = np.max(q_values, axis=0)  # [num_actions_evader]
                # Choose action that gives minimum of these max values
                return int(np.argmin(pursuer_max_values))

    def update(self, state: int, action_pursuer: int, action_evader: int,
               reward_pursuer: float, next_state: int, done: bool):
        """Minimax-Q update rule.

        Q(s, ap, ae) ← Q(s, ap, ae) + α[r + γ V(s') - Q(s, ap, ae)]
        where V(s') = max_π min_ae Σ Q(s', ap, ae) π(ap)
        """
        # Current Q-value
        q_current = self.q_table[state, action_pursuer, action_evader]

        # Compute minimax value of next state
        if done:
            minimax_value_next = 0.0
        else:
            q_next = self.q_table[next_state]
            _, minimax_value_next = solve_minimax(q_next)

        # TD target
        td_target = reward_pursuer + self.cfg.gamma * minimax_value_next

        # Update Q-value
        td_error = td_target - q_current
        self.q_table[state, action_pursuer, action_evader] += self.cfg.learning_rate * td_error

        return abs(td_error)

    def save(self, filepath: str):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'config': self.cfg
            }, f)
        print(f"Saved Q-table to {filepath}")

    def load(self, filepath: str):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
        print(f"Loaded Q-table from {filepath}")


@hydra.main(version_base=None, config_name="config")
def main(cfg: MinimaxQConfig) -> None:
    """Main training loop for Minimax-Q learning."""

    # Print configuration
    print("=" * 70)
    print("Minimax-Q Learning for Pursuit-Evasion")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # Set random seed
    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # Create environment
    env = PursuerEvaderEnv(
        boundary_type=cfg.boundary_type,
        boundary_size=cfg.boundary_size,
        max_steps=cfg.max_steps,
        capture_radius=cfg.capture_radius,
        max_force=cfg.max_force,
        wall_penalty_coef=cfg.wall_penalty_coef,
        velocity_reward_coef=cfg.velocity_reward_coef,
    )

    # Create Minimax-Q learner
    learner = MinimaxQLearning(cfg)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    td_errors = []

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    for episode in range(cfg.total_episodes):
        # Epsilon decay
        epsilon = cfg.start_epsilon - (cfg.start_epsilon - cfg.end_epsilon) * min(episode / cfg.epsilon_decay_episodes, 1.0)

        # Reset environment
        key, reset_key = jax.random.split(key)
        env_state, obs_dict = env.reset(reset_key)

        # Discretize initial state
        pursuer_state_idx = discretize_state(obs_dict["pursuer"], cfg)
        evader_state_idx = discretize_state(obs_dict["evader"], cfg)

        episode_reward = 0.0
        episode_length = 0
        episode_td_error = 0.0

        for step in range(cfg.max_steps):
            # Select actions
            pursuer_action_idx = learner.select_action(pursuer_state_idx, epsilon, "pursuer")
            evader_action_idx = learner.select_action(evader_state_idx, epsilon, "evader")

            # Convert to continuous forces
            pursuer_force = discretize_action(pursuer_action_idx, cfg.num_actions_per_dim, cfg.max_force)
            evader_force = discretize_action(evader_action_idx, cfg.num_actions_per_dim, cfg.max_force)

            # Step environment
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            next_env_state, next_obs_dict, rewards, done, info = env.step(env_state, actions)

            # Discretize next state
            next_pursuer_state_idx = discretize_state(next_obs_dict["pursuer"], cfg)
            next_evader_state_idx = discretize_state(next_obs_dict["evader"], cfg)

            # Update Q-table for both agents
            td_error_pursuer = learner.update(
                pursuer_state_idx, pursuer_action_idx, evader_action_idx,
                rewards["pursuer"], next_pursuer_state_idx, done
            )

            # For evader, reward is negative of pursuer (zero-sum)
            td_error_evader = learner.update(
                evader_state_idx, evader_action_idx, pursuer_action_idx,
                rewards["evader"], next_evader_state_idx, done
            )

            # Update state
            env_state = next_env_state
            pursuer_state_idx = next_pursuer_state_idx
            evader_state_idx = next_evader_state_idx

            episode_reward += rewards["pursuer"]
            episode_length += 1
            episode_td_error += (td_error_pursuer + td_error_evader) / 2

            if done:
                break

        # Log metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        td_errors.append(episode_td_error / episode_length if episode_length > 0 else 0)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_td_error = np.mean(td_errors[-100:])
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            print(f"Episode {episode+1:5d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Length: {avg_length:5.1f} | "
                  f"TD Error: {avg_td_error:6.4f} | "
                  f"ε: {epsilon:.3f} | "
                  f"Eps/sec: {eps_per_sec:.1f}")

        # Evaluation
        if (episode + 1) % cfg.eval_every == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at episode {episode+1}")
            print(f"{'='*70}")

            eval_rewards = []
            eval_captures = 0
            eval_timeouts = 0

            for eval_ep in range(cfg.eval_episodes):
                key, eval_key = jax.random.split(key)
                eval_state, eval_obs_dict = env.reset(eval_key)

                eval_reward = 0.0

                for eval_step in range(cfg.max_steps):
                    # Greedy actions (epsilon = 0)
                    pursuer_state_idx = discretize_state(eval_obs_dict["pursuer"], cfg)
                    evader_state_idx = discretize_state(eval_obs_dict["evader"], cfg)

                    pursuer_action_idx = learner.select_action(pursuer_state_idx, 0.0, "pursuer")
                    evader_action_idx = learner.select_action(evader_state_idx, 0.0, "evader")

                    pursuer_force = discretize_action(pursuer_action_idx, cfg.num_actions_per_dim, cfg.max_force)
                    evader_force = discretize_action(evader_action_idx, cfg.num_actions_per_dim, cfg.max_force)

                    eval_actions = {"pursuer": pursuer_force, "evader": evader_force}
                    eval_state, eval_obs_dict, eval_rewards_dict, eval_done, eval_info = env.step(eval_state, eval_actions)

                    eval_reward += eval_rewards_dict["pursuer"]

                    if eval_done:
                        if eval_info["captured"]:
                            eval_captures += 1
                        if eval_info["timeout"]:
                            eval_timeouts += 1
                        break

                eval_rewards.append(eval_reward)

            avg_eval_reward = np.mean(eval_rewards)
            capture_rate = eval_captures / cfg.eval_episodes
            timeout_rate = eval_timeouts / cfg.eval_episodes

            print(f"Average eval reward: {avg_eval_reward:.2f}")
            print(f"Capture rate: {capture_rate*100:.1f}%")
            print(f"Timeout rate: {timeout_rate*100:.1f}%")
            print(f"{'='*70}\n")

            # Save checkpoint
            learner.save(f"minimax_q_checkpoint_{episode+1}.pkl")

    print("=" * 70)
    print("Training completed!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Average reward (last 1000): {np.mean(episode_rewards[-1000:]):.2f}")
    print("=" * 70)

    # Save final Q-table
    learner.save("minimax_q_final.pkl")


if __name__ == "__main__":
    main()
