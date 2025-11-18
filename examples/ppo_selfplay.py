"""PPO with Self-Play for Pursuit-Evasion Game.

This implements Proximal Policy Optimization (PPO) adapted from CleanRL for the
PEAX environment with self-play. A single actor-critic network is trained to play
optimally from both pursuer and evader perspectives.

Key features:
- Continuous action space (force vectors)
- Self-play: both agents use same policy network
- GAE (Generalized Advantage Estimation)
- PPO clipped objective
- Faster and more stable than DQN

Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple

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
from PIL import Image

from peax import PursuerEvaderEnv, Observation


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    exp_name: str = "ppo_peax_selfplay"
    """experiment name"""
    seed: int = 1
    """random seed"""

    # PPO specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (for future vectorization)"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # Evaluation
    eval_every: int = 10000
    """evaluate every N timesteps"""
    eval_episodes: int = 20
    """number of episodes for evaluation"""

    # Environment specific arguments
    boundary_type: str = "square"
    """boundary type for the environment"""
    boundary_size: float = 10.0
    """size of the boundary"""
    max_steps: int = 200
    """maximum steps per episode"""
    capture_radius: float = 0.5
    """capture radius"""
    max_force: float = 5.0
    """maximum force"""
    wall_penalty_coef: float = 0.01
    """coefficient for wall proximity penalty"""
    velocity_reward_coef: float = 0.005
    """coefficient for velocity reward"""


# Register config
cs = ConfigStore.instance()
cs.store(name="config", node=PPOConfig)


def estimate_max_velocity(max_force: float, mass: float = 1.0, dt: float = 0.1, max_steps: int = 200) -> float:
    """Estimate maximum velocity an agent can reach."""
    acceleration = max_force / mass
    acceleration_duration = (max_steps / 10) * dt
    return acceleration * acceleration_duration


def observation_to_array(obs: Observation, boundary_size: float = 10.0, max_force: float = 5.0) -> np.ndarray:
    """Convert Observation to normalized numpy array.

    Returns 8D array: [rel_pos(2), rel_vel(2), own_vel(2), time(1), agent_id(1)]
    Agent ID is critical for self-play with shared policy!
    """
    max_velocity = estimate_max_velocity(max_force)
    max_distance = boundary_size * np.sqrt(2)

    return np.concatenate([
        np.array(obs.relative_position) / max_distance,
        np.array(obs.relative_velocity) / (2 * max_velocity),
        np.array(obs.own_velocity) / max_velocity,
        [obs.time_remaining],
        [obs.agent_id]  # CRITICAL: tells network if it's pursuer or evader!
    ])


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous action space.

    Actor outputs mean and log_std for a Gaussian policy.
    Critic outputs a single value estimate.
    """
    action_dim: int = 2

    @nn.compact
    def __call__(self, x):
        # Shared features
        features = nn.Dense(64)(x)
        features = nn.tanh(features)
        features = nn.Dense(64)(features)
        features = nn.tanh(features)

        # Actor head (policy)
        actor = nn.Dense(64)(features)
        actor = nn.tanh(actor)
        action_mean = nn.Dense(self.action_dim)(actor)
        # Learnable log_std (independent of state)
        action_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        # Critic head (value function)
        critic = nn.Dense(64)(features)
        critic = nn.tanh(critic)
        value = nn.Dense(1)(critic)

        return action_mean, action_logstd, value


class AgentState(flax.struct.PyTreeNode):
    """Training state for PPO agent."""
    params: flax.core.FrozenDict
    opt_state: optax.OptState
    step: int


def create_agent(key: jax.Array, obs_dim: int, action_dim: int, learning_rate: float) -> AgentState:
    """Create a new PPO agent."""
    network = ActorCritic(action_dim=action_dim)

    # Initialize network
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(key, dummy_obs)

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    return AgentState(params=params, opt_state=opt_state, step=0)


def get_action_and_value(
    network: ActorCritic,
    params: flax.core.FrozenDict,
    obs: jax.Array,
    key: jax.Array,
    action: jax.Array = None,
):
    """Get action from policy and compute log prob and value.

    If action is provided, compute log prob of that action.
    Otherwise, sample a new action.
    """
    action_mean, action_logstd, value = network.apply(params, obs)
    action_std = jnp.exp(action_logstd)

    if action is None:
        # Sample action
        action = action_mean + action_std * jax.random.normal(key, shape=action_mean.shape)

    # Compute log probability
    log_prob = -0.5 * (
        ((action - action_mean) / action_std) ** 2 +
        2 * action_logstd +
        jnp.log(2 * jnp.pi)
    )
    log_prob = log_prob.sum(axis=-1)

    return action, log_prob, value.squeeze(-1), action_mean


def ppo_loss(
    params: flax.core.FrozenDict,
    network: ActorCritic,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
):
    """Compute PPO loss."""
    # Get current policy outputs
    action_mean, action_logstd, values = network.apply(params, obs)
    action_std = jnp.exp(action_logstd)

    # Compute log probs of actions
    log_prob = -0.5 * (
        ((actions - action_mean) / action_std) ** 2 +
        2 * action_logstd +
        jnp.log(2 * jnp.pi)
    )
    log_prob = log_prob.sum(axis=-1)

    # Policy loss (clipped)
    ratio = jnp.exp(log_prob - old_log_probs)
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -advantages_normalized * ratio
    pg_loss2 = -advantages_normalized * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss (clipped)
    values = values.squeeze(-1)
    v_loss_unclipped = (values - returns) ** 2
    v_loss = v_loss_unclipped.mean()

    # Entropy bonus
    entropy = (action_logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(axis=-1).mean()

    # Total loss
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

    return loss, (pg_loss, v_loss, entropy, ratio.mean())


def render_episode_to_gif(
    env: PursuerEvaderEnv,
    network: ActorCritic,
    params: flax.core.FrozenDict,
    key: jax.Array,
    filename: str,
    max_steps: int = 200,
    num_episodes: int = 5
) -> Dict:
    """Render multiple episodes and save as GIF."""
    all_states = []
    total_reward = 0.0
    num_captured = 0
    num_timeout = 0
    total_length = 0

    # Collect states from multiple episodes
    for ep in range(num_episodes):
        key, reset_key, *action_keys = jax.random.split(key, max_steps + 2)
        env_state, obs_dict = env.reset(reset_key)
        all_states.append(env_state)

        episode_reward = 0.0

        for step in range(max_steps):
            # Greedy actions (use mean, no sampling)
            pursuer_obs = observation_to_array(obs_dict["pursuer"], env.params.boundary_size, env.params.max_force)
            evader_obs = observation_to_array(obs_dict["evader"], env.params.boundary_size, env.params.max_force)

            pursuer_obs_jax = jnp.array(pursuer_obs).reshape(1, -1)
            evader_obs_jax = jnp.array(evader_obs).reshape(1, -1)

            # Use action mean (deterministic policy for evaluation)
            action_mean_p, _, _, _ = network.apply(params, pursuer_obs_jax)
            action_mean_e, _, _, _ = network.apply(params, evader_obs_jax)

            pursuer_force = np.array(action_mean_p[0])
            evader_force = np.array(action_mean_e[0])

            # Clip to max_force
            pursuer_force = np.clip(pursuer_force, -env.params.max_force, env.params.max_force)
            evader_force = np.clip(evader_force, -env.params.max_force, env.params.max_force)

            # Step environment
            actions = {"pursuer": pursuer_force, "evader": evader_force}
            env_state, obs_dict, rewards, done, info = env.step(env_state, actions)

            all_states.append(env_state)
            episode_reward += rewards["pursuer"]

            if done:
                if info["captured"]:
                    num_captured += 1
                if info["timeout"]:
                    num_timeout += 1
                total_length += info["time"]
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

        state = all_states[frame]

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
        if np.linalg.norm(state.pursuer.velocity) > 0.1:
            ax.arrow(pursuer_pos[0], pursuer_pos[1],
                    state.pursuer.velocity[0] * 0.5, state.pursuer.velocity[1] * 0.5,
                    head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)

        # Evader (blue)
        ax.plot(evader_pos[0], evader_pos[1], 'bo', markersize=12, label='Evader')
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

        # Title
        time_remaining = env.params.max_steps - state.time
        ax.set_title(f'Step {state.time}/{env.params.max_steps} | Time Remaining: {time_remaining}')
        ax.legend(loc='upper right')

        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(all_states),
                        interval=100, blit=True, repeat=True)

    writer = PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)

    return {
        "avg_reward": total_reward / num_episodes,
        "capture_rate": num_captured / num_episodes,
        "timeout_rate": num_timeout / num_episodes,
        "avg_length": total_length / num_episodes if num_captured + num_timeout > 0 else max_steps
    }


@hydra.main(version_base=None, config_name="config")
def main(cfg: PPOConfig) -> None:
    """Main training loop for PPO."""

    print("=" * 70)
    print("PPO Self-Play for Pursuit-Evasion")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # Initialize tensorboard
    writer = SummaryWriter(f"runs/{cfg.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()]),
    )

    # Set random seeds
    random.seed(cfg.seed)
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

    obs_dim = 8  # [rel_pos(2), rel_vel(2), own_vel(2), time(1), agent_id(1)]
    action_dim = 2  # Force in x and y

    # Create network and agent
    key, network_key = jax.random.split(key)
    network = ActorCritic(action_dim=action_dim)

    # Learning rate schedule
    if cfg.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=cfg.learning_rate,
            end_value=0.0,
            transition_steps=cfg.total_timesteps,
        )
    else:
        learning_rate = cfg.learning_rate

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    # Initialize network
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(network_key, dummy_obs)
    opt_state = optimizer.init(params)

    # Training metrics
    global_step = 0
    num_episodes = 0
    start_time = time.time()

    # Batch size
    batch_size = cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches

    print(f"\nStarting training...")
    print(f"Batch size: {batch_size}")
    print(f"Minibatch size: {minibatch_size}")
    print(f"Updates per rollout: {cfg.update_epochs * cfg.num_minibatches}\n")

    # Storage for rollouts
    obs_buffer = np.zeros((batch_size, obs_dim))
    actions_buffer = np.zeros((batch_size, action_dim))
    log_probs_buffer = np.zeros(batch_size)
    rewards_buffer = np.zeros(batch_size)
    dones_buffer = np.zeros(batch_size)
    values_buffer = np.zeros(batch_size)

    # Initialize episode
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    # Track which agent we're currently controlling (alternate between them)
    current_agent = "pursuer"
    current_obs = observation_to_array(obs_dict[current_agent], env.params.boundary_size, env.params.max_force)

    episode_reward_pursuer = 0.0
    episode_reward_evader = 0.0
    episode_length = 0

    # Create jitted ppo_loss_fn
    @jax.jit
    def ppo_loss_fn(params, obs, actions, old_log_probs, advantages, returns):
        return ppo_loss(params, network, obs, actions, old_log_probs, advantages, returns,
                       cfg.clip_coef, cfg.vf_coef, cfg.ent_coef)

    # Training loop
    for update in range(cfg.total_timesteps // batch_size):
        # Collect rollout
        if update % 5 == 0:
            print(f"Collecting rollout {update}...")
        for step in range(batch_size):
            global_step += 1

            # Get action from policy
            key, action_key = jax.random.split(key)
            obs_jax = jnp.array(current_obs).reshape(1, -1)
            action, log_prob, value, _ = get_action_and_value(
                network, params, obs_jax, action_key
            )

            action_np = np.array(action[0])
            # Clip action to max_force
            action_np = np.clip(action_np, -cfg.max_force, cfg.max_force)

            # Store in buffer
            obs_buffer[step] = current_obs
            actions_buffer[step] = action_np
            log_probs_buffer[step] = float(log_prob[0])
            values_buffer[step] = float(value[0])

            # Get action for other agent (also from same policy)
            other_agent = "evader" if current_agent == "pursuer" else "pursuer"
            other_obs = observation_to_array(obs_dict[other_agent], env.params.boundary_size, env.params.max_force)
            other_obs_jax = jnp.array(other_obs).reshape(1, -1)

            key, other_action_key = jax.random.split(key)
            other_action, _, _, _ = get_action_and_value(
                network, params, other_obs_jax, other_action_key
            )
            other_action_np = np.array(other_action[0])
            other_action_np = np.clip(other_action_np, -cfg.max_force, cfg.max_force)

            # Step environment with both actions
            actions = {
                "pursuer": action_np if current_agent == "pursuer" else other_action_np,
                "evader": other_action_np if current_agent == "pursuer" else action_np
            }
            next_env_state, next_obs_dict, rewards_dict, done, info = env.step(env_state, actions)

            # Store reward for current agent
            reward = rewards_dict[current_agent]
            rewards_buffer[step] = reward
            dones_buffer[step] = float(done)

            # Update episode stats
            episode_reward_pursuer += rewards_dict["pursuer"]
            episode_reward_evader += rewards_dict["evader"]
            episode_length += 1

            # Update state
            env_state = next_env_state
            obs_dict = next_obs_dict

            # Switch agent perspective (alternate training between pursuer and evader)
            current_agent = "evader" if current_agent == "pursuer" else "pursuer"
            current_obs = observation_to_array(obs_dict[current_agent], env.params.boundary_size, env.params.max_force)

            if done:
                # Log episode metrics
                num_episodes += 1
                writer.add_scalar("charts/episodic_return_pursuer", episode_reward_pursuer, global_step)
                writer.add_scalar("charts/episodic_return_evader", episode_reward_evader, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                writer.add_scalar("charts/episode_number", num_episodes, global_step)

                # Reset episode
                key, reset_key = jax.random.split(key)
                env_state, obs_dict = env.reset(reset_key)
                current_agent = "pursuer"
                current_obs = observation_to_array(obs_dict[current_agent], env.params.boundary_size, env.params.max_force)
                episode_reward_pursuer = 0.0
                episode_reward_evader = 0.0
                episode_length = 0

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards_buffer)
        last_gae_lam = 0

        # Get value of next state
        next_obs_jax = jnp.array(current_obs).reshape(1, -1)
        _, _, next_value, _ = get_action_and_value(network, params, next_obs_jax, key)
        next_value = float(next_value[0]) * (1.0 - dones_buffer[-1])

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones_buffer[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones_buffer[t]
                next_val = values_buffer[t + 1]

            delta = rewards_buffer[t] + cfg.gamma * next_val * next_non_terminal - values_buffer[t]
            advantages[t] = last_gae_lam = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + values_buffer

        # Convert to JAX arrays
        obs_jax = jnp.array(obs_buffer)
        actions_jax = jnp.array(actions_buffer)
        log_probs_jax = jnp.array(log_probs_buffer)
        advantages_jax = jnp.array(advantages)
        returns_jax = jnp.array(returns)

        # PPO update
        for epoch in range(cfg.update_epochs):
            # Shuffle indices
            indices = np.arange(batch_size)
            np.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_obs = obs_jax[mb_indices]
                mb_actions = actions_jax[mb_indices]
                mb_log_probs = log_probs_jax[mb_indices]
                mb_advantages = advantages_jax[mb_indices]
                mb_returns = returns_jax[mb_indices]

                # Compute loss and gradients
                (loss, (pg_loss, v_loss, entropy, approx_kl)), grads = jax.value_and_grad(
                    ppo_loss_fn, has_aux=True
                )(
                    params, mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns
                )

                # Update parameters
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

        # Log training metrics
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", cfg.learning_rate if not cfg.anneal_lr else learning_rate(global_step), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_loss", float(pg_loss), global_step)
        writer.add_scalar("losses/value_loss", float(v_loss), global_step)
        writer.add_scalar("losses/entropy", float(entropy), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl), global_step)

        # Print progress
        if (update + 1) % 10 == 0:
            print(f"Update {update+1} | Step {global_step} | SPS: {sps} | "
                  f"Policy Loss: {pg_loss:.4f} | Value Loss: {v_loss:.4f} | "
                  f"Entropy: {entropy:.4f}")

        # Evaluation
        if global_step % cfg.eval_every == 0 and global_step > 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at step {global_step}")
            print(f"{'='*70}")

            eval_rewards_pursuer = []
            eval_rewards_evader = []
            eval_captures = 0
            eval_timeouts = 0

            for eval_ep in range(cfg.eval_episodes):
                key, eval_key = jax.random.split(key)
                eval_state, eval_obs_dict = env.reset(eval_key)

                eval_reward_p = 0.0
                eval_reward_e = 0.0

                for eval_step in range(cfg.max_steps):
                    # Greedy actions
                    pursuer_obs = observation_to_array(eval_obs_dict["pursuer"], env.params.boundary_size, env.params.max_force)
                    evader_obs = observation_to_array(eval_obs_dict["evader"], env.params.boundary_size, env.params.max_force)

                    pursuer_obs_jax = jnp.array(pursuer_obs).reshape(1, -1)
                    evader_obs_jax = jnp.array(evader_obs).reshape(1, -1)

                    action_mean_p, _, _, _ = network.apply(params, pursuer_obs_jax)
                    action_mean_e, _, _, _ = network.apply(params, evader_obs_jax)

                    pursuer_force = np.array(action_mean_p[0])
                    evader_force = np.array(action_mean_e[0])

                    pursuer_force = np.clip(pursuer_force, -cfg.max_force, cfg.max_force)
                    evader_force = np.clip(evader_force, -cfg.max_force, cfg.max_force)

                    eval_actions = {"pursuer": pursuer_force, "evader": evader_force}
                    eval_state, eval_obs_dict, eval_rewards_dict, eval_done, eval_info = env.step(eval_state, eval_actions)

                    eval_reward_p += eval_rewards_dict["pursuer"]
                    eval_reward_e += eval_rewards_dict["evader"]

                    if eval_done:
                        if eval_info["captured"]:
                            eval_captures += 1
                        if eval_info["timeout"]:
                            eval_timeouts += 1
                        break

                eval_rewards_pursuer.append(eval_reward_p)
                eval_rewards_evader.append(eval_reward_e)

            avg_eval_reward_p = np.mean(eval_rewards_pursuer)
            avg_eval_reward_e = np.mean(eval_rewards_evader)
            capture_rate = eval_captures / cfg.eval_episodes
            timeout_rate = eval_timeouts / cfg.eval_episodes

            print(f"Average eval reward (pursuer): {avg_eval_reward_p:.2f}")
            print(f"Average eval reward (evader): {avg_eval_reward_e:.2f}")
            print(f"Capture rate: {capture_rate*100:.1f}%")
            print(f"Timeout rate: {timeout_rate*100:.1f}%")

            # Log to tensorboard
            writer.add_scalar("eval/average_return_pursuer", avg_eval_reward_p, global_step)
            writer.add_scalar("eval/average_return_evader", avg_eval_reward_e, global_step)
            writer.add_scalar("eval/capture_rate", capture_rate, global_step)
            writer.add_scalar("eval/timeout_rate", timeout_rate, global_step)

            # Generate GIF
            print("Generating evaluation GIF...")
            gif_path = f"eval_ppo_step_{global_step}.gif"
            key, gif_key = jax.random.split(key)
            gif_info = render_episode_to_gif(
                env, network, params, gif_key, gif_path, cfg.max_steps, num_episodes=5
            )
            print(f"GIF saved to {gif_path}")
            print(f"  Avg reward: {gif_info['avg_reward']:.2f}")
            print(f"  Capture rate: {gif_info['capture_rate']*100:.1f}%")
            print(f"  Avg length: {gif_info['avg_length']:.1f}")

            # Log GIF to tensorboard
            try:
                gif_image = Image.open(gif_path)
                frames = []
                try:
                    while True:
                        frames.append(np.array(gif_image.convert('RGB')))
                        gif_image.seek(gif_image.tell() + 1)
                except EOFError:
                    pass
                if frames:
                    writer.add_image("eval/episode_visualization", frames[0], global_step, dataformats='HWC')
            except Exception as e:
                print(f"Warning: Could not log GIF to TensorBoard: {e}")

            print(f"{'='*70}\n")

    print("=" * 70)
    print("Training completed!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 70)

    # Save final model
    import pickle
    with open("ppo_final_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved final parameters to ppo_final_params.pkl")

    writer.close()


if __name__ == "__main__":
    main()
