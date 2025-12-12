import hydra
import jax
import jax.numpy as jnp
import numpy as np
import random

from typing import Tuple, Dict, List
from tqdm import tqdm

from peax.fpta import LSTQD
from peax.basis import simple_pursuer_evader_basis
from peax.environment import PursuerEvaderEnv, discretize_action


def run_selfplay_episode(
    env: PursuerEvaderEnv,
    fpta: LSTQD,
    C: jax.Array,
    possible_actions: List[jax.Array],
    key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Run a single self-play episode and collect transitions.

    Args:
        env: PursuerEvaderEnv environment
        q_network: Q-network
        params: Network parameters
        num_actions_per_dim: Number of discrete actions per dimension
        key: JAX random key

    Returns:
        Tuple of (observations, actions, rewards, next_observations, dones, agent_ids, info)
        Arrays are collected from both agents' perspectives
    """
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    # Lists to collect data
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones_list = []
    agent_ids = []  # 0 = pursuer, 1 = evader

    for step in range(env.params.max_steps):

        # Get obs for FPTA trianing not relative
        pursuer_obs_normal = jnp.array([
            env_state.pursuer.position[0], env_state.pursuer.position[1],
            env_state.pursuer.velocity[0], env_state.pursuer.velocity[1],
            env_state.time / env.params.max_steps, 0.0
        ])

        evader_obs_normal = jnp.array([
            env_state.evader.position[0], env_state.evader.position[1],
            env_state.evader.velocity[0], env_state.evader.velocity[1],
            env_state.time / env.params.max_steps, 1.0
        ])

        pursuer_force, evader_force = fpta.act(pursuer_obs_normal[None,:], evader_obs_normal[None,:], possible_actions, possible_actions, C)
        # Step environment
        actions_dict = {"pursuer": pursuer_force.flatten(), "evader": evader_force.flatten()}
        next_env_state, next_obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)


        next_pursuer_obs_normal = jnp.array([
            next_env_state.pursuer.position[0], next_env_state.pursuer.position[1],
            next_env_state.pursuer.velocity[0], next_env_state.pursuer.velocity[1],
            next_env_state.time / env.params.max_steps, 0.0
        ])

        next_evader_obs_normal = jnp.array([
            next_env_state.evader.position[0], next_env_state.evader.position[1],
            next_env_state.evader.velocity[0], next_env_state.evader.velocity[1],
            next_env_state.time / env.params.max_steps, 1.0
        ])

        # Store pursuer transition
        observations.append(jnp.concatenate((pursuer_obs_normal, evader_obs_normal)))
        actions.append(jnp.concatenate((pursuer_force, evader_force)))
        rewards.append([rewards_dict["pursuer"], rewards_dict["evader"]])
        next_observations.append(jnp.concatenate((next_pursuer_obs_normal, next_evader_obs_normal)))
        dones_list.append(done)
               # Update state
        env_state = next_env_state
        obs_dict = next_obs_dict

        if done:
            break

    # Convert to JAX arrays
    observations = jnp.array(observations, dtype=jnp.float32)
    actions = jnp.array(actions, dtype=jnp.int32)
    rewards = jnp.array(rewards, dtype=jnp.float32)
    next_observations = jnp.array(next_observations, dtype=jnp.float32)
    dones_array = jnp.array(dones_list, dtype=jnp.float32)
    #agent_ids = jnp.array(agent_ids, dtype=jnp.int32)

    return observations, actions, rewards, next_observations, dones_array, info


@hydra.main(version_base=None, config_path="/home/drs4568/peax/examples/conf/", config_name="lstq_config")
def main(cfg) -> None:
 
    """
    Run LSTQD End to End
    1: Define Basis
    2: For each iter. 
        a: Collect Data with self play
        b: Use data to fit new basis funcs
    """
    alpha = 0.2
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
    #num_actions = cfg.num_actions_per_dim ** 2
    act_dim = 2

    #key = jax.random.PRNGKey(cfg.seed)

    possible_actions = [discretize_action(i, cfg.num_actions_per_dim, env.params.max_force) for i in range(cfg.num_actions_per_dim**2)]

    # Basis setup    
    basis_fn = simple_pursuer_evader_basis(
        alpha=0.02,
        num_trait=obs_dim,
        num_actions=act_dim,
    )

    # Test basis functions
    x = jnp.array([0.5] * (obs_dim + act_dim))
    eval_x = [basis(x) for basis in basis_fn]

    # LSTQD setup
    lstqd = LSTQD(basis=basis_fn, num_actions=len(possible_actions))
    C = jnp.array(np.random.normal(size=(lstqd.m, lstqd.m)))
    C_old = [C]
    for _ in tqdm(range(cfg.num_iters)):
        D = []  # Dataset for this iteration
        # Collect data with self-play for some number of episodes
        for _ in tqdm(range(cfg.episodes_per_iter)):
            observations, actions, rewards, next_observations, dones, info = run_selfplay_episode(
                env, lstqd, C, possible_actions, key
            )
            D.append((observations, actions, rewards, next_observations, dones))
        
        # Train LSTQD with collected data
        C = lstqd.fit_D(D, p1_acts=possible_actions, p2_acts=possible_actions)
        C_old.append(C)

        
if __name__ == "__main__":
    main() 

