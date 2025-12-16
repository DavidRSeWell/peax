import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import random

from typing import Tuple, Dict, List
from tqdm import tqdm

from peax.fpta import LSTQD
from peax.basis import simple_pursuer_evader_basis
from peax.environment import PursuerEvaderEnv, discretize_action




def run_eval(
    env: PursuerEvaderEnv,
    fpta: LSTQD,
    C1: jax.Array,
    C2: jax.Array,
    possible_actions: List[jax.Array],
    num_episodes: int,
    key: jax.Array
) -> Tuple[Dict, Dict]:
    """
    Evaluate two lstqd agents
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

    
    p1_reward = {"p": [], "e": []}
    p2_reward = {"p": [], "e": []}
    for _ in range(num_episodes):
        for id_ in range(2): # Let both play as pursuer / evader the same number of times
            p_r = 0.0
            e_r = 0.0
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
                
                if id_ == 0:
                    pursuer_force, _ = fpta.act(pursuer_obs_normal[None,:], evader_obs_normal[None,:], possible_actions, possible_actions, C1)
                    _ , evader_force = fpta.act(pursuer_obs_normal[None,:], evader_obs_normal[None,:], possible_actions, possible_actions, C2)
                else:
                    pursuer_force, _ = fpta.act(pursuer_obs_normal[None,:], evader_obs_normal[None,:], possible_actions, possible_actions, C2)
                    _ , evader_force = fpta.act(pursuer_obs_normal[None,:], evader_obs_normal[None,:], possible_actions, possible_actions, C1)

                # Step environment
                actions_dict = {"pursuer": pursuer_force.flatten(), "evader": evader_force.flatten()}
                next_env_state, next_obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)
                
                p_r += rewards_dict["pursuer"].item()
                e_r += rewards_dict["pursuer"].item()

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
                       # Update state
                env_state = next_env_state

                if done:
                    break
            # Finished episodes
            if id_ == 0: # p1 is pursuer
                p1_reward["p"].append(p_r / (step + 1))
                p2_reward["e"].append(e_r / (step + 1))
            else:
                p2_reward["p"].append(p_r / (step + 1))
                p1_reward["e"].append(e_r / (step + 1))
    
    return p1_reward, p2_reward


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


@hydra.main(version_base=None, config_path="conf", config_name="lstq_config")
def main(cfg) -> None:
 
    """
    Run LSTQD End to End
    1: Define Basis
    2: For each iter. 
        a: Collect Data with self play
        b: Use data to fit new basis funcs
    """
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
    #C = jnp.array(np.random.normal(size=(lstqd.m, lstqd.m)))
    # Load first C matrix from disk
    C = jnp.array(np.load("/home/drs4568/peax/C_15_old.npy"))
    #C = jnp.array(np.load(cfg.init_C_path))[-1]
    C_old = [C]
    evals = []
    for iter_ in tqdm(range(cfg.num_iters)):
        D = []  # Dataset for this iteration
        # Collect data with self-play for some number of episodes
        for _ in tqdm(range(cfg.episodes_per_iter)):
            observations, actions, rewards, next_observations, dones, info = run_selfplay_episode(
                env, lstqd, C, possible_actions, key
            )
            D.append((observations, actions, rewards, next_observations, dones))
        
        # Train LSTQD with collected data
        C = lstqd.fit_D(D, C, p1_acts=possible_actions, p2_acts=possible_actions)
        C_old.append(C)

        if iter_ % cfg.eval_every == 0:
            reward_new, reward_old = run_eval(
                env, lstqd, C, C_old[-2], possible_actions, cfg.eval_episodes, key    
            )
            eval_ = (np.array(reward_new["p"]).mean() + np.array(reward_new["e"]).mean()) / 2.0
            evals.append(eval_)
            np.save("C_{iter}.npy".format(iter=iter_), np.array(C))
            print(f"Iter {_}: Reward: {eval_}")
    
    # Plot evals and save to disk
    plt.plot(evals)
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("LSTQD Evaluation Over Iterations")
    plt.savefig("lstqd_evaluation.png")
    
    np.save("C_matrices.npy", np.array(C_old))
    np.save("evals.npy", np.array(evals))


        
if __name__ == "__main__":
    main() 

