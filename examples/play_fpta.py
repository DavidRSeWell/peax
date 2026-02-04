"""Play a fitted FPTA model in PursuerEvaderEnv against itself or a DQN agent.

At each state, constructs the payoff matrix Q(s, a_p, a_e) for all action pairs
using the fitted bilinear model Q = b(x)^T C b(y), then solves the Nash
equilibrium (minimax for zero-sum) to select actions.

Game modes:
  fpta_vs_fpta                - FPTA self-play (both players from Nash)
  fpta_pursuer_vs_dqn_evader  - FPTA pursuer, DQN evader
  dqn_pursuer_vs_fpta_evader  - DQN pursuer, FPTA evader

Usage:
  python play_fpta.py --c_matrix C.npy --num_episodes 50
  python play_fpta.py --c_matrix C.npy --dqn_checkpoint checkpoint.pkl --mode all
"""

import argparse
import os
import sys
from typing import Tuple, Optional, List

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))

from peax import PursuerEvaderEnv
from peax.fpta import LSTQD, _abs_to_relative
from cleanrl_dqn import QNetwork, observation_to_array, discretize_action
from collect_selfplay_data import load_checkpoint as load_dqn_checkpoint
from search_basis import generate_fourier_basis_with_action_linear


# ---------------------------------------------------------------------------
# Action table
# ---------------------------------------------------------------------------

def build_action_table(num_actions_per_dim: int, max_force: float):
    """Build discrete action lookup tables.

    Returns:
        action_table_norm: (N^2, 2) normalized actions in [-1, 1]
        action_table_force: (N^2, 2) force vectors in [-max_force, max_force]
    """
    forces_1d = jnp.linspace(-1.0, 1.0, num_actions_per_dim)
    fx, fy = jnp.meshgrid(forces_1d, forces_1d, indexing='ij')
    action_table_norm = jnp.stack([fx.ravel(), fy.ravel()], axis=-1)
    action_table_force = action_table_norm * max_force
    return action_table_norm, action_table_force


# ---------------------------------------------------------------------------
# Nash equilibrium solvers
# ---------------------------------------------------------------------------

def solve_nash_lp(payoff_matrix: np.ndarray):
    """Solve zero-sum Nash equilibrium via linear programming.

    Args:
        payoff_matrix: (n_row, n_col) payoff for the row player (maximizer).

    Returns:
        row_strategy: (n_row,) mixed strategy for row player (pursuer)
        col_strategy: (n_col,) mixed strategy for column player (evader)
        game_value: value of the game
    """
    M = np.array(payoff_matrix, dtype=np.float64)
    n_row, n_col = M.shape

    # Row player (maximizer): maximize v s.t. M^T p >= v, sum p = 1, p >= 0
    # linprog minimizes, so minimize -v
    c = np.zeros(n_row + 1)
    c[-1] = -1.0

    # -M^T p + v <= 0
    A_ub = np.zeros((n_col, n_row + 1))
    A_ub[:, :n_row] = -M.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_col)

    A_eq = np.zeros((1, n_row + 1))
    A_eq[0, :n_row] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n_row + [(None, None)]

    res_row = scipy.optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method='highs')

    if not res_row.success:
        # Fallback to pure strategy
        row_idx, col_idx = solve_nash_pure(M)
        row_strategy = np.zeros(n_row)
        row_strategy[row_idx] = 1.0
        col_strategy = np.zeros(n_col)
        col_strategy[col_idx] = 1.0
        return row_strategy, col_strategy, float(M[row_idx, col_idx])

    row_strategy = res_row.x[:n_row]
    game_value = res_row.x[-1]

    # Column player (minimizer): minimize w s.t. M q <= w, sum q = 1, q >= 0
    c2 = np.zeros(n_col + 1)
    c2[-1] = 1.0

    A_ub2 = np.zeros((n_row, n_col + 1))
    A_ub2[:, :n_col] = M
    A_ub2[:, -1] = -1.0
    b_ub2 = np.zeros(n_row)

    A_eq2 = np.zeros((1, n_col + 1))
    A_eq2[0, :n_col] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0, None)] * n_col + [(None, None)]

    res_col = scipy.optimize.linprog(
        c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
        bounds=bounds2, method='highs')

    if not res_col.success:
        col_idx = int(np.argmin(np.max(M, axis=0)))
        col_strategy = np.zeros(n_col)
        col_strategy[col_idx] = 1.0
    else:
        col_strategy = res_col.x[:n_col]

    # Clean up tiny negatives from LP
    row_strategy = np.maximum(row_strategy, 0.0)
    row_strategy /= row_strategy.sum()
    col_strategy = np.maximum(col_strategy, 0.0)
    col_strategy /= col_strategy.sum()

    return row_strategy, col_strategy, game_value


def solve_nash_pure(payoff_matrix: np.ndarray):
    """Pure strategy minimax.

    Returns:
        (row_action_idx, col_action_idx)
    """
    M = np.array(payoff_matrix)
    row_action = int(np.argmax(np.min(M, axis=1)))
    col_action = int(np.argmin(np.max(M, axis=0)))
    return row_action, col_action


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------

def get_fpta_obs(env_state, env: PursuerEvaderEnv,
                 boundary_size: float = 10.0,
                 max_velocity: float = 12.0):
    """Build relative 6D observations per player from env_state.

    Pipeline: 12D absolute -> _abs_to_relative -> split at midpoint.
    Time/agent_id at indices 4,5 are dropped by _abs_to_relative.

    Returns:
        p1_obs: (1, 6) pursuer view [rel_pos, rel_vel, own_vel]
        p2_obs: (1, 6) evader view  [-rel_pos, -rel_vel, own_vel]
    """
    t_norm = env_state.time / env.params.max_steps

    abs_obs = jnp.array([[
        env_state.pursuer.position[0], env_state.pursuer.position[1],
        env_state.pursuer.velocity[0], env_state.pursuer.velocity[1],
        t_norm, 0.0,
        env_state.evader.position[0], env_state.evader.position[1],
        env_state.evader.velocity[0], env_state.evader.velocity[1],
        t_norm, 1.0,
    ]])

    rel_obs = _abs_to_relative(abs_obs, boundary_size, max_velocity)
    half = rel_obs.shape[1] // 2
    return rel_obs[:, :half], rel_obs[:, half:]


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def fpta_select_action(lstqd, C, p1_obs, p2_obs, action_table_norm,
                       nash_method, role):
    """Select action for *role* using FPTA payoff matrix + Nash.

    Args:
        lstqd: LSTQD model
        C: Fitted C matrix (m, m)
        p1_obs: (1, 6) pursuer observation
        p2_obs: (1, 6) evader observation
        action_table_norm: (N^2, 2) normalized action vectors
        nash_method: "lp" or "pure"
        role: "pursuer" (row player, maximizer) or "evader" (col player, minimizer)

    Returns:
        action_idx: selected action index
        info: dict with payoff_matrix, game_value, strategy
    """
    # get_f_xy returns (n_p2_acts, n_p1_acts) due to vmap ordering
    payoff_raw = lstqd.get_f_xy(
        p1_obs, p2_obs,
        action_table_norm, action_table_norm, C)
    # Transpose so M[i,j] = Q(p1_act_i, p2_act_j)
    payoff_matrix = np.array(payoff_raw.T)

    info = {"payoff_matrix": payoff_matrix}

    if nash_method == "lp":
        row_strat, col_strat, game_value = solve_nash_lp(payoff_matrix)
        info["game_value"] = game_value

        if role == "pursuer":
            action_idx = np.random.choice(len(row_strat), p=row_strat)
            info["strategy"] = row_strat
        else:
            action_idx = np.random.choice(len(col_strat), p=col_strat)
            info["strategy"] = col_strat
    else:
        row_idx, col_idx = solve_nash_pure(payoff_matrix)
        game_value = float(payoff_matrix[row_idx, col_idx])
        info["game_value"] = game_value
        info["strategy"] = None
        action_idx = row_idx if role == "pursuer" else col_idx

    return action_idx, info


def dqn_select_action(q_network, params, obs_dict, agent_role,
                       boundary_size, max_force, num_actions_per_dim):
    """Select greedy action using DQN.

    Returns:
        action_idx: discrete action index
        force: 2D force vector for the environment
    """
    obs = observation_to_array(
        obs_dict[agent_role],
        boundary_size=boundary_size,
        max_force=max_force)
    q_values = q_network.apply(params, obs)
    action_idx = int(jnp.argmax(q_values))
    force = discretize_action(action_idx, num_actions_per_dim, max_force)
    return action_idx, force


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, mode, lstqd, C, action_table_norm, action_table_force,
                nash_method, key, boundary_size=10.0, max_velocity=12.0,
                max_force=10.0, q_network=None, dqn_params=None,
                dqn_num_actions_per_dim=None, collect_states=False):
    """Run a single episode.

    Returns dict with pursuer_reward, evader_reward, captured, timeout,
    length, game_values, and optionally states (list of EnvState).
    """
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    pursuer_total = 0.0
    evader_total = 0.0
    game_values = []
    states = [env_state] if collect_states else None

    pursuer_is_fpta = mode in ("fpta_vs_fpta", "fpta_pursuer_vs_dqn_evader")
    evader_is_fpta = mode in ("fpta_vs_fpta", "dqn_pursuer_vs_fpta_evader")

    for step in range(env.params.max_steps):
        # Build FPTA observations if needed
        if pursuer_is_fpta or evader_is_fpta:
            p1_obs, p2_obs = get_fpta_obs(
                env_state, env, boundary_size, max_velocity)

        # Pursuer action
        if pursuer_is_fpta:
            p_idx, p_info = fpta_select_action(
                lstqd, C, p1_obs, p2_obs,
                action_table_norm, nash_method, "pursuer")
            pursuer_force = action_table_force[p_idx]
            game_values.append(p_info["game_value"])
        else:
            _, pursuer_force = dqn_select_action(
                q_network, dqn_params, obs_dict, "pursuer",
                boundary_size, max_force, dqn_num_actions_per_dim)

        # Evader action
        if evader_is_fpta:
            e_idx, e_info = fpta_select_action(
                lstqd, C, p1_obs, p2_obs,
                action_table_norm, nash_method, "evader")
            evader_force = action_table_force[e_idx]
        else:
            _, evader_force = dqn_select_action(
                q_network, dqn_params, obs_dict, "evader",
                boundary_size, max_force, dqn_num_actions_per_dim)

        # Step environment
        actions_dict = {
            "pursuer": jnp.array(pursuer_force),
            "evader": jnp.array(evader_force),
        }
        env_state, obs_dict, rewards, done, info = env.step(
            env_state, actions_dict)

        pursuer_total += float(rewards["pursuer"])
        evader_total += float(rewards["evader"])

        if collect_states:
            states.append(env_state)

        if done:
            break

    result = {
        "pursuer_reward": pursuer_total,
        "evader_reward": evader_total,
        "captured": bool(info["captured"]),
        "timeout": bool(info["timeout"]),
        "length": step + 1,
        "game_values": game_values,
    }
    if collect_states:
        result["states"] = states
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(num_episodes, env, mode, lstqd, C,
                   action_table_norm, action_table_force, nash_method, key,
                   boundary_size=10.0, max_velocity=12.0, max_force=10.0,
                   q_network=None, dqn_params=None,
                   dqn_num_actions_per_dim=None):
    """Run multiple episodes and aggregate statistics."""
    results = []
    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        result = run_episode(
            env, mode, lstqd, C,
            action_table_norm, action_table_force, nash_method, ep_key,
            boundary_size, max_velocity, max_force,
            q_network, dqn_params, dqn_num_actions_per_dim)
        results.append(result)

        outcome = "CAPTURED" if result["captured"] else "TIMEOUT"
        print(f"  Episode {ep+1:3d}/{num_episodes}: "
              f"Pursuer={result['pursuer_reward']:+7.3f}  "
              f"Length={result['length']:3d}  {outcome}")

    captures = sum(1 for r in results if r["captured"])
    timeouts = sum(1 for r in results if r["timeout"])

    return {
        "num_episodes": num_episodes,
        "captures": captures,
        "timeouts": timeouts,
        "capture_rate": captures / num_episodes,
        "timeout_rate": timeouts / num_episodes,
        "avg_pursuer_reward": float(np.mean(
            [r["pursuer_reward"] for r in results])),
        "avg_evader_reward": float(np.mean(
            [r["evader_reward"] for r in results])),
        "avg_length": float(np.mean([r["length"] for r in results])),
        "results": results,
    }


def render_episodes_to_gif(env, episodes_states, filename, mode,
                           episodes_info=None):
    """Render collected episode states to an animated GIF.

    Args:
        env: PursuerEvaderEnv
        episodes_states: list of lists of EnvState (one list per episode)
        filename: output .gif path
        mode: game mode label for title
        episodes_info: list of dicts with captured/timeout/length per episode
    """
    # Flatten all episodes into a single frame sequence with separators
    frames = []
    episode_boundaries = []  # (start_frame, ep_index) for title display
    for ep_idx, states in enumerate(episodes_states):
        episode_boundaries.append((len(frames), ep_idx))
        frames.extend(states)

    half_size = env.params.boundary_size / 2

    fig, ax = plt.subplots(figsize=(6, 6))

    def _find_episode(frame_idx):
        ep = 0
        for i, (start, _) in enumerate(episode_boundaries):
            if start <= frame_idx:
                ep = i
        return ep

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-half_size * 1.2, half_size * 1.2)
        ax.set_ylim(-half_size * 1.2, half_size * 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw boundary
        rect = patches.Rectangle(
            (-half_size, -half_size), half_size * 2, half_size * 2,
            linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        state = frames[frame_idx]

        # Pursuer (red)
        p_pos = state.pursuer.position
        ax.plot(float(p_pos[0]), float(p_pos[1]), 'ro', markersize=12,
                label='Pursuer')
        p_vel = state.pursuer.velocity
        if float(jnp.linalg.norm(p_vel)) > 0.1:
            ax.arrow(float(p_pos[0]), float(p_pos[1]),
                     float(p_vel[0]) * 0.5, float(p_vel[1]) * 0.5,
                     head_width=0.3, head_length=0.2,
                     fc='red', ec='red', alpha=0.6)

        # Evader (blue)
        e_pos = state.evader.position
        ax.plot(float(e_pos[0]), float(e_pos[1]), 'bo', markersize=12,
                label='Evader')
        e_vel = state.evader.velocity
        if float(jnp.linalg.norm(e_vel)) > 0.1:
            ax.arrow(float(e_pos[0]), float(e_pos[1]),
                     float(e_vel[0]) * 0.5, float(e_vel[1]) * 0.5,
                     head_width=0.3, head_length=0.2,
                     fc='blue', ec='blue', alpha=0.6)

        # Capture radius
        cap_circle = patches.Circle(
            (float(p_pos[0]), float(p_pos[1])),
            env.params.capture_radius,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.2)
        ax.add_patch(cap_circle)

        # Title
        ep = _find_episode(frame_idx)
        ep_start = episode_boundaries[ep][0]
        step_in_ep = frame_idx - ep_start
        ep_label = ""
        if episodes_info and ep < len(episodes_info):
            outcome = ("CAPTURED" if episodes_info[ep]["captured"]
                       else "TIMEOUT")
            ep_label = f" [{outcome}]"
        ax.set_title(f'{mode} | Episode {ep+1} Step {step_in_ep}{ep_label}')
        ax.legend(loc='upper right')
        return []

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=100, blit=True, repeat=True)
    writer = PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"  GIF saved: {filename} ({len(frames)} frames, "
          f"{len(episodes_states)} episodes)")


def print_summary(summary, mode):
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS -- {mode}")
    print("=" * 70)
    print(f"Episodes:           {summary['num_episodes']}")
    print(f"Captures:           {summary['captures']} "
          f"({summary['capture_rate']*100:.1f}%)")
    print(f"Timeouts:           {summary['timeouts']} "
          f"({summary['timeout_rate']*100:.1f}%)")
    print(f"Avg Pursuer Reward: {summary['avg_pursuer_reward']:+.4f}")
    print(f"Avg Evader Reward:  {summary['avg_evader_reward']:+.4f}")
    print(f"Avg Episode Length: {summary['avg_length']:.1f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play FPTA model in PursuerEvaderEnv")

    parser.add_argument("--c_matrix", type=str, required=True,
                        help="Path to fitted C matrix (.npy)")
    parser.add_argument("--mode", type=str, default="fpta_vs_fpta",
                        choices=["fpta_vs_fpta",
                                 "fpta_pursuer_vs_dqn_evader",
                                 "dqn_pursuer_vs_fpta_evader",
                                 "all"])
    parser.add_argument("--obs_dim", type=int, default=8,
                        help="Observation dim for fourier basis (default: 8)")
    parser.add_argument("--max_freq", type=int, default=3,
                        help="Max frequency for fourier basis (default: 3)")
    parser.add_argument("--nash_method", type=str, default="lp",
                        choices=["lp", "pure"])
    parser.add_argument("--fpta_num_actions_per_dim", type=int, default=3,
                        help="Actions per dim for FPTA (must match data, default: 3)")
    parser.add_argument("--dqn_checkpoint", type=str, default=None,
                        help="Path to DQN checkpoint (.pkl)")
    parser.add_argument("--boundary_size", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--capture_radius", type=float, default=0.5)
    parser.add_argument("--max_force", type=float, default=10.0)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif", type=str, default=None,
                        help="Save GIF of episodes to this path "
                             "(e.g. fpta_selfplay.gif)")
    parser.add_argument("--gif_episodes", type=int, default=3,
                        help="Number of episodes to render in GIF (default: 3)")

    args = parser.parse_args()

    # --- Load FPTA model ---
    print("Loading FPTA model...")
    C = jnp.array(np.load(args.c_matrix))
    basis = generate_fourier_basis_with_action_linear(
        obs_dim=args.obs_dim, max_freq=args.max_freq, action_indices=[6, 7])
    num_fpta_actions = args.fpta_num_actions_per_dim ** 2
    lstqd = LSTQD(basis, num_actions=num_fpta_actions)

    expected_m = lstqd.m
    if C.shape != (expected_m, expected_m):
        raise ValueError(
            f"C matrix shape {C.shape} does not match basis size "
            f"({expected_m}, {expected_m}). "
            f"Got m={expected_m} basis functions for "
            f"obs_dim={args.obs_dim}, max_freq={args.max_freq}.")

    print(f"  C shape: {C.shape}")
    print(f"  Basis: fourier (obs_dim={args.obs_dim}, max_freq={args.max_freq}, "
          f"m={expected_m})")
    print(f"  Actions: {args.fpta_num_actions_per_dim}x"
          f"{args.fpta_num_actions_per_dim} = {num_fpta_actions}")

    # --- Build action tables ---
    action_table_norm, action_table_force = build_action_table(
        args.fpta_num_actions_per_dim, args.max_force)

    # --- Load DQN if needed ---
    modes = (["fpta_vs_fpta", "fpta_pursuer_vs_dqn_evader",
              "dqn_pursuer_vs_fpta_evader"]
             if args.mode == "all" else [args.mode])

    needs_dqn = any(m != "fpta_vs_fpta" for m in modes)
    q_network = None
    dqn_params = None
    dqn_num_actions_per_dim = None

    if needs_dqn:
        if args.dqn_checkpoint is None:
            raise ValueError(
                "--dqn_checkpoint required for DQN modes")
        print(f"\nLoading DQN checkpoint: {args.dqn_checkpoint}")
        q_network, dqn_params, dqn_config = load_dqn_checkpoint(
            args.dqn_checkpoint)
        dqn_num_actions_per_dim = dqn_config.num_actions_per_dim
        print(f"  DQN actions: {dqn_num_actions_per_dim}x"
              f"{dqn_num_actions_per_dim} = "
              f"{dqn_num_actions_per_dim**2}")

    # --- Create environment ---
    env = PursuerEvaderEnv(
        boundary_type="square",
        boundary_size=args.boundary_size,
        max_steps=args.max_steps,
        capture_radius=args.capture_radius,
    )

    max_velocity = 12.0
    key = jax.random.PRNGKey(args.seed)

    # --- Run evaluations ---
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Running: {mode}")
        print(f"{'='*70}")

        key, eval_key = jax.random.split(key)
        summary = run_evaluation(
            args.num_episodes, env, mode, lstqd, C,
            action_table_norm, action_table_force,
            args.nash_method, eval_key,
            args.boundary_size, max_velocity, args.max_force,
            q_network, dqn_params, dqn_num_actions_per_dim)

        print_summary(summary, mode)

        # --- GIF rendering ---
        if args.gif is not None:
            n_gif = min(args.gif_episodes, args.num_episodes)
            print(f"\nRendering {n_gif} episodes to GIF...")

            gif_states = []
            gif_info = []
            key, gif_key = jax.random.split(key)
            for i in range(n_gif):
                gif_key, ep_key = jax.random.split(gif_key)
                result = run_episode(
                    env, mode, lstqd, C,
                    action_table_norm, action_table_force,
                    args.nash_method, ep_key,
                    args.boundary_size, max_velocity, args.max_force,
                    q_network, dqn_params, dqn_num_actions_per_dim,
                    collect_states=True)
                gif_states.append(result["states"])
                gif_info.append(result)

            # Build filename: insert mode if running "all"
            if len(modes) > 1:
                base, ext = os.path.splitext(args.gif)
                gif_path = f"{base}_{mode}{ext}"
            else:
                gif_path = args.gif

            render_episodes_to_gif(env, gif_states, gif_path, mode,
                                   gif_info)


if __name__ == "__main__":
    main()
