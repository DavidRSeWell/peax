"""Hydra-configurable LSTQD-FPTA fitting script.

Allows specifying basis function family and parameters via YAML config or CLI overrides.

Usage:
    python examples/run_lstqd_hydra.py
    python examples/run_lstqd_hydra.py basis_type=chebyshev max_degree=4
    python examples/run_lstqd_hydra.py basis_type=polynomial max_degree=3 cross_terms=true
    python examples/run_lstqd_hydra.py basis_type=rbf n_centers=20
    python examples/run_lstqd_hydra.py --multirun basis_type=polynomial,chebyshev,fourier
"""
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from peax.fpta import LSTQD, load_buffer_data
from search_basis import (
    generate_polynomial_basis,
    generate_chebyshev_basis,
    generate_fourier_basis,
    generate_rbf_basis,
    generate_poly_distance_basis,
    generate_mixed_poly_fourier_basis,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LSTQDHydraConfig:
    """Configuration for Hydra-driven LSTQD-FPTA fitting."""

    # Experiment
    seed: int = 0
    data_dir: str = "/home/drs4568/peax/examples/"

    # LSTQD fitting
    batch_size: int = 10000
    num_samples: int = 10
    gamma: float = 0.95
    num_actions_per_dim: int = 5

    # Basis selection
    # Options: exponential, polynomial, chebyshev, fourier,
    #          rbf, poly_distance, mixed_poly_fourier
    basis_type: str = "exponential"
    obs_dim: int = 6

    # Polynomial / Chebyshev parameters
    max_degree: int = 2
    cross_terms: bool = False

    # Fourier parameters
    max_freq: int = 1

    # RBF parameters
    n_centers: int = 10
    rbf_sigma: float = 0.0       # 0.0 = auto-estimate from data
    rbf_center_seed: int = 7
    rbf_subsample: int = 500

    # Original exponential basis parameter
    exp_alpha: float = 0.1

    # Plotting
    save_plots: bool = True
    plot_disc_games: bool = True


cs = ConfigStore.instance()
cs.store(name="lstq_hydra_schema", node=LSTQDHydraConfig)


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def _select_rbf_centers(buffer, buffer_state, n_centers, sigma, center_seed,
                        subsample_size, obs_dim):
    """Sample RBF centers from the replay buffer and optionally auto-estimate sigma."""
    center_key = jax.random.PRNGKey(center_seed)
    center_batch = buffer.sample(buffer_state, center_key)
    center_obs = center_batch.experience['observation'][:, :obs_dim]

    if sigma <= 0.0:
        sub = center_obs[:subsample_size]
        dists = jnp.sqrt(
            jnp.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        )
        sigma = float(jnp.median(dists[dists > 0]))
        print(f"RBF sigma auto-estimated (median pairwise dist): {sigma:.4f}")

    idx = jax.random.choice(
        jax.random.PRNGKey(center_seed + 1),
        center_obs.shape[0],
        shape=(n_centers,),
        replace=False,
    )
    centers = center_obs[idx]
    return centers, sigma


def build_basis(cfg, buffer=None, buffer_state=None):
    """Construct a list of basis functions from the Hydra config.

    Args:
        cfg: The Hydra config object.
        buffer: Required only when cfg.basis_type == "rbf".
        buffer_state: Required only when cfg.basis_type == "rbf".

    Returns:
        A list of callable basis functions, each mapping x -> scalar.
    """
    t = cfg.basis_type

    if t == "exponential":
        basis = [lambda x: 1.0]
        basis += [
            lambda x, idx=i: jnp.exp(cfg.exp_alpha * (1 - 2 * x[0]) * x[idx])
            for i in range(cfg.obs_dim)
        ]
        return basis

    elif t == "polynomial":
        return generate_polynomial_basis(cfg.obs_dim, cfg.max_degree, cfg.cross_terms)

    elif t == "chebyshev":
        return generate_chebyshev_basis(cfg.obs_dim, cfg.max_degree)

    elif t == "fourier":
        return generate_fourier_basis(cfg.obs_dim, cfg.max_freq)

    elif t == "rbf":
        assert buffer is not None and buffer_state is not None, (
            "RBF basis requires buffer and buffer_state for center selection"
        )
        centers, sigma = _select_rbf_centers(
            buffer, buffer_state, cfg.n_centers, cfg.rbf_sigma,
            cfg.rbf_center_seed, cfg.rbf_subsample, cfg.obs_dim,
        )
        return generate_rbf_basis(cfg.n_centers, centers, sigma)

    elif t == "poly_distance":
        return generate_poly_distance_basis(cfg.obs_dim)

    elif t == "mixed_poly_fourier":
        return generate_mixed_poly_fourier_basis(cfg.obs_dim)

    else:
        raise ValueError(
            f"Unknown basis_type: {t!r}. "
            f"Options: exponential, polynomial, chebyshev, fourier, "
            f"rbf, poly_distance, mixed_poly_fourier"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_disc_game(Y_1, Y_2, c_1, c_2, grey, title, plot_grey=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    if plot_grey:
        ax.scatter(grey[:,0], grey[:, 1], color="grey", alpha=0.6, s=20, label="Non active")

    ax.scatter(Y_1[:,0], Y_1[:, 1], c=c_1, alpha=0.6, s=200, label="Agent 1")
    ax.scatter(Y_2[:,0], Y_2[:, 1], c=c_2, alpha=0.6, s=200, label="Agent 2")

    if plot_grey:
        all_x = jnp.concatenate([grey[:, 0], grey[:, 0]])
        all_y = jnp.concatenate([grey[:, 1], grey[:, 1]])
    else:
        all_x = jnp.concatenate([Y_1[:, 0], Y_2[:, 0]])
        all_y = jnp.concatenate([Y_1[:, 1], Y_2[:, 1]])

    x_min, x_max = float(jnp.min(all_x)), float(jnp.max(all_x))
    y_min, y_max = float(jnp.min(all_y)), float(jnp.max(all_y))
    x_min = max(x_min, -6)
    x_max = min(x_max, 6)
    y_min = max(y_min, -6)
    y_max = min(y_max, 6)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min

    max_range = max(x_range, y_range)
    padding = 0.15 * max_range
    half_range = max_range / 2 + padding

    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    grid_density = 15
    x_grid = jnp.linspace(x_center - half_range, x_center + half_range, grid_density)
    y_grid = jnp.linspace(y_center - half_range, y_center + half_range, grid_density)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    U = Y
    V = -X

    magnitude = jnp.sqrt(U**2 + V**2)
    magnitude = jnp.where(magnitude == 0, 1, magnitude)
    U_normalized = U / magnitude
    V_normalized = V / magnitude

    ax.quiver(X, Y, U_normalized, V_normalized,
              alpha=0.3, color='gray', scale=25, width=0.003,
              headwidth=3, headlength=4)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(lstq, C, C_raw, L, Q, buffer, buffer_state, batch_size, gamma=0.99):
    """Compute and display fit quality metrics on a held-out batch."""

    eval_key = jax.random.PRNGKey(42)
    batch = buffer.sample(buffer_state, eval_key)
    experience = batch.experience

    obs = experience['observation']
    rewards = experience['reward']
    next_obs = experience['next_observation']
    dones = experience['done']

    p1_obs, p2_obs = lstq.get_p_obs(obs)
    p1_next_obs, p2_next_obs = lstq.get_p_obs(next_obs)

    B_x = lstq.basis_eval(p1_obs)
    B_y = lstq.basis_eval(p2_obs)
    B_next_x = lstq.basis_eval(p1_next_obs)
    B_next_y = lstq.basis_eval(p2_next_obs)

    C_mat = jnp.array(C)

    # 1. Eigenvalue Spectrum
    eig_magnitudes = L.flatten()
    num_disc = len(eig_magnitudes) // 2
    disc_eigs = eig_magnitudes[::2][:num_disc]

    eig_sq = disc_eigs ** 2
    total_var = jnp.sum(eig_sq)
    var_explained = eig_sq / total_var if float(total_var) > 0 else eig_sq
    cum_var = jnp.cumsum(var_explained)

    print("\n" + "=" * 70)
    print("EIGENVALUE SPECTRUM")
    print("=" * 70)
    print(f"{'Disc Game':<12} {'Eigenvalue':<15} {'Var %':<15} {'Cumulative %':<15}")
    print("-" * 57)
    for k in range(num_disc):
        print(f"{k+1:<12} {float(disc_eigs[k]):<15.6f} "
              f"{float(var_explained[k]):<15.4%} {float(cum_var[k]):<15.4%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = range(1, num_disc + 1)
    ax1.bar(xs, np.array(disc_eigs))
    ax1.set_xlabel('Disc Game')
    ax1.set_ylabel('Eigenvalue Magnitude')
    ax1.set_title('Eigenvalue Spectrum')
    ax1.set_xticks(list(xs))

    ax2.plot(list(xs), np.array(cum_var), 'o-')
    ax2.set_xlabel('Number of Disc Games')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_xticks(list(xs))
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("eigenvalue_spectrum.png")
    plt.close()

    # 2. Q-value vs TD Target (full C)
    pred = jnp.sum(B_x @ C_mat * B_y, axis=1)
    pred_next = jnp.sum(B_next_x @ C_mat * B_next_y, axis=1) * (1.0 - dones)
    td_target = rewards + gamma * pred_next

    bellman_res = pred - td_target
    mse = float(jnp.mean(bellman_res ** 2))
    mae = float(jnp.mean(jnp.abs(bellman_res)))
    ss_res = float(jnp.sum(bellman_res ** 2))
    ss_tot = float(jnp.sum((td_target - jnp.mean(td_target)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print("\n" + "=" * 70)
    print("Q-VALUE FIT QUALITY (Q(s) vs r + gamma*Q(s'))")
    print("=" * 70)
    print(f"  Bellman MSE:  {mse:.6f}")
    print(f"  Bellman MAE:  {mae:.6f}")
    print(f"  R\u00b2:           {r_squared:.6f}")
    print(f"  Q(s) range:   [{float(jnp.min(pred)):.4f}, {float(jnp.max(pred)):.4f}]")
    print(f"  TD tgt range: [{float(jnp.min(td_target)):.4f}, {float(jnp.max(td_target)):.4f}]")
    print(f"  Reward range: [{float(jnp.min(rewards)):.4f}, {float(jnp.max(rewards)):.4f}]")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.array(td_target), np.array(pred), alpha=0.3, s=10)
    lo = min(float(jnp.min(td_target)), float(jnp.min(pred)))
    hi = max(float(jnp.max(td_target)), float(jnp.max(pred)))
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.8, label='y = x')
    ax.set_xlabel('TD Target (r + \u03b3Q(s\'))')
    ax.set_ylabel('Predicted Q(s)')
    ax.set_title(f'Q(s) vs TD Target  (R\u00b2 = {r_squared:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pred_vs_actual.png")
    plt.close()

    # 3. Per-k Disc Game Reconstruction Error
    Y_fn = jax.vmap(lambda x: lstq.Y(Q, L, x))
    Y_1 = Y_fn(p1_obs)
    Y_2 = Y_fn(p2_obs)

    errors = []
    for k in range(1, num_disc + 1):
        pred_k = jnp.zeros(len(rewards))
        for j in range(k):
            y1_j = Y_1[:, 2*j:2*(j+1)]
            y2_j = Y_2[:, 2*j:2*(j+1)]
            pred_k = pred_k + y1_j[:, 0] * y2_j[:, 1] - y1_j[:, 1] * y2_j[:, 0]
        err_k = float(jnp.mean((pred_k - td_target) ** 2))
        errors.append(err_k)

    print("\n" + "=" * 70)
    print("PER-K DISC GAME RECONSTRUCTION ERROR (MSE vs TD target)")
    print("=" * 70)
    for k, err in enumerate(errors):
        print(f"  k={k+1} disc games:  MSE = {err:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, num_disc + 1), errors, 'o-')
    ax.set_xlabel('Number of Disc Games')
    ax.set_ylabel('MSE vs TD Target')
    ax.set_title('Reconstruction Error vs Number of Disc Games')
    ax.set_xticks(range(1, num_disc + 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reconstruction_error.png")
    plt.close()

    # 4. Skew-Symmetry Check (before enforcement)
    C_raw_jnp = jnp.array(C_raw)
    sym_part = (C_raw_jnp + C_raw_jnp.T) / 2
    raw_norm = float(jnp.linalg.norm(C_raw_jnp))
    relative_asym = float(jnp.linalg.norm(sym_part)) / raw_norm if raw_norm > 0 else 0.0

    print("\n" + "=" * 70)
    print("SKEW-SYMMETRY CHECK (before enforcement)")
    print("=" * 70)
    print(f"  ||sym part|| / ||C_raw||:  {relative_asym:.6f}")
    print(f"  (0.0 = perfectly skew-symmetric)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="lstq_hydra_config")
def main(cfg: LSTQDHydraConfig) -> None:
    print("=" * 70)
    print("LSTQD-FPTA Configuration:")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    # 1. Load data
    print("Loading data...")
    buffer, buffer_state, meta_data = load_buffer_data(
        cfg.data_dir, batch_size=cfg.batch_size
    )

    # 2. Build basis
    basis = build_basis(cfg, buffer=buffer, buffer_state=buffer_state)
    m = len(basis)
    print(f"\nBasis type: {cfg.basis_type}")
    print(f"Basis size (m): {m}")
    print(f"C matrix size (m x m): {m} x {m} = {m**2}")

    # 3. Create LSTQD and fit
    num_actions = cfg.num_actions_per_dim ** 2
    lstq = LSTQD(basis, num_actions=num_actions, gamma=cfg.gamma)

    key = jax.random.PRNGKey(cfg.seed)
    batch = buffer.sample(buffer_state, key)
    experience = batch.experience
    obs = experience['observation']
    p1_obs, p2_obs = lstq.get_p_obs(obs)

    print(f"\nFitting LSTQD (batch_size={cfg.batch_size}, num_samples={cfg.num_samples})...")
    C, tds, C_raw = lstq.fit(
        buffer, buffer_state,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
    )

    # 4. Low-rank decomposition
    C = np.asarray(C)
    L, Q = lstq.get_low_rank(C)

    # 5. Compute metrics
    compute_metrics(lstq, C, C_raw, L, Q, buffer, buffer_state,
                    cfg.batch_size, gamma=cfg.gamma)

    # 6. Disc game plots
    if cfg.plot_disc_games:
        Y_ = jax.vmap(lambda x: lstq.Y(Q, L, x))
        Y_1 = Y_(p1_obs)
        Y_2 = Y_(p2_obs)
        Y_grey = jnp.vstack([Y_1, Y_2])
        np.save("Y_grey.npy", Y_grey)
        print(f"Y_1 shape: {Y_1.shape}")
        print(f"Y_2 shape: {Y_2.shape}")

        num_disc = Y_1.shape[1] // 2
        for i in range(num_disc):
            for j in range(p1_obs.shape[1]):
                c_1 = p1_obs[:, j]
                c_2 = p2_obs[:, j]
                fig = plot_disc_game(
                    Y_1[:, 2*i:2*(i+1)], Y_2[:, 2*i:2*(i+1)],
                    c_1, c_2, grey=[], plot_grey=False,
                    title=f"Disc Game {i+1} Eig = {L[i]} color = {j+1}",
                )
                fig.savefig(f"disc_game_{i}_{j}.png")

    # 7. Save outputs
    jnp.save("C.npy", C)

    plt.close('all')
    plt.figure(figsize=(8, 8))
    plt.title(f"Iter vs TD Error ({cfg.basis_type})")
    x_ = list(range(len(tds)))
    plt.scatter(x_, tds)
    plt.plot(x_, tds)
    plt.savefig("TD_error.png")
    plt.close()

    print(f"\nDone. Outputs saved to current directory.")


if __name__ == "__main__":
    main()
