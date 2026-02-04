"""
Search over candidate basis function sets to find the best fit for LSTQD-FPTA.

Evaluates polynomial, Chebyshev, Fourier, RBF, and mixed bases.
Primary metric: MSE of predicted vs actual rewards on held-out data.
Secondary metric: Bellman residual RMSE.
"""
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from tabulate import tabulate
from tqdm import tqdm

from peax.fpta import LSTQD, load_buffer_data


# ---------------------------------------------------------------------------
# Basis generation helpers
# ---------------------------------------------------------------------------

OBS_DIM = 6  # per-player observation dimension


def generate_polynomial_basis(obs_dim: int, max_degree: int, cross_terms: bool = False):
    """Uncoupled polynomial basis {1, x_i, x_i^2, ...} up to max_degree.

    If cross_terms=True and max_degree>=2, also add all x_i * x_j (i<j).
    """
    basis = [lambda x: 1.0]

    for d in range(1, max_degree + 1):
        for i in range(obs_dim):
            basis.append(lambda x, _i=i, _d=d: x[_i] ** _d)

    if cross_terms and max_degree >= 2:
        for i in range(obs_dim):
            for j in range(i + 1, obs_dim):
                basis.append(lambda x, _i=i, _j=j: x[_i] * x[_j])

    return basis


def generate_chebyshev_basis(obs_dim: int, max_degree: int):
    """Uncoupled Chebyshev polynomial basis T_0, T_1(x_i), ..., T_k(x_i).

    Uses the recurrence: T_0=1, T_1(x)=x, T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x).
    """
    def _cheb(n, val):
        """Evaluate Chebyshev polynomial T_n at val."""
        if n == 0:
            return 1.0
        elif n == 1:
            return val
        t_prev, t_curr = 1.0, val
        for _ in range(2, n + 1):
            t_prev, t_curr = t_curr, 2.0 * val * t_curr - t_prev
        return t_curr

    basis = [lambda x: 1.0]

    for d in range(1, max_degree + 1):
        for i in range(obs_dim):
            basis.append(lambda x, _i=i, _d=d: _cheb(_d, x[_i]))

    return basis


def generate_fourier_basis(obs_dim: int, max_freq: int):
    """Uncoupled Fourier basis: {1, cos(pi*k*x_i), sin(pi*k*x_i)} for k=1..max_freq."""
    basis = [lambda x: 1.0]

    for k in range(1, max_freq + 1):
        for i in range(obs_dim):
            basis.append(lambda x, _i=i, _k=k: jnp.cos(jnp.pi * _k * x[_i]))
            basis.append(lambda x, _i=i, _k=k: jnp.sin(jnp.pi * _k * x[_i]))

    return basis


def generate_fourier_basis_with_action_linear(obs_dim: int, max_freq: int, action_indices: list = None):
    """Fourier basis with linear terms for action dimensions.

    The standard fourier basis can't distinguish between action values at -1 and +1
    because cos(k*pi*(-1)) = cos(k*pi*1). Adding linear terms fixes this.

    Args:
        obs_dim: Total observation dimension (including actions)
        max_freq: Max frequency for fourier terms
        action_indices: Indices of action dimensions to add linear terms for.
                       Default is [obs_dim-2, obs_dim-1] (last 2 dims).
    """
    if action_indices is None:
        action_indices = [obs_dim - 2, obs_dim - 1]

    basis = [lambda x: 1.0]

    # Fourier terms for all dimensions
    for k in range(1, max_freq + 1):
        for i in range(obs_dim):
            basis.append(lambda x, _i=i, _k=k: jnp.cos(jnp.pi * _k * x[_i]))
            basis.append(lambda x, _i=i, _k=k: jnp.sin(jnp.pi * _k * x[_i]))

    # Linear terms for action dimensions (to break symmetry)
    for i in action_indices:
        basis.append(lambda x, _i=i: x[_i])

    return basis


def generate_rbf_basis(n_centers: int, centers: jnp.ndarray, sigma: float):
    """RBF basis: exp(-||x - c_j||^2 / (2*sigma^2)) for each center c_j."""
    basis = []
    for j in range(n_centers):
        c = centers[j]
        basis.append(lambda x, _c=c, _s=sigma: jnp.exp(-jnp.sum((x - _c) ** 2) / (2.0 * _s ** 2)))
    return basis


def generate_current_basis():
    """The original exponential basis from run_lstqd_fpta.py."""
    basis = [lambda x: 1.0]
    basis += [lambda x, idx=i: jnp.exp(0.1 * (1 - 2 * x[0]) * x[idx]) for i in range(OBS_DIM)]
    return basis


def generate_poly_distance_basis(obs_dim: int):
    """Physics-informed: {1, x_i, ||x||, rel_dist^2, vel_magnitude^2}."""
    basis = [lambda x: 1.0]
    # Linear terms
    for i in range(obs_dim):
        basis.append(lambda x, _i=i: x[_i])
    # Euclidean norm
    basis.append(lambda x: jnp.sqrt(jnp.sum(x ** 2) + 1e-8))
    # Relative distance squared (first 2 dims = relative position)
    basis.append(lambda x: x[0] ** 2 + x[1] ** 2)
    return basis


def generate_mixed_poly_fourier_basis(obs_dim: int):
    """Mixed: {1, x_i, x_i^2, cos(pi*x_i), sin(pi*x_i)}."""
    basis = [lambda x: 1.0]
    for i in range(obs_dim):
        basis.append(lambda x, _i=i: x[_i])
        basis.append(lambda x, _i=i: x[_i] ** 2)
        basis.append(lambda x, _i=i: jnp.cos(jnp.pi * x[_i]))
        basis.append(lambda x, _i=i: jnp.sin(jnp.pi * x[_i]))
    return basis


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_basis(basis, name, buffer, buffer_state, batch_size, num_samples,
                   gamma, train_seed=0, eval_seed=42):
    """Fit LSTQD with the given basis and evaluate on held-out data.

    Returns a dict with metrics.
    """
    m = len(basis)
    lstq = LSTQD(basis, num_actions=25, gamma=gamma)

    # Fit
    try:
        C, tds, C_raw = lstq.fit(
            buffer, buffer_state,
            batch_size=batch_size, num_samples=num_samples, seed=train_seed,
            verbose=False,
        )
    except Exception as e:
        return {"name": name, "m": m, "mse": float('inf'),
                "bellman_rmse": float('inf'), "r_squared": float('-inf'),
                "skew_ratio": float('nan'), "error": str(e)}

    C_arr = np.asarray(C)
    C_mat = jnp.array(C_arr)

    # Held-out evaluation
    eval_key = jax.random.PRNGKey(eval_seed)
    batch = buffer.sample(buffer_state, eval_key)
    exp = batch.experience

    obs = exp['observation']
    rewards = exp['reward']
    next_obs = exp['next_observation']
    dones = exp['done']

    p1_obs, p2_obs = lstq.get_p_obs(obs)
    p1_next, p2_next = lstq.get_p_obs(next_obs)

    B_x = lstq.basis_eval(p1_obs)
    B_y = lstq.basis_eval(p2_obs)
    B_nx = lstq.basis_eval(p1_next)
    B_ny = lstq.basis_eval(p2_next)

    # Q-value predictions
    pred = jnp.sum(B_x @ C_mat * B_y, axis=1)
    pred_next = jnp.sum(B_nx @ C_mat * B_ny, axis=1) * (1.0 - dones)

    # Bellman residual: Q(s) - (r + gamma * Q(s'))
    td_target = rewards + gamma * pred_next
    bellman_res = pred - td_target
    bellman_rmse = float(jnp.sqrt(jnp.mean(bellman_res ** 2)))

    # MSE and R² vs TD target (the correct comparison for Q-values)
    mse = float(jnp.mean(bellman_res ** 2))
    ss_res = float(jnp.sum(bellman_res ** 2))
    ss_tot = float(jnp.sum((td_target - jnp.mean(td_target)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Skew-symmetry of raw C
    C_raw_jnp = jnp.array(C_raw)
    sym_part = (C_raw_jnp + C_raw_jnp.T) / 2
    raw_norm = float(jnp.linalg.norm(C_raw_jnp))
    skew_ratio = float(jnp.linalg.norm(sym_part)) / raw_norm if raw_norm > 0 else 0.0

    return {
        "name": name,
        "m": m,
        "mse": mse,
        "bellman_rmse": bellman_rmse,
        "r_squared": r_sq,
        "skew_ratio": skew_ratio,
        "C": C_arr,
        "lstq": lstq,
        "C_raw": C_raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = "/home/drs4568/peax/examples/"
    batch_size = 10000
    num_samples = 10
    gamma = 0.95

    print("Loading data...")
    buffer, buffer_state, meta_data = load_buffer_data(data_dir, batch_size=batch_size)

    # Sample some data for RBF center selection
    center_key = jax.random.PRNGKey(99)
    center_batch = buffer.sample(buffer_state, center_key)
    center_obs = center_batch.experience['observation'][:, :OBS_DIM]  # p1 obs

    # Estimate sigma for RBFs from median pairwise distance (subsample for speed)
    sub = center_obs[:500]
    dists = jnp.sqrt(jnp.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1))
    sigma = float(jnp.median(dists[dists > 0]))
    print(f"RBF sigma (median pairwise dist): {sigma:.4f}")

    # Select RBF centers via k-means-like random sampling
    rbf_idx = jax.random.choice(jax.random.PRNGKey(7), center_obs.shape[0], shape=(20,), replace=False)
    rbf_centers = center_obs[rbf_idx]

    # ---- Build all candidate basis sets ----
    candidates = [
        ("current_exp",       generate_current_basis()),
        ("poly_1",            generate_polynomial_basis(OBS_DIM, 1)),
        ("poly_2",            generate_polynomial_basis(OBS_DIM, 2)),
        ("poly_3",            generate_polynomial_basis(OBS_DIM, 3)),
        ("poly_2_cross",      generate_polynomial_basis(OBS_DIM, 2, cross_terms=True)),
        ("cheb_2",            generate_chebyshev_basis(OBS_DIM, 2)),
        ("cheb_3",            generate_chebyshev_basis(OBS_DIM, 3)),
        ("cheb_4",            generate_chebyshev_basis(OBS_DIM, 4)),
        ("fourier_1",         generate_fourier_basis(OBS_DIM, 1)),
        ("fourier_2",         generate_fourier_basis(OBS_DIM, 2)),
        ("fourier_3",         generate_fourier_basis(OBS_DIM, 3)),
        ("rbf_5",             generate_rbf_basis(5, rbf_centers[:5], sigma)),
        ("rbf_10",            generate_rbf_basis(10, rbf_centers[:10], sigma)),
        ("rbf_20",            generate_rbf_basis(20, rbf_centers[:20], sigma)),
        ("poly_dist",         generate_poly_distance_basis(OBS_DIM)),
        ("mixed_poly_four",   generate_mixed_poly_fourier_basis(OBS_DIM)),
    ]

    # ---- Evaluate each ----
    results = []
    for name, basis in tqdm(candidates, desc="Evaluating bases"):
        print(f"\n>>> Evaluating: {name}  (m={len(basis)})")
        res = evaluate_basis(
            basis, name, buffer, buffer_state,
            batch_size=batch_size, num_samples=num_samples, gamma=gamma,
        )
        results.append(res)
        print(f"    MSE={res['mse']:.6f}  R²={res['r_squared']:.4f}  "
              f"Bellman RMSE={res['bellman_rmse']:.6f}  skew={res['skew_ratio']:.4f}")

    # ---- Results table sorted by MSE ----
    results.sort(key=lambda r: r['mse'])

    headers = ["Rank", "Name", "m", "m²", "MSE", "R²", "Bellman RMSE", "Skew Ratio"]
    rows = []
    for i, r in enumerate(results):
        rows.append([
            i + 1,
            r['name'],
            r['m'],
            r['m'] ** 2,
            f"{r['mse']:.6f}",
            f"{r['r_squared']:.4f}",
            f"{r['bellman_rmse']:.6f}",
            f"{r['skew_ratio']:.4f}",
        ])

    print("\n" + "=" * 90)
    print("BASIS SEARCH RESULTS (sorted by MSE)")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print("=" * 90)

    # ---- Bar chart of MSE ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    names = [r['name'] for r in results]
    mses = [r['mse'] for r in results]
    bellmans = [r['bellman_rmse'] for r in results]

    ax1.barh(names[::-1], mses[::-1])
    ax1.set_xlabel('MSE (Predicted vs Actual)')
    ax1.set_title('Basis Function Search — MSE')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.barh(names[::-1], bellmans[::-1])
    ax2.set_xlabel('Bellman Residual RMSE')
    ax2.set_title('Basis Function Search — Bellman RMSE')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig("basis_search_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Detailed output for the best basis ----
    best = results[0]
    print(f"\nBest basis: {best['name']}  (m={best['m']})")
    print(f"  MSE:          {best['mse']:.6f}")
    print(f"  R²:           {best['r_squared']:.4f}")
    print(f"  Bellman RMSE: {best['bellman_rmse']:.6f}")

    # Save best C
    if 'C' in best:
        np.save("best_C.npy", best['C'])
        print("  Saved best C to best_C.npy")

    print("\nDone. See basis_search_results.png for comparison chart.")


if __name__ == "__main__":
    main()
