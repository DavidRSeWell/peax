import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from peax.fpta import LSTQD, load_buffer_data
from search_basis import generate_fourier_basis_with_action_linear

basis = generate_fourier_basis_with_action_linear(obs_dim=8, max_freq=3, action_indices=[6, 7])
print(f"Fourier basis with action linear (obs_dim=8, max_freq=3): m={len(basis)}")

def plot_disc_game(Y_1, Y_2, c_1, c_2, grey, title, plot_grey=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    if plot_grey:
        ax.scatter(grey[:,0], grey[:, 1], color="grey", alpha=0.6, s=20, label="Non active")

    # Plot the scatter points
    #ax.scatter(Y_1[:,0], Y_1[:, 1], color="blue", alpha=0.6, s=200, label="Agent 1")
    ax.scatter(Y_1[:,0], Y_1[:, 1], c=c_1, alpha=0.6, s=200, label="Agent 1")
    ax.scatter(Y_2[:,0], Y_2[:, 1], c=c_2, alpha=0.6, s=200, label="Agent 2")
        # Create rotating clockwise vector field
        # Get bounds from the data
    
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

    # Calculate center and ranges
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Use the larger range and add padding to create square axis limits
    max_range = max(x_range, y_range)
    padding = 0.15 * max_range  # 15% padding
    half_range = max_range / 2 + padding

    # Set square axis limits centered on the data
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    # Create grid for vector field using the square limits
    grid_density = 15  # Number of vectors in each direction
    x_grid = jnp.linspace(x_center - half_range, x_center + half_range, grid_density)
    y_grid = jnp.linspace(y_center - half_range, y_center + half_range, grid_density)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    # Clockwise rotating vector field: for point (x, y), vector is (y, -x)
    # This creates a clockwise rotation around the origin
    U = Y  # x-component of vector
    V = -X  # y-component of vector

    # Normalize the vectors
    magnitude = jnp.sqrt(U**2 + V**2)
    # Avoid division by zero
    magnitude = jnp.where(magnitude == 0, 1, magnitude)
    U_normalized = U / magnitude
    V_normalized = V / magnitude

    # Plot the vector field
    ax.quiver(X, Y, U_normalized, V_normalized,
              alpha=0.3, color='gray', scale=25, width=0.003,
              headwidth=3, headlength=4)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    #ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # Now safe since limits are already square

    plt.tight_layout()
    #plt.show()
    return fig



def compute_metrics(lstq, C, C_raw, L, Q, buffer, buffer_state, batch_size, gamma=0.99):
    """Compute and display fit quality metrics on a held-out batch."""

    # Sample a fresh eval batch with a different seed
    eval_key = jax.random.PRNGKey(42)
    batch = buffer.sample(buffer_state, eval_key)
    experience = batch.experience

    obs = experience['observation']
    rewards = experience['reward']
    next_obs = experience['next_observation']
    dones = experience['done']

    p1_obs, p2_obs = lstq.get_p_obs(obs)
    p1_next_obs, p2_next_obs = lstq.get_p_obs(next_obs)

    # Basis evaluations
    B_x = lstq.basis_eval(p1_obs)
    B_y = lstq.basis_eval(p2_obs)
    B_next_x = lstq.basis_eval(p1_next_obs)
    B_next_y = lstq.basis_eval(p2_next_obs)

    C_mat = jnp.array(C)

    # ---- 1. Eigenvalue Spectrum ----
    eig_magnitudes = L.flatten()
    num_disc = len(eig_magnitudes) // 2
    # Eigenvalues come in pairs from 2x2 Schur blocks; take one per disc game
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

    # ---- 2. Q-value vs TD Target (full C) ----
    # LSTDQ fits Q-values, not immediate rewards. The correct comparison
    # is Q(s) vs the TD target r + gamma * Q(s').
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
    print(f"  R²:           {r_squared:.6f}")
    print(f"  Q(s) range:   [{float(jnp.min(pred)):.4f}, {float(jnp.max(pred)):.4f}]")
    print(f"  TD tgt range: [{float(jnp.min(td_target)):.4f}, {float(jnp.max(td_target)):.4f}]")
    print(f"  Reward range: [{float(jnp.min(rewards)):.4f}, {float(jnp.max(rewards)):.4f}]")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.array(td_target), np.array(pred), alpha=0.3, s=10)
    lo = min(float(jnp.min(td_target)), float(jnp.min(pred)))
    hi = max(float(jnp.max(td_target)), float(jnp.max(pred)))
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.8, label='y = x')
    ax.set_xlabel('TD Target (r + γQ(s\'))')
    ax.set_ylabel('Predicted Q(s)')
    ax.set_title(f'Q(s) vs TD Target  (R² = {r_squared:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pred_vs_actual.png")
    plt.close()

    # ---- 3. Per-k Disc Game Reconstruction Error ----
    Y_fn = jax.vmap(lambda x: lstq.Y(Q, L, x))
    Y_1 = Y_fn(p1_obs)
    Y_2 = Y_fn(p2_obs)

    errors = []
    for k in range(1, num_disc + 1):
        pred_k = jnp.zeros(len(rewards))
        for j in range(k):
            y1_j = Y_1[:, 2*j:2*(j+1)]
            y2_j = Y_2[:, 2*j:2*(j+1)]
            # Cross product: Y1_0 * Y2_1 - Y1_1 * Y2_0
            pred_k = pred_k + y1_j[:, 0] * y2_j[:, 1] - y1_j[:, 1] * y2_j[:, 0]
        err_k = float(jnp.mean((pred_k - td_target) ** 2))
        errors.append(err_k)

    print("\n" + "=" * 70)
    print("PER-K DISC GAME RECONSTRUCTION ERROR (MSE)")
    print("=" * 70)
    for k, err in enumerate(errors):
        print(f"  k={k+1} disc games:  MSE = {err:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, num_disc + 1), errors, 'o-')
    ax.set_xlabel('Number of Disc Games')
    ax.set_ylabel('MSE vs Actual Rewards')
    ax.set_title('Reconstruction Error vs Number of Disc Games')
    ax.set_xticks(range(1, num_disc + 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reconstruction_error.png")
    plt.close()

    # ---- 4. Skew-Symmetry Check (before enforcement) ----
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


def main():
    data_dir = "/home/drs4568/peax/examples/data/"

    seed = 0
    batch_size = 10000
    num_samples = 10

    # num_actions_per_dim must match what the DQN used during data collection
    num_actions_per_dim = 3

    buffer, buffer_state, meta_data = load_buffer_data(
        data_dir, batch_size=batch_size, include_actions=True,
        num_actions_per_dim=num_actions_per_dim)

    lstq = LSTQD(basis, num_actions=num_actions_per_dim**2)

    key = jax.random.PRNGKey(seed)
    batch = buffer.sample(buffer_state, key)

    experience = batch.experience

    # Optional: filter by agent
    obs = experience['observation']
    #p_id = experience["agent_id"]
    #o_id = 1 - p_id
    #p_id = jnp.expand_dims(p_id, 1)
    #o_id = jnp.expand_dims(o_id, 1)
    p1_obs, p2_obs = lstq.get_p_obs(obs)

    print("obs before concat23")
    print(p1_obs.shape)

    #p1_obs = jnp.concatenate((p_id, p1_obs), axis=1)
    #p2_obs = jnp.concatenate((o_id, p2_obs), axis=1)


    C, tds, C_raw = lstq.fit(buffer, buffer_state, batch_size=batch_size, num_samples=num_samples , seed=0)
    #C = np.load("C.npy")

    C = np.asarray(C)
    L, Q = lstq.get_low_rank(C)

    # Compute fit quality metrics
    compute_metrics(lstq, C, C_raw, L, Q, buffer, buffer_state, batch_size, gamma=lstq.gamma)

    Y_ = lambda x: lstq.Y(Q, L, x)

    Y_ = jax.vmap(Y_)
    Y_1 = Y_(p1_obs)
    Y_2 = Y_(p2_obs)
    Y_grey = jnp.vstack([Y_1, Y_2])
    np.save("Y_grey.npy", Y_grey)
    print(Y_1.shape)
    print(Y_2.shape)

    num_disc = Y_1.shape[1] // 2

    for i in range(num_disc):
        for j in range(p1_obs.shape[1]):
            c_1 = p1_obs[:, j]
            c_2 = p2_obs[:, j]
            fig = plot_disc_game(Y_1[:, 2*i:2*(i + 1)], Y_2[:, 2*i: 2*(i + 1)], c_1, c_2, grey=[], plot_grey=False, title=f"Disc Game {i + 1} Eig = {L[i]} color = {j + 1}")
            fig.savefig(f"disc_game_{i}_{j}.png")

    jnp.save("C.npy", C)

    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.title("Iter vs TD Error")
    x_ = list(range(len(tds)))
    plt.scatter(x_, tds)
    plt.plot(x_, tds)
    plt.savefig("TD_error.png")





if __name__ == "__main__":
    main()
