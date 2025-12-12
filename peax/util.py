import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_disc_game(Y_1, Y_2, grey, title, plot_grey=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    if plot_grey:
        ax.scatter(grey[:,0], grey[:, 1], color="grey", alpha=0.6, s=20, label="Non active")

    # Plot the scatter points
    ax.scatter(Y_1[:,0], Y_1[:, 1], color="blue", alpha=0.6, s=200, label="Agent 1")
    ax.scatter(Y_2[:,0], Y_2[:, 1], color="red", alpha=0.6, s=200, label="Agent 2")
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
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # Now safe since limits are already square

    plt.tight_layout()
    #plt.show()
    return fig