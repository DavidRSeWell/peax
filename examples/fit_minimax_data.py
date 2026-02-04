import hydra
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm

from collect_minimax_data import load_checkpoint, run_episode
from hydra.core.config_store import ConfigStore
from minimax_dqn import get_players_obs_from_global, get_minimax_action, discretize_action

from peax.environment import PursuerEvaderEnv
from peax.fpta import LSTQD, load_buffer_data
from peax.basis import simple_pursuer_evader_basis

from minimax_dqn import MinimaxDQNConfig

cs = ConfigStore.instance()
cs.store(name="config", node=MinimaxDQNConfig)


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



@hydra.main(version_base=None, config_name="config")
def main(config):

    checkpoint_path = "/home/drs4568/peax/minimax_checkpoint_step_490000.pkl"

    
    seed = 0
    num_episodes = 1000
   
    key = jax.random.PRNGKey(seed)

    # Create environment
    env = PursuerEvaderEnv(
        boundary_type=config.boundary_type,
        boundary_size=config.boundary_size,
        max_steps=config.max_steps,
        capture_radius=config.capture_radius,
        wall_penalty_coef=config.wall_penalty_coef,
        velocity_reward_coef=config.velocity_reward_coef,
    ) 

    print(f"Loading checkpoint from {checkpoint_path}")
    q_network, params, config = load_checkpoint(checkpoint_path)
    print(f"Loaded agent trained for {config.total_timesteps} timesteps")

    obs_dim = env.observation_space_dim
    num_actions_per_dim = 5
    #num_actions = cfg.num_actions_per_dim ** 2
    act_dim = 2
    eval_every = 5
    eval_iters = 3

    basis = simple_pursuer_evader_basis(alpha=0.01, num_trait=obs_dim, num_actions=act_dim)
    possible_actions = [discretize_action(i, num_actions_per_dim, env.params.max_force) for i in range(num_actions_per_dim**2)]
    norm_possible_actions = [discretize_action(i, num_actions_per_dim, env.params.max_force) / env.params.max_force for i in range(num_actions_per_dim**2)]
    m = len(basis)

    lstq = LSTQD(basis, num_actions=num_actions_per_dim**2)
 

    A = jnp.zeros((m**2, m**2))
    C = jnp.zeros((m, m))
    b = jnp.zeros((m**2, 1))
    M = 0.0

    total_transitions = 0
    total_captures = 0
    total_timeouts = 0
    episode_rewards = []
    errors = [ ] 

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        key, episode_key = jax.random.split(key)

        # Run episode
        states, q_matrix_list, pursuer_actions, evader_actions, rewards, next_states, dones, info = run_episode(
            env, q_network, params, num_actions_per_dim, episode_key
        )

        if episode % eval_every == 0:
            # Compare FTPA predictions with DQN Q Values
            fro_error = 0.0 
            for _ in range(eval_iters):
                states, q_matrix_list, pursuer_actions, evader_actions, rewards, next_states, dones, info = run_episode(
                            env, q_network, params, num_actions_per_dim, episode_key
                        )
                for i in range(states.shape[0]):
                    global_state = states[i]
                    pursuer_obs, evader_obs = get_players_obs_from_global(global_state)
                    pursuer_obs = jnp.concatenate([pursuer_obs, norm_possible_actions[pursuer_actions[i]]])
                    evader_obs = jnp.concatenate([evader_obs, norm_possible_actions[evader_actions[i]]])
                    pursuer_obs = pursuer_obs[None, :]
                    evader_obs = evader_obs[None, :]
                    B_xy = lstq.get_players_B(pursuer_obs, evader_obs)
                    B_yx = lstq.get_players_B(evader_obs, pursuer_obs)
                    f_xy = lstq.get_f_xy(pursuer_obs, evader_obs, norm_possible_actions, norm_possible_actions, C)
                    fro_error += jnp.linalg.norm(f_xy - q_matrix_list[i], ord='fro')
            fro_error /= (states.shape[0] * eval_iters)
            errors.append(fro_error)
            print(f"Episode {episode}: Frobeneous Error in Q Matrix Prediction: {fro_error}")
                

        for i in tqdm(range(states.shape[0])):
            # Fit FPTA at each step
            global_state = states[i]
            global_next_state = next_states[i]
            pursuer_obs, evader_obs = get_players_obs_from_global(global_state)
            #pursuer_obs = jnp.concatenate([pursuer_obs, jnp.array([pursuer_actions[i]], dtype=jnp.float32)])
            pursuer_obs = jnp.concatenate([pursuer_obs, norm_possible_actions[pursuer_actions[i]]])
            evader_obs = jnp.concatenate([evader_obs, norm_possible_actions[evader_actions[i]]])

            pursuer_obs = pursuer_obs[None, :]
            evader_obs = evader_obs[None, :]

            pursuer_next_obs, evader_next_obs = get_players_obs_from_global(global_next_state)

            # Get actions for next states using minimax
            q_matrix_next = q_matrix_list[i + 1]
            
            pursuer_next_action = get_minimax_action(q_matrix_next, is_pursuer=True)
            #pursuer_next_action = discretize_action(pursuer_next_action, num_actions_per_dim, env.params.max_force)
            pursuer_next_action = norm_possible_actions[pursuer_next_action]

            evader_next_action = get_minimax_action(q_matrix_next, is_pursuer=False)
            #evader_next_action = discretize_action(evader_next_action, num_actions_per_dim, env.params.max_force)
            evader_next_action = norm_possible_actions[evader_next_action]

            pursuer_next_obs = jnp.concatenate([pursuer_next_obs, pursuer_next_action])
            evader_next_obs = jnp.concatenate([evader_next_obs, evader_next_action])

            pursuer_next_obs = pursuer_next_obs[None, :]
            evader_next_obs = evader_next_obs[None, :]

            B_xy = lstq.get_players_B(pursuer_obs, evader_obs)
            B_yx = lstq.get_players_B(evader_obs, pursuer_obs)

            B_next_xy = lstq.get_players_B(pursuer_next_obs, evader_next_obs)
            B_next_yx = lstq.get_players_B(evader_next_obs, pursuer_next_obs)

            A_ = B_xy @ (B_xy - 0.99*B_next_xy).T
            b_ = B_xy @ jnp.array([[rewards[i]]])
            A += A_
            b += b_

            A_ = B_yx @ (B_yx - 0.99*B_next_yx).T
            b_ = B_yx @ -jnp.array([[rewards[i]]])

            A += A_
            b += b_
            M += 1

            '''
            A_c = A / M
            b_c = b / M
            C = jnp.linalg.pinv(A_c) @ b_c  
            C = C.reshape((m, m))
            '''

        A /= M
        b /= M

        C = jnp.linalg.pinv(A) @ b  
        C = C.reshape((m, m))
        print("Norm of C:", jnp.linalg.norm(C, ord='fro'))

    # Saving FPTA
    jnp.save("fpta_minimax.npy", C)

    

    #Plotting error over time
    plt.figure()
    plt.plot(np.arange(len(errors)) * eval_every, errors)
    plt.xlabel("Episodes")
    plt.ylabel("Frobenious Norm Error")
    plt.title("FPTA Q Matrix Prediction Error Over Time")
    plt.grid(True)
    plt.savefig("fpta_q_matrix_error.png")
    plt.close()

if __name__ == "__main__":
    main()
