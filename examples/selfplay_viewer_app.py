"""Streamlit app to visualize self-play episodes from trained DQN agents.

This app loads a checkpoint and allows interactive step-through of episodes.
"""

import streamlit as st
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from cleanrl_dqn import QNetwork, DQNConfig, observation_to_array, discretize_action
from peax import PursuerEvaderEnv
from peax.fpta import LSTQD
from run_lstqd_fpta import basis, plot_disc_game


# Page config
st.set_page_config(
    page_title="Self-Play Episode Viewer",
    page_icon="🎮",
    layout="wide"
)


@st.cache_resource
def load_checkpoint(checkpoint_path: str) -> Tuple[QNetwork, Dict, DQNConfig]:
    """Load a trained agent checkpoint.

    Args:
        checkpoint_path: Path to checkpoint pickle file

    Returns:
        Tuple of (q_network, params, config)
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = DQNConfig(**checkpoint['config'])
    params = checkpoint['params']

    # Create Q-network
    num_actions = config.num_actions_per_dim ** 2
    q_network = QNetwork(action_dim=num_actions)

    return q_network, params, config


def select_greedy_action(
    q_network: QNetwork,
    params: Dict,
    obs: jnp.ndarray,
    num_actions_per_dim: int,
    max_force: float
) -> Tuple[int, jnp.ndarray]:
    """Select greedy action from Q-network."""
    q_values = q_network.apply(params, obs)
    action_idx = int(jnp.argmax(q_values))
    force = discretize_action(action_idx, num_actions_per_dim, max_force)
    return action_idx, force


@st.cache_data
def generate_episode(
    _q_network: QNetwork,
    _params: Dict,
    config: DQNConfig,
    seed: int
) -> List[Dict]:
    """Generate a self-play episode and store all states.

    Args:
        _q_network: Q-network (underscore to prevent streamlit from hashing)
        _params: Network parameters
        config: DQN configuration
        seed: Random seed

    Returns:
        List of state dictionaries for each step
    """
    # Create environment
    env = PursuerEvaderEnv(
        boundary_type=config.boundary_type,
        boundary_size=config.boundary_size,
        max_steps=config.max_steps,
        capture_radius=config.capture_radius,
    )

    # Initialize
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state, obs_dict = env.reset(reset_key)

    def get_p_obs(obs: np.array):
        """
        Break up obs by player
        """
        '''
        p1_mask = jnp.ones(obs.shape[1], dtype=jnp.int2)
        p1_mask = p1_mask.at[4:-1].set(0)

        p2_mask = jnp.ones(obs.shape[1], dtype=jnp.int2)
        p2_mask = p2_mask.at[:4].set(0)

        print(p1_mask)
        '''

        print("get p obs")
        print(obs.shape)
        print(type(obs))
        print(obs[[1,2]])

        return obs[[0, 1, 2, 3, -1]], obs[[4, 5, 6, 7, -1]]



    # Store episode data
    episode_data = []

    for step in range(env.params.max_steps):
        # Get current state info
        pursuer_obs = observation_to_array(obs_dict["pursuer"])
        evader_obs = observation_to_array(obs_dict["evader"])
       

        # Select actions (self-play - both use same network)
        pursuer_action_idx, pursuer_force = select_greedy_action(
            _q_network, _params, pursuer_obs, config.num_actions_per_dim, env.params.max_force
        )
        evader_action_idx, evader_force = select_greedy_action(
            _q_network, _params, evader_obs, config.num_actions_per_dim, env.params.max_force
        )
         #p1_obs, p2_obs = get_p_obs(pursuer_obs)
        p1_obs = np.array(np.concatenate((env_state.pursuer.position, env_state.pursuer.velocity, env_state.evader.position, env_state.evader.velocity, [int(env_state.time)])))
        p2_obs = np.array(np.concatenate((env_state.evader.position, env_state.evader.velocity, env_state.pursuer.position, env_state.pursuer.velocity, [int(env_state.time)])))

        # Store current state
        episode_data.append({
            'step': step,
            'env_state': env_state,
            'pursuer_pos': np.array(env_state.pursuer.position),
            'pursuer_vel': np.array(env_state.pursuer.velocity),
            'evader_pos': np.array(env_state.evader.position),
            'evader_vel': np.array(env_state.evader.velocity),
            'pursuer_action': pursuer_action_idx,
            'evader_action': evader_action_idx,
            'pursuer_force': np.array(pursuer_force),
            'evader_force': np.array(evader_force),
            'pursuer_trait': p1_obs, 
            'evader_trait': p2_obs, 
            'time': int(env_state.time),
        })

        # Step environment
        actions_dict = {"pursuer": pursuer_force, "evader": evader_force}
        env_state, obs_dict, rewards_dict, done, info = env.step(env_state, actions_dict)

        #p1_obs, p2_obs = get_p_obs(obs_dict["pursuer"])

        p1_obs = np.array(np.concatenate((env_state.pursuer.position, env_state.pursuer.velocity, [0], [int(env_state.time)], env_state.evader.position, env_state.evader.velocity, [1], [int(env_state.time)])))
        p2_obs = np.array(np.concatenate((env_state.evader.position, env_state.evader.velocity, [1], [int(env_state.time)], env_state.pursuer.position, env_state.pursuer.velocity, [0], [int(env_state.time)])))



        # Add reward and done info to the stored state
        episode_data[-1]['pursuer_reward'] = float(rewards_dict["pursuer"])
        episode_data[-1]['evader_reward'] = float(rewards_dict["evader"])
        episode_data[-1]['done'] = bool(done)
        episode_data[-1]['captured'] = info.get('captured', False)
        episode_data[-1]['timeout'] = info.get('timeout', False)

        if done:
            # Store final state
            episode_data.append({
                'step': step + 1,
                'env_state': env_state,
                'pursuer_pos': np.array(env_state.pursuer.position),
                'pursuer_vel': np.array(env_state.pursuer.velocity),
                'evader_pos': np.array(env_state.evader.position),
                'evader_vel': np.array(env_state.evader.velocity),
                'pursuer_action': None,
                'evader_action': None,
                'pursuer_force': None,
                'evader_force': None,
                'pursuer_trait': p1_obs, 
                'evader_trait': p2_obs, 
                'time': int(env_state.time),
                'pursuer_reward': 0.0,
                'evader_reward': 0.0,
                'done': True,
                'captured': info['captured'],
                'timeout': info['timeout'],
            })
            break

    # Store environment config
    episode_data[0]['env'] = env

    return episode_data


def plot_episode_state(state_data: Dict, boundary_size: float, capture_radius: float) -> Figure:
    """Create a matplotlib figure showing the current episode state.

    Args:
        state_data: Dictionary with state information
        boundary_size: Size of the boundary
        capture_radius: Capture radius

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Get boundary type from environment if available
    env = state_data.get('env')
    boundary_type = env.boundary_type if env else "square"

    # Set axis limits
    limit = boundary_size * 0.6
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw boundary
    if boundary_type == "square":
        size = boundary_size / 2
        rect = patches.Rectangle(
            (-size, -size), size * 2, size * 2,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
    elif boundary_type == "circle":
        circle = patches.Circle(
            (0, 0), boundary_size / 2,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(circle)

    # Draw pursuer (red)
    pursuer_pos = state_data['pursuer_pos']
    pursuer_vel = state_data['pursuer_vel']
    ax.plot(pursuer_pos[0], pursuer_pos[1], 'ro', markersize=15, label='Pursuer', zorder=5)

    # Draw pursuer velocity arrow
    if np.linalg.norm(pursuer_vel) > 0.1:
        ax.arrow(pursuer_pos[0], pursuer_pos[1],
                pursuer_vel[0] * 0.5, pursuer_vel[1] * 0.5,
                head_width=0.4, head_length=0.3, fc='red', ec='red', alpha=0.6, zorder=4)

    # Draw evader (blue)
    evader_pos = state_data['evader_pos']
    evader_vel = state_data['evader_vel']
    ax.plot(evader_pos[0], evader_pos[1], 'bo', markersize=15, label='Evader', zorder=5)

    # Draw evader velocity arrow
    if np.linalg.norm(evader_vel) > 0.1:
        ax.arrow(evader_pos[0], evader_pos[1],
                evader_vel[0] * 0.5, evader_vel[1] * 0.5,
                head_width=0.4, head_length=0.3, fc='blue', ec='blue', alpha=0.6, zorder=4)

    # Draw capture radius
    capture_circle = patches.Circle(
        pursuer_pos, capture_radius,
        linewidth=1, edgecolor='red', facecolor='red', alpha=0.2, zorder=3
    )
    ax.add_patch(capture_circle)

    # Draw force vectors if available
    if state_data['pursuer_force'] is not None:
        pursuer_force = state_data['pursuer_force']
        if np.linalg.norm(pursuer_force) > 0.1:
            ax.arrow(pursuer_pos[0], pursuer_pos[1],
                    pursuer_force[0] * 0.3, pursuer_force[1] * 0.3,
                    head_width=0.3, head_length=0.2, fc='darkred', ec='darkred',
                    alpha=0.4, linestyle='--', zorder=2)

    if state_data['evader_force'] is not None:
        evader_force = state_data['evader_force']
        if np.linalg.norm(evader_force) > 0.1:
            ax.arrow(evader_pos[0], evader_pos[1],
                    evader_force[0] * 0.3, evader_force[1] * 0.3,
                    head_width=0.3, head_length=0.2, fc='darkblue', ec='darkblue',
                    alpha=0.4, linestyle='--', zorder=2)

    # Title with step info
    max_steps = state_data.get('env').params.max_steps if state_data.get('env') else 200
    time_remaining = max_steps - state_data['time']
    ax.set_title(f"Step {state_data['step']}/{state_data['time']} | Time Remaining: {time_remaining}",
                 fontsize=14, fontweight='bold')

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def main():
    st.title("🎮 Self-Play Episode Viewer")
    st.markdown("Load a trained DQN agent and interactively step through self-play episodes.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Checkpoint selection
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoint_step_8000.pkl",
        help="Path to the trained agent checkpoint"
    )
    checkpoint_path = "/home/drs4568/peax/checkpoint_step_490000.pkl"

    # Load FPTA model
    C = jnp.load("C.npy")
    Y_grey = jnp.load("Y_grey.npy")
    
    model = LSTQD(basis)
    L, Q = model.get_low_rank(C)


        

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Please provide a valid checkpoint path. You can train an agent with:\n\n"
                "```bash\npython examples/cleanrl_dqn.py total_timesteps=50000\n```")
        return

    # Load checkpoint
    try:
        with st.spinner("Loading checkpoint..."):
            q_network, params, config = load_checkpoint(checkpoint_path)
        st.sidebar.success("✅ Checkpoint loaded!")

        # Display config info
        with st.sidebar.expander("Model Info"):
            st.write(f"**Boundary:** {config.boundary_type}")
            st.write(f"**Boundary Size:** {config.boundary_size}")
            st.write(f"**Max Steps:** {config.max_steps}")
            st.write(f"**Actions:** {config.num_actions_per_dim}×{config.num_actions_per_dim} = {config.num_actions_per_dim**2}")
            st.write(f"**Trained Steps:** {config.total_timesteps}")
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return


    # Episode generation controls
    st.sidebar.header("Episode Controls")

    seed = st.sidebar.number_input("Random Seed", value=0, min_value=0, step=1)

    if st.sidebar.button("🎲 Generate New Episode", type="primary"):
        # Clear cache for this specific episode
        if 'episode_data' in st.session_state:
            del st.session_state['episode_data']
        st.session_state['current_step'] = 0

    # Generate episode if not already generated
    if 'episode_data' not in st.session_state:
        with st.spinner("Generating episode..."):
            episode_data = generate_episode(q_network, params, config, seed)
            st.session_state['episode_data'] = episode_data
            st.session_state['current_step'] = 0

    episode_data = st.session_state['episode_data']

    # Episode statistics
    final_state = episode_data[-1]
    episode_length = len(episode_data) - 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Episode Length", f"{episode_length} steps")
    with col2:
        outcome = "🎯 Captured" if final_state['captured'] else "⏰ Timeout"
        st.metric("Outcome", outcome)
    with col3:
        total_pursuer_reward = sum(s['pursuer_reward'] for s in episode_data[:-1])
        st.metric("Pursuer Reward", f"{total_pursuer_reward:.2f}")
    with col4:
        total_evader_reward = sum(s['evader_reward'] for s in episode_data[:-1])
        st.metric("Evader Reward", f"{total_evader_reward:.2f}")

    st.divider()

    # Auto-play controls
    st.sidebar.header("Auto-Play")
    auto_play = st.sidebar.checkbox("Auto-play", value=False)
    if auto_play:
        play_speed = st.sidebar.slider("Speed (steps/sec)", min_value=1, max_value=20, value=5)
        import time
        if st.session_state.get('current_step', 0) < len(episode_data) - 1:
            time.sleep(1.0 / play_speed)
            st.session_state['current_step'] = st.session_state.get('current_step', 0) + 1
            st.rerun()
        else:
            st.sidebar.info("Episode finished!")

    # Initialize current_step if not set
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 0


    # Display current state
    current_step = st.session_state['current_step']
    current_state = episode_data[current_step]


    # Visualization and info columns
    col_viz, col_info = st.columns(2)

    with col_viz:
        print("Current state")
        print(current_state)
        fig = plot_episode_state(current_state, config.boundary_size, config.capture_radius)
        st.pyplot(fig, width=900)
        plt.close(fig)

    with col_info:
        p_trait = jnp.expand_dims(current_state["pursuer_trait"], 1)
        e_trait = jnp.expand_dims(current_state["evader_trait"], 1)
        
        Y_p = model.Y(Q, L, p_trait)
        Y_e = model.Y(Q, L, e_trait)

        #grey_p = jnp.vstack([jnp.expand_dims(episode_data[i]["pursuer_trait"], 1) for i in range(len(episode_data)) if i != current_step])
        #grey_e = jnp.vstack([jnp.expand_dims(episode_data[i]["evader_trait"], 1) for i in range(len(episode_data)) if i != current_step])
        #grey_total = jnp.vstack([grey_p, grey_e])
        #Y_grey = jax.vmap(lambda x: model.Y(Q, L, x))(grey_total)

        print(f"Y_p = {Y_p.shape}")
        #print(f"grey_total = {grey_total.shape}")
        print(f"Y_grey = {Y_grey.shape}")
        print(f"p_trait = {p_trait.shape}")

        fig = plot_disc_game(Y_p[None, :2], Y_e[None, :2], Y_grey, title="Adv Map")
        st.pyplot(fig)
        plt.close(fig)


        _='''
        st.subheader("State Information")

        st.markdown("### 🔴 Pursuer")
        st.write(f"**Position:** ({current_state['pursuer_pos'][0]:.2f}, {current_state['pursuer_pos'][1]:.2f})")
        st.write(f"**Velocity:** ({current_state['pursuer_vel'][0]:.2f}, {current_state['pursuer_vel'][1]:.2f})")
        if current_state['pursuer_action'] is not None:
            st.write(f"**Action:** {current_state['pursuer_action']}")
            st.write(f"**Force:** ({current_state['pursuer_force'][0]:.2f}, {current_state['pursuer_force'][1]:.2f})")
            st.write(f"**Reward:** {current_state['pursuer_reward']:.2f}")

        st.markdown("### 🔵 Evader")
        st.write(f"**Position:** ({current_state['evader_pos'][0]:.2f}, {current_state['evader_pos'][1]:.2f})")
        st.write(f"**Velocity:** ({current_state['evader_vel'][0]:.2f}, {current_state['evader_vel'][1]:.2f})")
        if current_state['evader_action'] is not None:
            st.write(f"**Action:** {current_state['evader_action']}")
            st.write(f"**Force:** ({current_state['evader_force'][0]:.2f}, {current_state['evader_force'][1]:.2f})")
            st.write(f"**Reward:** {current_state['evader_reward']:.2f}")

        # Distance between agents
        distance = np.linalg.norm(current_state['pursuer_pos'] - current_state['evader_pos'])
        st.markdown("### 📏 Distance")
        st.write(f"**Pursuer ↔ Evader:** {distance:.2f}")
        st.write(f"**Capture Radius:** {config.capture_radius}")

        if current_state['done']:
            st.markdown("---")
            if current_state['captured']:
                st.success("🎯 **Episode Complete: Pursuer Captured Evader!**")
            elif current_state['timeout']:
                st.warning("⏰ **Episode Complete: Timeout - Evader Escaped!**")
        '''



    # Navigation controls at the bottom
    st.divider()
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])

    with col1:
        if st.button("⏮️ First", use_container_width=True):
            st.session_state['current_step'] = 0

    with col2:
        if st.button("◀️ Prev", use_container_width=True):
            st.session_state['current_step'] = max(0, st.session_state['current_step'] - 1)

    with col3:
        # Use a different key for the slider and sync manually
        slider_value = st.slider(
            "Step",
            min_value=0,
            max_value=len(episode_data) - 1,
            value=st.session_state['current_step'],
            key='step_slider'
        )
        # Update current_step from slider if it changed
        st.session_state['current_step'] = slider_value

    with col4:
        if st.button("▶️ Next", use_container_width=True):
            st.session_state['current_step'] = min(len(episode_data) - 1, st.session_state['current_step'] + 1)

    with col5:
        if st.button("⏭️ Last", use_container_width=True):
            st.session_state['current_step'] = len(episode_data) - 1

if __name__ == "__main__":
    main()
