"""
An implementation of iterative FPTA
"""
import flashbax as fbx
import jax
import jax.numpy as jnp
import nashpy as nash
import numpy as np
import pickle
import scipy

from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from typing import Callable, List


def _abs_to_relative(obs, boundary_size=10.0, max_velocity=12.0):
    """Convert absolute observations to normalized relative observations.

    Handles both 12D (state-only) and 16D (state+action) observations.

    12D input layout:
        [p1_pos(2), p1_vel(2), p1_time, p1_id,
         p2_pos(2), p2_vel(2), p2_time, p2_id]
    12D output layout:
        [rel_pos(2), rel_vel(2), own_vel(2),
         -rel_pos(2), -rel_vel(2), own_vel(2)]

    16D input layout (with actions):
        [p1_pos(2), p1_vel(2), p1_time, p1_id, p1_action(2),
         p2_pos(2), p2_vel(2), p2_time, p2_id, p2_action(2)]
    16D output layout:
        [rel_pos(2), rel_vel(2), own_vel(2), own_action(2),
         -rel_pos(2), -rel_vel(2), own_vel(2), own_action(2)]

    Normalization: rel_pos / boundary_size, rel_vel / (2*max_vel),
    own_vel / max_vel. Actions are assumed already normalized to [-1, 1].
    """
    obs_dim = obs.shape[-1]
    has_actions = (obs_dim == 16)
    half = obs_dim // 2  # 6 or 8

    # Extract absolute features
    p1_pos = obs[..., 0:2]
    p1_vel = obs[..., 2:4]
    p2_pos = obs[..., half:half+2]
    p2_vel = obs[..., half+2:half+4]

    # Relative features (from p1's perspective)
    rel_pos = p2_pos - p1_pos
    rel_vel = p2_vel - p1_vel

    # Normalize
    rel_pos_norm = rel_pos / boundary_size
    rel_vel_norm = rel_vel / (2.0 * max_velocity)
    p1_vel_norm = p1_vel / max_velocity
    p2_vel_norm = p2_vel / max_velocity

    # Assemble views
    p1_parts = [rel_pos_norm, rel_vel_norm, p1_vel_norm]
    p2_parts = [-rel_pos_norm, -rel_vel_norm, p2_vel_norm]

    if has_actions:
        # Actions are at indices 6:8 and 14:16, already normalized
        p1_action = obs[..., 6:8]
        p2_action = obs[..., 14:16]
        p1_parts.append(p1_action)
        p2_parts.append(p2_action)

    p1_view = jnp.concatenate(p1_parts, axis=-1)
    p2_view = jnp.concatenate(p2_parts, axis=-1)

    return jnp.concatenate([p1_view, p2_view], axis=-1)


def _augment_with_actions(buffer_state, num_actions_per_dim=5):
    """Augment observations with both players' action vectors.

    The buffer stores transitions in alternating order:
        [pursuer_step_0, evader_step_0, pursuer_step_1, evader_step_1, ...]
    Each transition has only the acting player's action. This function
    reconstructs the opponent's action from the paired transition.

    Observations go from 12D to 16D:
        [p1_state(6), p1_action(2), p2_state(6), p2_action(2)]
    Actions are normalized to [-1, 1] by dividing by max force magnitude.

    Next-observations are augmented with the next timestep's actions.
    At terminal transitions (done=1), next-action values are irrelevant
    since they get multiplied by (1 - done) = 0 during fitting.
    """
    exp = buffer_state.experience
    obs = exp['observation']       # (1, N, 12)
    next_obs = exp['next_observation']  # (1, N, 12)
    actions = exp['action']        # (1, N) discrete indices

    N = obs.shape[1]

    # Build action lookup table: index -> normalized 2D force in [-1, 1]
    forces_1d = jnp.linspace(-1.0, 1.0, num_actions_per_dim)
    # Use 'ij' indexing to match discretize_action: fx_idx = k // N, fy_idx = k % N
    fx, fy = jnp.meshgrid(forces_1d, forces_1d, indexing='ij')
    action_table = jnp.stack([fx.ravel(), fy.ravel()], axis=-1)  # (num_actions, 2)

    # Own action is stored in each transition
    own_vecs = action_table[actions[0]]  # (N, 2)

    # Opponent action: even index i -> opp at i+1, odd index i -> opp at i-1
    idx = jnp.arange(N)
    opp_idx = jnp.where(idx % 2 == 0, idx + 1, idx - 1)
    opp_idx = jnp.clip(opp_idx, 0, N - 1)
    opp_vecs = action_table[actions[0, opp_idx]]  # (N, 2)

    # Next-step actions (one timestep forward):
    # Even i (pursuer step k): next_own = actions[i+2], next_opp = actions[i+3]
    # Odd  i (evader  step k): next_own = actions[i+2], next_opp = actions[i+1]
    next_own_idx = jnp.clip(idx + 2, 0, N - 1)
    next_opp_idx = jnp.where(idx % 2 == 0, idx + 3, idx + 1)
    next_opp_idx = jnp.clip(next_opp_idx, 0, N - 1)

    next_own_vecs = action_table[actions[0, next_own_idx]]  # (N, 2)
    next_opp_vecs = action_table[actions[0, next_opp_idx]]  # (N, 2)

    # Augment: [p1_state(6), p1_action(2), p2_state(6), p2_action(2)] = 16D
    new_obs = jnp.concatenate([
        obs[0, :, :6], own_vecs,
        obs[0, :, 6:], opp_vecs,
    ], axis=-1)[None, ...]  # (1, N, 16)

    new_next_obs = jnp.concatenate([
        next_obs[0, :, :6], next_own_vecs,
        next_obs[0, :, 6:], next_opp_vecs,
    ], axis=-1)[None, ...]  # (1, N, 16)

    new_exp = {**exp, 'observation': new_obs, 'next_observation': new_next_obs}
    return buffer_state.replace(experience=new_exp)


def load_buffer_data(data_dir: str, batch_size: int, normalize: bool = True,
                     boundary_size: float = 10.0, max_velocity: float = 12.0,
                     include_actions: bool = False, num_actions_per_dim: int = 5):
    """Load self-play data from saved buffer.

    Args:
        data_dir: Directory containing buffer_state.pkl and metadata.pkl
        normalize: If True, convert absolute obs to normalized relative obs
        boundary_size: Environment boundary size (for normalization)
        max_velocity: Approximate max velocity (for normalization)
        include_actions: If True, augment observations with normalized action
            vectors (12D -> 16D, per-player 6D -> 8D)
        num_actions_per_dim: Number of discrete actions per dimension (for
            action index -> force vector conversion)

    Returns:
        Tuple of (buffer, buffer_state, metadata)
    """
    data_path = Path(data_dir)

    # Load metadata
    print("Loading metadata...")
    with open(data_path / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    print("\nDataset Info:")
    print("=" * 70)
    print(f"Episodes collected: {metadata['num_episodes']}")
    print(f"Total transitions: {metadata['total_transitions']}")
    print(f"Captures: {metadata['captures']} ({metadata['captures']/metadata['num_episodes']*100:.1f}%)")
    print(f"Timeouts: {metadata['timeouts']} ({metadata['timeouts']/metadata['num_episodes']*100:.1f}%)")
    print(f"Avg Pursuer Reward: {jnp.mean(jnp.array(metadata['episode_rewards_pursuer'])):.2f}")
    print(f"Avg Evader Reward: {jnp.mean(jnp.array(metadata['episode_rewards_evader'])):.2f}")
    print("\nBuffer Shapes:")
    for key, shape in metadata['buffer_shape'].items():
        print(f"  {key}: {shape}")
    print("=" * 70)

    # Load buffer state
    print("\nLoading buffer state...")
    with open(data_path / "buffer_state.pkl", 'rb') as f:
        buffer_state = pickle.load(f)

    # Augment observations with action vectors (must happen BEFORE relative
    # transform since it uses the raw 12D layout with absolute positions)
    if include_actions:
        print(f"\nAugmenting observations with action vectors "
              f"(num_actions_per_dim={num_actions_per_dim})...")
        buffer_state = _augment_with_actions(buffer_state, num_actions_per_dim)
        print("  Observations augmented: 12D -> 16D (actions normalized to [-1, 1])")

    # Convert to relative + normalized observations
    if normalize:
        print(f"Converting to relative observations "
              f"(boundary_size={boundary_size}, max_velocity={max_velocity})...")
        exp = buffer_state.experience
        rel_obs = _abs_to_relative(exp['observation'], boundary_size, max_velocity)
        rel_next = _abs_to_relative(exp['next_observation'], boundary_size, max_velocity)
        new_exp = {**exp, 'observation': rel_obs, 'next_observation': rel_next}
        buffer_state = buffer_state.replace(experience=new_exp)
        per_player = 8 if include_actions else 6
        features = "rel_pos(2), rel_vel(2), own_vel(2)"
        if include_actions:
            features += ", action(2)"
        print(f"  Layout per player ({per_player}D): [{features}]")
        print("  All features normalized to ~[-1, 1]")

    # Recreate buffer with same configuration
    estimated_size = metadata['total_transitions']
    obs_shape = buffer_state.experience['observation'].shape[2:]

    # Create example transition for buffer initialization
    example_transition = {
        "observation": jnp.zeros(obs_shape, dtype=jnp.float32),
        "action": jnp.array(0, dtype=jnp.int32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "next_observation": jnp.zeros(obs_shape, dtype=jnp.float32),
        "done": jnp.array(0.0, dtype=jnp.float32),
        "agent_id": jnp.array(0, dtype=jnp.int32),
    }

    buffer = fbx.make_item_buffer(
        max_length=estimated_size,
        min_length=1,
        sample_batch_size=batch_size,
        add_batches=False,
    )

    print("Data loaded successfully!")

    return buffer, buffer_state, metadata



class LSTQD:
    """
    Run Least squares TD learning
    """
    def __init__(self, basis: List[Callable], num_actions: int, gamma: float = 0.99) -> None:

        self.basis = basis
        self.m = len(basis)
        self.A = num_actions
        self.gamma = gamma

        key = jax.random.key(0) # Using 0 as the seed

        self.w = jax.random.uniform(key, shape=(self.m**2,))

    def get_f_xy(self, p1_obs, p2_obs, p1_acts, p2_acts, C):
        f_xy = jnp.zeros((self.A, self.A))
        p1_acts = jnp.expand_dims(jnp.array(p1_acts), 1)
        p2_acts = jnp.expand_dims(jnp.array(p2_acts), 1)
        def get_f_xy(p1_obs_, p2_obs_, p1_act, p2_act, C_):
            p1_obs_a = jnp.concatenate((p1_obs_, p1_act), axis=1)
            p2_obs_a = jnp.concatenate((p2_obs_, p2_act), axis=1)
            p1_B_a = self.basis_eval(p1_obs_a)
            p2_B_a = self.basis_eval(p2_obs_a)
            return (p1_B_a @ C_ @ p2_B_a.T).mean()
        
        # now vectorize over i,j
        get_f_xy_vmap = jax.vmap(jax.vmap(get_f_xy, in_axes=(None, None, 0, None, None)), in_axes=(None, None, None, 0, None))
        f_xy = get_f_xy_vmap(p1_obs, p2_obs, p1_acts, p2_acts, C)
        return f_xy

    def act(self, p1_obs, p2_obs, p1_acts: List[jax.Array], p2_acts: List[jax.Array], C):
        """
        Take action based on current value function
        """
        #p1_B = self.basis_eval(p1_obs)
        #p2_B = self.basis_eval(p2_obs)
        # Now add all possible actions 
        #f_xy = p1_B @ C @ p2_B.T
        f_xy = self.get_f_xy(p1_obs, p2_obs, p1_acts, p2_acts, C)

        # Solve Nash eq
        game = nash.Game(f_xy)
        play_counts_generator = game.fictitious_play(iterations=10)
        play_counts_list = tuple(play_counts_generator)
        p1_strategy = np.array(play_counts_list[-1][0]) / np.sum(np.array(play_counts_list[-1][0]))
        p1_strategy /= np.sum(p1_strategy)
        p1_idx = np.random.choice(np.arange(len(p1_strategy)), p=p1_strategy)
        p2_strategy = np.array(play_counts_list[-1][1]) / np.sum(np.array(play_counts_list[-1][1]))
        p2_strategy /= np.sum(p2_strategy)
        p2_idx = np.random.choice(np.arange(len(p2_strategy)), p=p2_strategy)
        return p1_acts[p1_idx], p2_acts[p2_idx]


    def get_p_obs(self, obs):
        """Break up obs by player. Splits at midpoint to handle both
        12D (state-only) and 16D (state+action) observations."""
        half = obs.shape[1] // 2
        return obs[:, :half], obs[:, half:]

    def basis_eval(self, obs):

        B = jnp.hstack([jnp.expand_dims(jax.vmap(b)(obs), 1) for b in self.basis])

        return B

    def get_players_B(self, p1_obs, p2_obs):
        """
        Take players obs and eval on all basis.
        Get a |S| x |d| mat
        """
        p1_B = self.basis_eval(p1_obs)
        p2_B = self.basis_eval(p2_obs)

        def b_xy(i, j, p1_B_, p2_B_):
            return p1_B_[:,i]*p2_B_[:,j]
        
        b_xy_ = lambda t: b_xy(t[0], t[1], p1_B, p2_B)

        b_xy_ = jax.vmap(b_xy_)

        all_pairs = jnp.array([[i,j] for i in range(p1_B.shape[1]) for j in range(p2_B.shape[1])])

        B_xy = b_xy_(all_pairs)

        return B_xy

    def get_low_rank(self, C):
        #U, Q = jax.scipy.linalg.schur(C)
        U, Q = scipy.linalg.schur(C)

        pred_eigs = jnp.abs(scipy.linalg.eigvals(jnp.array(U)))

        if len(pred_eigs % 2) != 0:
            zero_row = jnp.zeros((pred_eigs.shape[0], 1))
            pred_eigs = jnp.concatenate((pred_eigs, jnp.array([0.0])), axis=0)
            Q = jnp.concatenate((Q, zero_row), axis=1)

        eigs_idx = jnp.argsort(pred_eigs)[::-1]
        L = jnp.expand_dims(pred_eigs[eigs_idx], 1)
        Q = Q[:, eigs_idx]
        return L, Q
    
    def Y(self, Q, L, x):
        print("get Y")
        print(Q.shape)
        print(L.shape)
        print(x.shape)
        b_x = jnp.vstack([b(x) for b in self.basis])
        print(b_x.max())
        return ((Q.T @ b_x) * jnp.sqrt(L)).flatten()

    def fit_D(self, D: List[tuple], C, p1_acts, p2_acts) ->  jnp.ndarray:
        """
        Fit value function using dataset D
        D: List of (obs, action, reward, next_obs, done)
        """

        A = jnp.zeros((self.m**2, self.m**2))
        b = jnp.zeros((self.m**2, 1))
        M = 0.0

        for (obs, action, reward, next_obs, done) in D:

            p1_obs, p2_obs = self.get_p_obs(obs)
            p1_act = action[:, 0]
            p2_act = action[:, 1]
            #append action to observation
            p1_obs = jnp.concatenate((p1_obs, p1_act), axis=1)
            p2_obs = jnp.concatenate((p2_obs, p2_act), axis=1)

            p1_next_obs, p2_next_obs = self.get_p_obs(next_obs)
            #act_ = jax.vmap(self.act, in_axes=(0,0,None,None,None))
            next_act = jnp.array([self.act(p1_next_obs[i][None,:], p2_next_obs[i][None,:], p1_acts, p2_acts, C) for i in range(p1_next_obs.shape[0])])
            next_act = next_act.squeeze()

            p1_next_obs = jnp.concatenate((p1_next_obs, next_act[:,0]), axis=1)
            p2_next_obs = jnp.concatenate((p2_next_obs, next_act[:, 1]), axis=1)

            # Mask for non-terminal transitions
            not_done = jnp.expand_dims(1.0 - done, 0)  # (1, N)

            B_xy = self.get_players_B(p1_obs, p2_obs) # m² x N
            B_yx = self.get_players_B(p2_obs, p1_obs) # m² x N
            B_next_xy = self.get_players_B(p1_next_obs, p2_next_obs) * not_done
            B_next_yx = self.get_players_B(p2_next_obs, p1_next_obs) * not_done

            A_ = B_xy @ (B_xy - self.gamma*B_next_xy).T
            b_ = B_xy @ reward[:,:1]
            A += A_
            b += b_

            A_ = B_yx @ (B_yx - self.gamma*B_next_yx).T
            b_ = B_yx @ -reward[:,:1]

            A += A_
            b += b_
            M += p1_obs.shape[0]

        A /= M
        b /= M

        C = jnp.linalg.pinv(A) @ b
        C = C.reshape((self.m , self.m))
        # Enforce skew-symmetry
        C = (C - C.T) / 2

        return C
    
    def fit_minimaxQ(self, checkpoint_path: str, buffer, buffer_state, batch_size, num_samples, seed, tranform_action: Callable = lambda x: x):
        """
        Assuming that we are trying to fit a Q function that is of the form Q(s, *, *)
        """
        q_network, params, config = load_checkpoint(checkpoint_path)

        # Create random key
        key = jax.random.PRNGKey(seed)
        A = jnp.zeros((self.m**2, self.m**2))
        b = jnp.zeros((self.m**2, 1))
        M = batch_size*num_samples

        tds = []
        for iter_ in tqdm(range(num_samples)):
            key, subkey = jax.random.split(key)
            # Note: Flashbax item buffer samples are in 'experience' field
            batch = buffer.sample(buffer_state, subkey)

            experience = batch.experience

            # Optional: filter by agent
            obs = experience['observation']
            p1_obs, p2_obs = self.get_p_obs(obs)
            actions = experience['action']
            rewards = jnp.expand_dims(experience['reward'], 1)
            next_obs = experience['next_observation']
            p1_next_obs, p2_next_obs = self.get_p_obs(next_obs)
            dones = experience['done']

            # Mask for non-terminal transitions
            not_done = jnp.expand_dims(1.0 - dones, 0)  # (1, N)

            B_xy = self.get_players_B(p1_obs, p2_obs) # m² x N
            B_yx = self.get_players_B(p2_obs, p1_obs) # m² x N
            B_next_xy = self.get_players_B(p1_next_obs, p2_next_obs) * not_done
            B_next_yx = self.get_players_B(p2_next_obs, p1_next_obs) * not_done

            A_ = B_xy @ (B_xy - self.gamma*B_next_xy).T
            b_ = B_xy @ rewards
            A += A_
            b += b_

            A_ = B_yx @ (B_yx - self.gamma*B_next_yx).T
            b_ = B_yx @ -rewards

            A += A_
            b += b_

            A_current = A / ((iter_ + 1)*num_samples)
            b_current = b / ((iter_ + 1)*num_samples)
            C = jnp.linalg.pinv(A_current) @ b_current

            B_x = self.basis_eval(p1_obs)
            B_y = self.basis_eval(p2_obs)
            B_next_x = self.basis_eval(p1_next_obs)
            B_next_y = self.basis_eval(p2_next_obs)
            C_mat = C.reshape((self.m, self.m))
            pred_ = jnp.sum(B_x @ C_mat * B_y, axis=1)
            pred_next_ = jnp.sum(B_next_x @ C_mat * B_next_y, axis=1) * (1.0 - dones)

            # TD Error
            td_error = (pred_ - (rewards.flatten() + self.gamma * pred_next_)).mean()
            tds.append(td_error)


        A /= M
        b /= M

        print("----A---")
        print(tabulate(A))
        print("----b----")
        print(tabulate(b))

        C = jnp.linalg.pinv(A) @ b
        C = C.reshape((self.m , self.m))
        C_raw = C
        # Enforce skew-symmetry
        C = (C - C.T) / 2

        print("-----C----")
        print(tabulate(C))

        return C, tds, C_raw

    def fit(self, buffer, buffer_state, batch_size, num_samples, seed, tranform_action: Callable = lambda x: x, verbose: bool = True):

        # Create random key
        key = jax.random.PRNGKey(seed)
        A = jnp.zeros((self.m**2, self.m**2))
        b = jnp.zeros((self.m**2, 1))
        M = batch_size*num_samples

        tds = []
        for iter_ in tqdm(range(num_samples)):
            key, subkey = jax.random.split(key)
            # Note: Flashbax item buffer samples are in 'experience' field
            batch = buffer.sample(buffer_state, subkey)

            experience = batch.experience

            # Optional: filter by agent
            obs = experience['observation']
            p1_obs, p2_obs = self.get_p_obs(obs)
            actions = experience['action']
            rewards = jnp.expand_dims(experience['reward'], 1)
            next_obs = experience['next_observation']
            p1_next_obs, p2_next_obs = self.get_p_obs(next_obs)
            dones = experience['done']

            # Mask for non-terminal transitions: zero out next-state basis at terminal steps
            not_done = jnp.expand_dims(1.0 - dones, 0)  # (1, N) for broadcasting with (m², N)

            B_xy = self.get_players_B(p1_obs, p2_obs) # m² x N
            B_yx = self.get_players_B(p2_obs, p1_obs) # m² x N
            B_next_xy = self.get_players_B(p1_next_obs, p2_next_obs) * not_done
            B_next_yx = self.get_players_B(p2_next_obs, p1_next_obs) * not_done

            A_ = B_xy @ (B_xy - self.gamma*B_next_xy).T
            b_ = B_xy @ rewards
            A += A_
            b += b_

            A_ = B_yx @ (B_yx - self.gamma*B_next_yx).T
            b_ = B_yx @ -rewards

            A += A_
            b += b_

            A_current = A / ((iter_ + 1)*num_samples)
            b_current = b / ((iter_ + 1)*num_samples)
            C = jnp.linalg.pinv(A_current) @ b_current

            B_x = self.basis_eval(p1_obs)
            B_y = self.basis_eval(p2_obs)
            B_next_x = self.basis_eval(p1_next_obs)
            B_next_y = self.basis_eval(p2_next_obs)
            C_mat = C.reshape((self.m, self.m))
            pred_ = jnp.sum(B_x @ C_mat * B_y, axis=1)
            pred_next_ = jnp.sum(B_next_x @ C_mat * B_next_y, axis=1) * (1.0 - dones)

            # TD Error
            td_error = (pred_ - (rewards.flatten() + self.gamma * pred_next_)).mean()
            tds.append(td_error)


        A /= M
        b /= M

        if verbose:
            print("----A---")
            print(tabulate(A))
            print("----b----")
            print(tabulate(b))

        C = jnp.linalg.pinv(A) @ b
        C = C.reshape((self.m , self.m))
        C_raw = C
        # Enforce skew-symmetry
        C = (C - C.T) / 2

        if verbose:
            print("-----C----")
            print(tabulate(C))

        return C, tds, C_raw







