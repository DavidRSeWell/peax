"""
An implementation of iterative FPTA
"""
import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import scipy

from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from typing import Callable, List

def load_buffer_data(data_dir: str, batch_size: int):
    """Load self-play data from saved buffer.

    Args:
        data_dir: Directory containing buffer_state.pkl and metadata.pkl

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

    # Recreate buffer with same configuration
    estimated_size = metadata['total_transitions']

    # Create example transition for buffer initialization
    example_transition = {
        "observation": jnp.zeros(metadata['buffer_shape']['observation'], dtype=jnp.float32),
        "action": jnp.array(0, dtype=jnp.int32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "next_observation": jnp.zeros(metadata['buffer_shape']['next_observation'], dtype=jnp.float32),
        "done": jnp.array(0.0, dtype=jnp.float32),
        "agent_id": jnp.array(0, dtype=jnp.int32),
    }

    buffer = fbx.make_item_buffer(
        max_length=estimated_size,
        min_length=1,
        sample_batch_size=batch_size,
        add_batches=False,
    )

    print("✅ Data loaded successfully!")

    return buffer, buffer_state, metadata



class LSTQD:
    """
    Run Least squares TD learning
    """
    def __init__(self, basis: List[Callable]) -> None:

        self.basis = basis
        self.m = len(basis)

        key = jax.random.key(0) # Using 0 as the seed

        self.w = jax.random.uniform(key, shape=(self.m**2,))

    def get_p_obs(self, obs):
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

        return obs[:, :6], obs[:, 6:]

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

    def fit(self, buffer, buffer_state, batch_size, num_samples, seed):

        # Create random key
        key = jax.random.PRNGKey(seed)
        A = jnp.zeros((self.m**2, self.m**2))
        b = jnp.zeros((self.m**2, 1))
        M = batch_size*num_samples

        for _ in tqdm(range(num_samples)):
            # Note: Flashbax item buffer samples are in 'experience' field
            batch = buffer.sample(buffer_state, key)

            experience = batch.experience

            # Optional: filter by agent
            obs = experience['observation']
            #p_id = experience["agent_id"]
            #o_id = 1 - p_id
            #print(p_id)
            #print('---')
            #print(o_id)
            #p_id = jnp.expand_dims(p_id, 1)
            #o_id = jnp.expand_dims(o_id, 1)
            p1_obs, p2_obs = self.get_p_obs(obs)

            #p1_obs = jnp.concatenate((p_id, p1_obs), axis=1)
            #p2_obs = jnp.concatenate((o_id, p2_obs), axis=1)

            actions = experience['action']
            rewards = jnp.expand_dims(experience['reward'], 1)
            next_obs = experience['next_observation']
            p1_next_obs, p2_next_obs = self.get_p_obs(next_obs)

            #p1_next_obs = jnp.concatenate((p_id, p1_next_obs), axis=1)
            #p2_next_obs = jnp.concatenate((o_id, p2_next_obs), axis=1)

            dones = experience['done']

            #if experience["agent_id"][0] == 1:
            #    p1_obs , p2_obs = p2_obs, p1_obs
            #    p1_next_obs , p2_next_obs = p2_next_obs, p1_next_obs
            #    rewards = -rewards
            
            B_xy = self.get_players_B(p1_obs, p2_obs) # m x |S||A|
            B_yx = self.get_players_B(p2_obs, p1_obs) # m x |S||A|
            B_next_xy = self.get_players_B(p1_next_obs, p2_next_obs)
            B_next_yx = self.get_players_B(p2_next_obs, p1_next_obs)
            
            A_ = B_xy @ (B_xy - 0.99*B_next_xy).T
            b_ = B_xy @ rewards
            A += A_
            b += b_

            A_ = B_yx @ (B_yx - 0.99*B_next_yx).T
            b_ = B_yx @ -rewards

            A += A_
            b += b_


        A /= M
        b /= M
        
        print("----A---")
        print(tabulate(A))
        print("----b----")
        print(tabulate(b))

        C = jnp.linalg.inv(A) @ b
        C = C.reshape((self.m , self.m))
        
        print("-----C----")
        print(tabulate(C))

        return C







