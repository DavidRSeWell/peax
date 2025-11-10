"""Example script showing how to load and use self-play data from Flashbax buffer.

This demonstrates loading the buffer state and sampling batches for offline RL training.
"""

import pickle
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import flashbax as fbx


def load_selfplay_data(data_dir: str):
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
        sample_batch_size=256,
        add_batches=False,
    )

    print("✅ Data loaded successfully!")

    return buffer, buffer_state, metadata


def sample_batch_example(buffer, buffer_state, batch_size: int = 32, seed: int = 0):
    """Example of sampling a batch from the buffer.

    Args:
        buffer: Flashbax buffer
        buffer_state: Loaded buffer state
        batch_size: Number of transitions to sample
        seed: Random seed
    """
    print(f"\nSampling batch of {batch_size} transitions...")

    # Create random key
    key = jax.random.PRNGKey(seed)

    # Sample batch
    # Note: Flashbax item buffer samples are in 'experience' field
    batch = buffer.sample(buffer_state, key)

    print("\nBatch structure:")
    print(f"  Type: {type(batch)}")

    # Access the experience data
    experience = batch.experience

    print("\nTransition data:")
    print(f"  observation shape: {experience['observation'].shape}")
    print(f"  action shape: {experience['action'].shape}")
    print(f"  reward shape: {experience['reward'].shape}")
    print(f"  next_observation shape: {experience['next_observation'].shape}")
    print(f"  done shape: {experience['done'].shape}")
    print(f"  agent_id shape: {experience['agent_id'].shape}")

    print("\nSample values (first transition):")
    print(f"  observation: {experience['observation'][0]}")
    print(f"  action: {experience['action'][0]}")
    print(f"  reward: {experience['reward'][0]:.2f}")
    print(f"  done: {experience['done'][0]}")
    print(f"  agent_id: {experience['agent_id'][0]} ({'pursuer' if experience['agent_id'][0] == 0 else 'evader'})")

    # Count pursuer vs evader transitions in batch
    pursuer_count = jnp.sum(experience['agent_id'] == 0)
    evader_count = jnp.sum(experience['agent_id'] == 1)
    print(f"\nBatch composition:")
    print(f"  Pursuer transitions: {pursuer_count} ({pursuer_count/batch_size*100:.1f}%)")
    print(f"  Evader transitions: {evader_count} ({evader_count/batch_size*100:.1f}%)")

    return batch


def filter_by_agent(batch, agent_id: int):
    """Filter batch to only include transitions from one agent.

    Args:
        batch: Sampled batch from buffer
        agent_id: 0 for pursuer, 1 for evader

    Returns:
        Filtered batch
    """
    experience = batch.experience
    mask = experience['agent_id'] == agent_id

    filtered = {
        'observation': experience['observation'][mask],
        'action': experience['action'][mask],
        'reward': experience['reward'][mask],
        'next_observation': experience['next_observation'][mask],
        'done': experience['done'][mask],
        'agent_id': experience['agent_id'][mask],
    }

    agent_name = 'pursuer' if agent_id == 0 else 'evader'
    print(f"\nFiltered to {agent_name} only: {jnp.sum(mask)} transitions")

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Load and inspect self-play data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test_run",
        help="Directory containing buffer data (default: data/test_run)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for sampling (default: 32)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Self-Play Data Loader Example")
    print("=" * 70)

    # Load data
    buffer, buffer_state, metadata = load_selfplay_data(args.data_dir)

    # Sample a batch
    batch = sample_batch_example(buffer, buffer_state, args.batch_size, args.seed)

    # Example: filter to pursuer transitions only
    print("\n" + "=" * 70)
    print("Filtering Example")
    print("=" * 70)
    pursuer_batch = filter_by_agent(batch, agent_id=0)
    evader_batch = filter_by_agent(batch, agent_id=1)

    print("\n" + "=" * 70)
    print("Usage in Training Loop")
    print("=" * 70)
    print("""
# In your training loop:
for step in range(num_training_steps):
    # Sample batch from buffer
    key, sample_key = jax.random.split(key)
    batch = buffer.sample(buffer_state, sample_key)

    # Get experience data
    experience = batch.experience

    # Optional: filter by agent
    mask = experience['agent_id'] == 0  # pursuer only
    obs = experience['observation'][mask]
    actions = experience['action'][mask]
    rewards = experience['reward'][mask]
    next_obs = experience['next_observation'][mask]
    dones = experience['done'][mask]

    # Use for training...
    loss = train_step(obs, actions, rewards, next_obs, dones)
    """)


if __name__ == "__main__":
    main()
