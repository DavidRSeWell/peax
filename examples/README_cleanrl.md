# CleanRL DQN for PursuerEvaderEnv

This script adapts the [CleanRL](https://github.com/vwxyzjn/cleanrl) JAX DQN implementation to work with the PursuerEvaderEnv.

## Overview

The implementation uses:
- **Flax** for neural network definitions (instead of Equinox used in other examples)
- **CleanRL-style architecture** with clean, modular components
- **JAX JIT compilation** for fast training
- **Target network** for stability (hard updates every 500 steps)
- **Epsilon-greedy exploration** with linear decay

## Installation

Install the required dependencies:

```bash
# Install Flax (required for this example)
pip install flax

# Or install all example dependencies
pip install -e ".[examples]"
```

## Usage

### Basic Training

Run with default hyperparameters:

```bash
python examples/cleanrl_dqn.py
```

This will:
- Train for 500,000 timesteps
- Use a 5x5 discrete action grid (25 total actions)
- Save training statistics
- Run evaluation on 20 episodes after training

### Customizing Hyperparameters

You can modify the `Args` dataclass in the script to customize training:

```python
from examples.cleanrl_dqn import Args, main

# Create custom args
args = Args()
args.total_timesteps = 100000  # Shorter training
args.learning_rate = 1e-4      # Lower learning rate
args.batch_size = 64           # Smaller batches
args.num_actions_per_dim = 7   # Finer action discretization (7x7 = 49 actions)

# Then modify main() to use your args
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 500000 | Total training timesteps |
| `learning_rate` | 2.5e-4 | Adam learning rate |
| `buffer_size` | 10000 | Replay buffer capacity |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 128 | Training batch size |
| `start_e` | 1.0 | Initial epsilon (exploration) |
| `end_e` | 0.05 | Final epsilon |
| `exploration_fraction` | 0.5 | Fraction of training for epsilon decay |
| `learning_starts` | 10000 | Timesteps before learning starts |
| `train_frequency` | 10 | Timesteps between training updates |
| `target_network_frequency` | 500 | Timesteps between target network updates |
| `num_actions_per_dim` | 5 | Actions per dimension (total = this squared) |

## Architecture

### Q-Network
```
Input (9D observation)
  → Dense(120) + ReLU
  → Dense(84) + ReLU
  → Dense(num_actions)
```

### Training Loop
1. **Environment interaction**: Epsilon-greedy action selection
2. **Experience storage**: Store in replay buffer
3. **Training**: Sample batch and compute TD loss
4. **Target update**: Hard update every `target_network_frequency` steps

### Action Discretization
The continuous 2D force space is discretized into a grid:
- 5x5 grid = 25 discrete actions
- Each action maps to a force vector `[fx, fy]`
- Forces uniformly span `[-max_force, max_force]`

## Differences from Standard CleanRL

1. **Environment**: Uses PursuerEvaderEnv instead of Gymnasium
2. **Action space**: Discretizes continuous 2D forces into grid
3. **Observation**: Converts NamedTuple to flat array
4. **Multi-agent**: Trains pursuer vs random evader (not self-play)

## Differences from PEAX's Equinox Examples

| Feature | CleanRL DQN | PEAX DQN Examples |
|---------|-------------|-------------------|
| Framework | Flax | Equinox |
| Network def | `nn.Module` | `eqx.Module` with `eqx.nn.MLP` |
| Training state | Flax `TrainState` | Manual state management |
| Style | CleanRL conventions | Custom implementation |
| JIT | Manual `@jax.jit` | `@eqx.filter_jit` |

## Expected Performance

After training:
- **Capture rate**: ~60-80% against random evader
- **Average reward**: Positive (pursuer winning more often)
- **Training time**: ~10-20 minutes on CPU (depends on hardware)

## Tips

- **Faster training**: Reduce `total_timesteps` to 100k-200k for quick experiments
- **Better performance**: Increase `num_actions_per_dim` to 7 or 9 for finer control
- **More stable**: Increase `buffer_size` to 50k or 100k
- **Quicker convergence**: Reduce `learning_starts` to 5000

## References

- [CleanRL DQN JAX](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Flax Documentation](https://flax.readthedocs.io/)
