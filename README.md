# Pursuer-Evader Environment (PEAX)

A JAX-based pursuer-evader reinforcement learning environment following the OpenAI Gym interface.

## Overview

This is a 2-player zero-sum game where:
- **Pursuer**: Tries to catch the evader
- **Evader**: Tries to avoid being caught until time runs out

## Features

- JAX-native implementation for fast vectorization and GPU acceleration
- Configurable boundary geometries (square, triangle, etc.)
- Double integrator dynamics for both agents (point masses)
- Gymnasium-compatible interface
- Functional programming style with stateless class methods

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from peax import PursuerEvaderEnv

# Create environment
env = PursuerEvaderEnv(boundary_type="square", boundary_size=10.0)

# Reset environment
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)

# Take a step
action_pursuer = jnp.array([1.0, 0.0])  # Force applied by pursuer
action_evader = jnp.array([-1.0, 0.0])  # Force applied by evader
actions = {"pursuer": action_pursuer, "evader": action_evader}

state, obs, reward, done, info = env.step(state, actions)
```

## Dynamics

Both agents are point masses with double integrator dynamics:
- State: [x, y, vx, vy]
- Control: [fx, fy] (force inputs)
- Dynamics: acceleration = force / mass

## Winning Conditions

- **Pursuer wins**: Catches evader (distance < capture_radius) before time limit
- **Evader wins**: Time limit reached without being caught

## Examples

The `examples/` directory contains reinforcement learning implementations:

### 1. DQN for Pursuer (vs Random Evader)

Train a pursuer agent using DQN against a random evader:

```bash
# Install with example dependencies
pip install -e ".[examples]"

# Run DQN training
python examples/dqn_pursuer.py
```

### 2. DQN with Self-Play (Separate Networks)

Train both pursuer and evader using competitive self-play with separate networks:

```bash
python examples/dqn_selfplay.py
```

### 3. DQN with Self-Play (Shared Network)

Train both agents using a single shared network (symmetric learning):

```bash
python examples/dqn_shared.py
```

All examples include:
- Q-Networks implemented with Equinox MLP
- Experience replay buffers
- Epsilon-greedy exploration
- Training visualization and evaluation

See [examples/README.md](examples/README.md) for detailed documentation and comparison.

## Development

### Running the Demo

```bash
python main.py
```

### Running Tests

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=peax --cov-report=html
```
