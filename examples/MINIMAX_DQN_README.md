# Minimax-Q DQN for Pursuer-Evader Games

This implementation adapts the Minimax-Q algorithm (Littman, ICML 1994) to the pursuer-evader environment using deep Q-networks.

## Key Differences from Standard DQN (`cleanrl_dqn.py`)

### 1. Q-Function Representation

**Standard DQN:**
- Q(observation, action) → scalar
- Each agent has its own Q-network
- Input: agent-specific observation (8D relative coordinates)
- Output: Q-values for each action (e.g., 9 values for 3×3 grid)

**Minimax-Q DQN:**
- Q(global_state, pursuer_action, evader_action) → scalar
- Single centralized Q-network for both agents
- Input: global state (9D: both agents' positions, velocities, time)
- Output: Q-matrix of shape [num_pursuer_actions × num_evader_actions]

### 2. Global State vs. Agent Observations

**Global State (Minimax-Q):**
```python
[pursuer_pos_x, pursuer_pos_y,      # 2D
 pursuer_vel_x, pursuer_vel_y,      # 2D
 evader_pos_x, evader_pos_y,        # 2D
 evader_vel_x, evader_vel_y,        # 2D
 time_normalized]                    # 1D
# Total: 9D
```

**Agent Observation (Standard DQN):**
```python
[relative_pos_x, relative_pos_y,    # 2D
 relative_vel_x, relative_vel_y,    # 2D
 own_vel_x, own_vel_y,              # 2D
 time_normalized,                    # 1D
 agent_id]                           # 1D
# Total: 8D (relative coordinates)
```

### 3. Action Selection

**Standard DQN (ε-greedy):**
```python
if random() < epsilon:
    action = random_action()
else:
    action = argmax Q(obs, a)
```

**Minimax-Q (ε-greedy with minimax):**
```python
if random() < epsilon:
    pursuer_action = random_action()
    evader_action = random_action()
else:
    # Pursuer: max_a1 min_a2 Q(s, a1, a2)
    pursuer_action = argmax_a1 min_a2 Q(s, a1, a2)
    # Evader: min_a2 max_a1 Q(s, a1, a2)
    evader_action = argmin_a2 max_a1 Q(s, a1, a2)
```

### 4. Bellman Backup

**Standard DQN:**
```python
Q_target = r + γ * max_a' Q(s', a')
```

**Minimax-Q:**
```python
V(s') = max_a1 min_a2 Q(s', a1, a2)  # Minimax value
Q_target = r + γ * V(s')
```

### 5. Network Architecture

**Standard DQN:**
```
Input: obs (8D)
  ↓
Hidden: 120 → 84
  ↓
Output: num_actions (e.g., 9)
```

**Minimax-Q:**
```
Input: global_state (9D)
  ↓
Hidden: 256 → 256 → 128
  ↓
Output: num_pursuer_actions × num_evader_actions
Reshape to: [num_pursuer_actions, num_evader_actions]
```

## Theoretical Foundation

The Minimax-Q algorithm is designed for two-player zero-sum Markov games where:
- Pursuer tries to maximize reward
- Evader tries to minimize reward (or maximize negative reward)
- The game value is the solution to the minimax equilibrium

For zero-sum games, the Nash equilibrium can be found by solving:
```
V(s) = max_π1 min_π2 Σ_a1,a2 π1(a1) π2(a2) Q(s, a1, a2)
```

In the greedy case (deterministic policies), this simplifies to:
```
V(s) = max_a1 min_a2 Q(s, a1, a2)
```

## Usage

### Training
```bash
# Basic training
python minimax_dqn.py

# Custom configuration
python minimax_dqn.py total_timesteps=500000 learning_rate=1e-4 num_actions_per_dim=5

# Different boundary types
python minimax_dqn.py boundary_type=circle boundary_size=15.0
```

### Key Hyperparameters
- `total_timesteps`: Total training steps (default: 500000)
- `learning_rate`: Adam learning rate (default: 1e-4)
- `gamma`: Discount factor (default: 0.9)
- `num_actions_per_dim`: Actions per dimension (default: 3, giving 3²=9 actions per agent)
- `buffer_size`: Replay buffer size (default: 50000)
- `batch_size`: Training batch size (default: 128)

## Advantages of Minimax-Q

1. **Game-theoretic optimality**: Converges to Nash equilibrium in zero-sum games
2. **Centralized learning**: Uses global state for more informed decisions
3. **Theoretical guarantees**: Provably optimal in tabular case
4. **Robust policies**: Accounts for opponent's best response

## Limitations

1. **Scalability**: Joint action space grows as O(|A1| × |A2|)
2. **Zero-sum assumption**: Assumes perfectly adversarial opponents
3. **Centralized state**: Requires global state information
4. **Exploration**: Standard ε-greedy may not explore joint action space efficiently

## Future Extensions

- **Mixed strategies**: Use linear programming to find optimal mixed strategies instead of pure maximin
- **Function approximation for large action spaces**: Use separate actor networks
- **Correlated-Q**: Extend to general-sum games
- **Multi-agent exploration**: Implement joint action exploration strategies

## References

Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. In Machine Learning Proceedings 1994 (pp. 157-163). Morgan Kaufmann.
