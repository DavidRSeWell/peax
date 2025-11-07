# PEAX Examples

This directory contains examples of training reinforcement learning agents in the Pursuer-Evader environment.

## Installation

Install the package with example dependencies:

```bash
pip install -e ".[examples]"
```

This will install:
- `equinox` - JAX neural network library
- `optax` - JAX optimization library
- `matplotlib` - For plotting results

## Examples

### 1. Deep Q-Network (DQN) for Pursuer

`dqn_pursuer.py` - Trains a pursuer agent using Deep Q-Learning against a random evader.

**Features:**
- Q-Network implemented with Equinox MLP
- Experience replay buffer
- Epsilon-greedy exploration
- Target network for stable learning
- Discrete action space (grid of force values)

**Run the example:**

```bash
python examples/dqn_pursuer.py
```

**Key components:**
- `QNetwork`: Equinox module that maps observations to Q-values
- `ReplayBuffer`: Stores and samples transitions
- `train_dqn()`: Main training loop with epsilon-greedy exploration
- `evaluate_agent()`: Evaluates the trained agent

**Algorithm Details:**

The pursuer agent learns to catch a randomly-acting evader. The continuous 2D action space (force in x and y) is discretized into a grid (default: 5x5 = 25 actions).

- **Observation space**: 9 dimensions (own pos/vel, other pos/vel, time remaining)
- **Action space**: Discretized 2D force grid
- **Reward**: +1 for capture, -1 for timeout, 0 otherwise
- **Training**: DQN with experience replay and target networks

**Hyperparameters:**
- Learning rate: 1e-3
- Batch size: 64
- Discount factor (γ): 0.99
- Replay buffer: 10,000 transitions
- Epsilon decay: 0.995 per episode
- Target network update: Every 10 episodes

### 2. Deep Q-Network (DQN) with Self-Play

`dqn_selfplay.py` - Trains both pursuer and evader agents using Deep Q-Learning in a self-play setup.

**Features:**
- Two separate Q-Networks (one for pursuer, one for evader)
- Both agents learn simultaneously through competitive play
- Separate replay buffers for each agent
- Epsilon-greedy exploration for both agents
- Symmetric training setup

**Run the example:**

```bash
python examples/dqn_selfplay.py
```

**Key differences from single-agent DQN:**
- `train_dqn_selfplay()`: Trains both agents simultaneously
- Each agent has its own Q-network, target network, optimizer, and replay buffer
- Both agents explore with epsilon-greedy and improve over time
- Evaluation shows performance of both trained agents

**Algorithm Details:**

In self-play, both the pursuer and evader are controlled by learned policies. This creates a competitive co-evolution where:
- The pursuer learns to catch increasingly skilled evaders
- The evader learns to avoid increasingly skilled pursuers
- Both agents improve through competitive pressure

**Training dynamics:**
- Early training: Both agents explore randomly, high variance
- Mid training: Agents develop basic strategies
- Late training: Agents refine strategies, may reach equilibrium

**Expected behavior:**
- Capture rate should stabilize around 50% (balanced competition)
- Episode lengths may increase as evader improves
- Both agents should exhibit strategic behavior (pursuit/evasion patterns)

### 3. Deep Q-Network (DQN) with Shared Network

`dqn_shared.py` - Trains a single Q-network that both pursuer and evader use (symmetric self-play).

**Features:**
- Single shared Q-Network used by both agents
- Both agents query the same network for action selection
- Single replay buffer storing experiences from both agents
- Symmetric learning - network learns optimal play for both roles
- Similar to AlphaGo Zero's single-network approach

**Run the example:**

```bash
python examples/dqn_shared.py
```

**Key differences from separate networks:**
- Only one Q-network (not two separate networks)
- Both pursuer and evader use this single network
- Network sees observations from both perspectives
- Single optimizer and single target network
- 2x more training data per episode (both agents contribute)

**Algorithm Details:**

The key insight is that observations are already from each agent's perspective:
- Pursuer sees: `[own_pos, own_vel, other_pos, other_vel, time_remaining]`
- Evader sees: `[own_pos, own_vel, other_pos, other_vel, time_remaining]`

The network learns a general strategy: "given my state and opponent's state, what's the best action?"

**Advantages:**
- More sample efficient (network learns from both roles simultaneously)
- Inherently symmetric strategy
- Single network to train and deploy
- Natural curriculum learning (as network improves, both agents improve)

**Expected behavior:**
- Should converge to a Nash equilibrium strategy
- Capture rate around 50% (truly balanced play)
- Network learns "optimal" pursuit and evasion simultaneously
- May show interesting emergent strategies

## Comparison of Approaches

| Aspect | Single-Agent | Separate Networks | Shared Network |
|--------|-------------|-------------------|----------------|
| Q-Networks | 1 | 2 | 1 |
| Opponent | Random | Learned (separate) | Learned (same) |
| Replay Buffers | 1 | 2 | 1 |
| Training Data/Episode | 1x | 2x | 2x |
| Symmetry | No | No | Yes |
| Complexity | Low | High | Medium |
| Sample Efficiency | Low | Medium | High |

## Customization

You can modify the training by changing parameters in the `train_dqn()` function:

```python
q_network, episode_rewards, episode_lengths, losses = train_dqn(
    env,
    num_episodes=500,          # Number of training episodes
    batch_size=64,             # Batch size for updates
    learning_rate=1e-3,        # Adam learning rate
    gamma=0.99,                # Discount factor
    epsilon_start=1.0,         # Initial exploration
    epsilon_end=0.01,          # Final exploration
    epsilon_decay=0.995,       # Decay rate per episode
    buffer_size=10000,         # Replay buffer size
    target_update_freq=10,     # Target network update frequency
    num_actions_per_dim=5,     # Actions per dimension (5x5=25 total)
    hidden_size=128,           # Hidden layer size
    seed=0                     # Random seed
)
```

## Expected Results

After training for 500 episodes, the pursuer agent should:
- Learn to move towards the evader
- Achieve a capture rate > 80% against random evaders
- Show decreasing loss values over time
- Exhibit strategic behavior near boundaries

## Next Steps

Potential extensions:
1. Train the evader with DQN for a competitive setup
2. Implement multi-agent reinforcement learning (MARL)
3. Use continuous action spaces with policy gradients (PPO, SAC)
4. Add curriculum learning with different boundary types
5. Visualize agent trajectories during evaluation
