# Intro to AI - Berkeley Course Projects

Reinforcement learning implementations featuring Q-learning and value iteration agents for Pac-Man decision-making.

---

## Overview

Collection of AI coursework projects from UC Berkeley's Introduction to AI course. Focuses on implementing core reinforcement learning algorithms: Q-learning for temporal difference learning and value iteration for dynamic programming. Agents learn optimal policies in grid-based game environments.

---

## Technologies & Concepts

**Reinforcement Learning:** Q-learning, value iteration, temporal difference learning, Bellman equations  
**Algorithms:** Greedy policies, epsilon-greedy exploration, discount factors, learning rates  
**Game AI:** Agent decision-making, state-action spaces, reward functions  
**Optimization:** Policy convergence, convergence analysis, hyperparameter tuning  
**Python Programming:** Object-oriented agent design, algorithm implementation, performance analysis

---

## Project Structure

```
├── qlearningAgents.py       # Q-learning implementation
├── valueIterationAgents.py  # Value iteration algorithm
├── analysis.py              # Performance metrics & evaluation
└── README.md               # Project documentation
```

---

## Algorithms Implemented

**Q-Learning (Temporal Difference):**
- Off-policy learning algorithm
- Updates Q-values based on agent experience: `Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]`
- Learns optimal policy through trial and error
- Convergence guarantee with appropriate learning parameters

**Value Iteration (Dynamic Programming):**
- Iterative algorithm computing optimal value function
- Processes all states synchronously: `V(s) ← max_a Σ P(s'|s,a)[R(s,a,s') + γ·V(s')]`
- Guarantees optimal policy discovery
- Faster convergence than Q-learning in known environments

---

## Key Features

✓ Full Q-learning agent with exploration-exploitation trade-off  
✓ Value iteration for policy optimization  
✓ Performance analysis and convergence metrics  
✓ Modular agent architecture for extensibility  
✓ Complete coursework implementation with full marks

---

## Concepts Demonstrated

**Reinforcement Learning Fundamentals:**
- State-action-reward model
- Markov Decision Processes (MDPs)
- Value functions & Q-functions
- Policy evaluation & improvement

**Algorithm Design:**
- Bellman equations
- Dynamic programming
- Temporal difference methods
- Convergence analysis

**Practical Implementation:**
- Agent-environment interaction loops
- Epsilon-greedy policy
- Learning rate & discount factor tuning
- Performance evaluation

---

## Installation

```bash
python3 -c "import sys; print(sys.version)"  # Python 3.6+
```

---

## Skills Demonstrated

**Reinforcement Learning:** Q-learning, value iteration, policy gradient methods  
**Algorithm Implementation:** Bellman equations, convergence, optimization  
**AI Fundamentals:** MDPs, state spaces, reward design  
**Python Programming:** OOP agent design, numerical computation  
**Course Completion:** Full marks on all assignments

---

## Author

**Aroutin Nazarian** | Amirkabir University