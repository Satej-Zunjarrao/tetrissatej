# Tetris AI — Deep Q-Network Agent

**[▶ Live Demo](https://satej-zunjarrao.github.io/tetrissatej)**

A reinforcement learning agent that learns to play Tetris through trial and error — trained from scratch in PyTorch, running live in the browser via TensorFlow.js.

---

## What This Is

This started as an interview problem: build Tetris. I built it, then went further — because the more interesting question isn't *how do you render a Tetris board*, it's *how do you teach a machine to play one*.

The agent has never been told the rules of Tetris. It learned entirely from reward signals — getting better over 1,000 episodes by figuring out what board states lead to good outcomes and which ones lead to death.

---

## How It Works

### The Agent — Deep Q-Network (DQN)

The agent uses a neural network to map board states to action values. At every step it asks: *given what the board looks like right now, what is each possible action worth?*

**Network architecture:**
```
Input (15 features) → 256 → 256 → 128 → Output (4 Q-values)
```

The four outputs correspond to: move left, move right, rotate, hard drop. The agent picks the action with the highest Q-value.

### What the Agent Sees — 15 Input Features

The agent never sees the board visually. It sees 15 numbers:

| Feature | Description |
|---|---|
| Column heights (×10) | How tall each of the 10 columns is |
| Total holes | Empty cells trapped beneath filled ones |
| Bumpiness | Sum of height differences between adjacent columns |
| Max height | Tallest column on the board |
| Average height | Mean column height |
| Lines cleared | Running total of lines cleared this game |

This is the complete picture the agent reasons from. No pixels. No visual understanding. Just numbers.

### The Reward Function — The Only Human Judgment

This is where engineering decisions matter. The reward function is the only thing I told the agent — everything else it figured out itself.

| Event | Reward | Reasoning |
|---|---|---|
| Clear a line | +100 | Primary objective |
| Clear 4 lines (Tetris) | +800 | Optimal play bonus — rewards planning ahead |
| Per hole created | −5 | Holes are nearly impossible to recover from |
| Per unit of bumpiness | −0.5 | Uneven surface limits future placement options |
| Per unit of max height | −0.3 | Penalises growing tall — death comes faster |
| Game over | −500 | Heavy penalty to make survival a core drive |

Designing this reward function is the same problem as defining any ML objective: there's no ground truth, you choose what to optimise for and accept the tradeoffs. Weight holes too heavily and the agent becomes overly conservative. Ignore bumpiness and it builds unplayable surfaces even while keeping low.

### Training — How the Agent Learns

**Experience Replay:** The agent stores every (state, action, reward, next state) tuple in a memory buffer. It learns from random batches of past experience — not just the most recent step. This breaks the correlation between consecutive states that would otherwise make learning unstable.

**Epsilon-Greedy Exploration:** The agent starts fully random (ε = 1.0) and gradually shifts to exploiting what it has learned (ε → 0.01). This ensures it explores enough of the state space before committing to a policy.

**Policy + Target Networks:** Two identical networks. The policy network updates every step. The target network updates every 500 steps. Without this, the agent chases a moving target — the same instability problem that prevented deep RL from working until DeepMind's 2013 DQN paper solved it with this exact technique.

**Bellman Equation (the core update):**
```
target = reward + γ × max(Q(next_state))
loss   = (predicted_Q - target)²
```

γ = 0.95 — the agent values near-term rewards slightly more than distant ones, like preferring a certain outcome now over an uncertain one later.

---

## Training Results

| Metric | Value |
|---|---|
| Episodes trained | 1,000 |
| Best score achieved | 200 |
| Final epsilon | 0.01 |
| Total parameters | 103,300 |

The score is modest — 1,000 episodes is early for Tetris. A fully trained DQN on Tetris typically needs 10,000–50,000 episodes. What matters here is that the agent demonstrably learned: it avoids holes, prefers flat surfaces, and survives longer than random play. The training curve in the dashboard shows the upward trend.

Given more compute time the natural next step would be: longer training, a convolutional input layer that processes the raw board visually, and a prioritised experience replay buffer that oversamples high-reward transitions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Training | Python, PyTorch |
| Training environment | Google Colab (T4 GPU) |
| Browser inference | TensorFlow.js |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | GitHub Pages |

---

## Run Locally

```bash
git clone https://github.com/Satej-Zunjarrao/tetrissatej
cd tetrissatej
npx serve .
# Open http://localhost:3000
```

No build step. No dependencies to install. The model weights are pre-baked into `model_weights.json`.

---

## Project Structure

```
tetrissatej/
├── index.html            # Game + dashboard + TF.js inference
├── model_weights.json    # Trained DQN weights exported from PyTorch
└── training_stats.json   # Episode scores and metadata for the chart
```

---

*Built as a demonstration of applied reinforcement learning — from environment design through reward engineering to browser deployment.*
