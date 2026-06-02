# Implementing and Training a Neural Network to Play Hex in PyTorch

This guide walks through a complete implementation: the Hex game environment, the neural network (policy + value heads sharing a residual trunk), Monte Carlo Tree Search, self-play data generation, and the training loop. Every code block is meant to be runnable.

---

## 1. Why Hex Maps Cleanly onto the AlphaGo Framework

| Property | Tic-Tac-Toe | **Hex** | Go |
|---|---|---|---|
| Board size | 3×3 | 11×11 (or 5×5) | 19×19 |
| Branching factor | ~5 | ~60 (11×11) | ~250 |
| Draws possible | Yes | **No** | Rare |
| Capture mechanic | No | No | Yes |
| Connection / territory | No | **Connection** | Territory |
| Solved by brute force | Yes | No | No |

Hex has **no draws** (proven by John Nash — one player must always win), which simplifies the value network target to a clean +1/−1. Its connection goal is spatially rich enough that a policy network must learn genuine pattern recognition, but the rules are simple enough to implement in ~100 lines of Python.

---

## 2. Prerequisites

```bash
pip install torch numpy
```

All code below assumes Python 3.9+ and PyTorch 2.x.

---

## 3. The Hex Game Environment

### 3.1 Rules

- The board is an N×N rhombus grid where each cell connects to its 6 neighbours.
- **Player 1 (Red)** wins by forming a connected path from the **top row to the bottom row**.
- **Player 2 (Blue)** wins by forming a connected path from the **left column to the right column**.
- Players alternate placing one stone per turn. No captures, no passes.

### 3.2 Neighbour structure

Unlike a square grid, each Hex cell has up to 6 neighbours (not 4):

```
     (r-1,c)  (r-1,c+1)
(r,c-1)  [ r,c ]  (r,c+1)
     (r+1,c-1)  (r+1,c)
```

### 3.3 Win detection with Union-Find

Checking connectivity after every move is efficient with a Union-Find (disjoint-set) structure. We add two **virtual nodes**: one connected to all top-row cells and one connected to all bottom-row cells. Player 1 wins when those two virtual nodes are in the same component.

```python
# hex_env.py

import numpy as np

EMPTY, RED, BLUE = 0, 1, 2

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class HexGame:
    """
    board[r][c] == EMPTY | RED | BLUE
    Player 1 = RED,  wins top→bottom  (row 0 → row N-1)
    Player 2 = BLUE, wins left→right  (col 0 → col N-1)
    """

    NEIGHBOURS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, size=11):
        self.size = size
        self.reset()

    def reset(self):
        N = self.size
        self.board       = np.zeros((N, N), dtype=np.int8)
        self.current     = RED        # RED moves first
        self.winner      = None
        self.move_count  = 0

        # Union-Find: N*N cells + 4 virtual nodes
        # virtual indices: N*N   = red_top
        #                  N*N+1 = red_bottom
        #                  N*N+2 = blue_left
        #                  N*N+3 = blue_right
        self._uf = UnionFind(N * N + 4)
        return self

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _idx(self, r, c):
        return r * self.size + c

    def _virtual(self, player, side):
        # side 0 = top/left virtual, side 1 = bottom/right virtual
        base = self.size * self.size
        if player == RED:
            return base + side          # 0 = top, 1 = bottom
        else:
            return base + 2 + side     # 2 = left, 3 = right

    def _connect_virtuals(self, r, c, player):
        N = self.size
        if player == RED:
            if r == 0:
                self._uf.union(self._idx(r, c), self._virtual(RED, 0))
            if r == N - 1:
                self._uf.union(self._idx(r, c), self._virtual(RED, 1))
        else:
            if c == 0:
                self._uf.union(self._idx(r, c), self._virtual(BLUE, 0))
            if c == N - 1:
                self._uf.union(self._idx(r, c), self._virtual(BLUE, 1))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def legal_moves(self):
        """Returns list of (r, c) tuples for empty cells."""
        return list(zip(*np.where(self.board == EMPTY)))

    def make_move(self, r, c):
        """Place a stone. Returns (next_state, reward, done)."""
        assert self.board[r, c] == EMPTY, "Cell already occupied"
        assert self.winner is None, "Game already over"

        player = self.current
        self.board[r, c] = player
        self.move_count += 1

        # Connect to same-colour neighbours
        for dr, dc in self.NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] == player:
                    self._uf.union(self._idx(r, c), self._idx(nr, nc))

        # Connect to virtual nodes
        self._connect_virtuals(r, c, player)

        # Check win
        if self._uf.connected(self._virtual(player, 0), self._virtual(player, 1)):
            self.winner = player
            reward = 1.0   # current player just won
            done   = True
        else:
            reward = 0.0
            done   = False

        # Switch player
        self.current = BLUE if player == RED else RED
        return reward, done

    def is_terminal(self):
        return self.winner is not None

    def encode(self):
        """
        Returns a (2, N, N) float32 tensor from the perspective of the
        current player.
          channel 0: cells occupied by the current player
          channel 1: cells occupied by the opponent
        """
        N = self.size
        me  = (self.board == self.current).astype(np.float32)
        opp = (self.board == (BLUE if self.current == RED else RED)).astype(np.float32)
        return np.stack([me, opp], axis=0)   # shape (2, N, N)

    def clone(self):
        g = HexGame(self.size)
        g.board      = self.board.copy()
        g.current    = self.current
        g.winner     = self.winner
        g.move_count = self.move_count
        g._uf        = UnionFind(self.size * self.size + 4)
        # Rebuild union-find from scratch (simple but correct)
        for r in range(self.size):
            for c in range(self.size):
                p = int(self.board[r, c])
                if p != EMPTY:
                    for dr, dc in self.NEIGHBOURS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.board[nr, nc] == p:
                                g._uf.union(g._idx(r, c), g._idx(nr, nc))
                    g._connect_virtuals(r, c, p)
        return g

    def render(self):
        symbols = {EMPTY: '.', RED: 'R', BLUE: 'B'}
        for r in range(self.size):
            print(' ' * r + ' '.join(symbols[int(c)] for c in self.board[r]))
```

---

## 4. The Neural Network

The architecture mirrors AlphaGo: a **shared residual trunk** that extracts spatial features, followed by two separate heads — one for the policy and one for the value.

```
Input (2 × N × N)
        │
  ┌─────▼──────┐
  │  Conv + BN │   ← initial projection to num_channels feature maps
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Res Block │ × num_res_blocks
  └─────┬──────┘
        │
   ┌────┴────┐
   │         │
Policy     Value
 head       head
   │         │
P(move)    V(s) ∈ [-1,+1]
```

```python
# hex_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)   # skip connection


class HexNet(nn.Module):
    """
    Input:  (batch, 2, board_size, board_size)
    Output: policy logits (batch, board_size²)   — raw scores, no softmax
            value         (batch, 1)              — tanh output in [-1, +1]
    """

    def __init__(self, board_size=11, num_channels=128, num_res_blocks=6):
        super().__init__()
        self.board_size = board_size
        N = board_size

        # Stem: map 2 input channels → num_channels feature maps
        self.stem    = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.tower = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head: reduce to 2 channels → flatten → linear → N² logits
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * N * N, N * N)

        # Value head: reduce to 1 channel → flatten → two FC layers → scalar
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(N * N, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.tower(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)                  # (batch, 2*N*N)
        p = self.policy_fc(p)             # (batch, N*N)  — raw logits

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)                  # (batch, N*N)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # (batch, 1)  — in [-1, +1]

        return p, v
```

### Why two heads on one network?

Training them jointly forces the trunk to learn features useful for *both* tasks. A feature like "I have a chain of stones almost connecting my sides" is both evidence that I am likely to win (value) *and* a reason to extend that chain (policy). Sharing the trunk means that signal is computed once and used twice.

---

## 5. Monte Carlo Tree Search

MCTS maintains a tree of game states. Each node stores:
- `N(s, a)` — visit count for action `a` from state `s`
- `W(s, a)` — total value accumulated
- `Q(s, a) = W/N` — mean value (our best estimate of how good the move is)
- `P(s, a)` — prior probability from the policy network

The **PUCT formula** balances exploitation (high Q) with exploration (high prior P, low visit count N):

```
U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
```

```python
# mcts.py

import math
import numpy as np
import torch

C_PUCT = 1.5   # exploration constant


class MCTSNode:
    def __init__(self, prior):
        self.prior    = prior   # P(s,a) from policy network
        self.visit_n  = 0
        self.value_w  = 0.0
        self.children = {}      # action -> MCTSNode

    @property
    def q_value(self):
        if self.visit_n == 0:
            return 0.0
        return self.value_w / self.visit_n

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        """PUCT: pick the child with the highest U score."""
        sqrt_total = math.sqrt(sum(c.visit_n for c in self.children.values()))
        best_score, best_action = -float('inf'), None
        for action, child in self.children.items():
            u = child.q_value + C_PUCT * child.prior * sqrt_total / (1 + child.visit_n)
            if u > best_score:
                best_score, best_action = u, action
        return best_action, self.children[best_action]

    def expand(self, policy_probs, legal_moves):
        """Create one child node per legal move, using policy_probs as priors."""
        for r, c in legal_moves:
            action = (r, c)
            self.children[action] = MCTSNode(prior=policy_probs[r, c])

    def backup(self, value):
        self.visit_n  += 1
        self.value_w  += value


class MCTS:
    def __init__(self, net, num_simulations=200, device='cpu'):
        self.net             = net
        self.num_simulations = num_simulations
        self.device          = device

    @torch.no_grad()
    def _evaluate(self, game):
        """Run the network on the current board state."""
        state = torch.tensor(
            game.encode(), dtype=torch.float32
        ).unsqueeze(0).to(self.device)            # (1, 2, N, N)

        policy_logits, value = self.net(state)

        N = game.size
        # Mask illegal moves (already occupied cells)
        mask = torch.tensor(
            (game.board == 0).flatten(), dtype=torch.float32
        ).to(self.device)
        policy_logits = policy_logits.squeeze(0)
        policy_logits[mask == 0] = -1e9           # set occupied cells to -inf

        policy_probs = torch.softmax(policy_logits, dim=0)
        policy_probs = policy_probs.cpu().numpy().reshape(N, N)
        value        = value.item()               # scalar in [-1, +1]
        return policy_probs, value

    def search(self, root_game):
        """
        Run num_simulations from the current game state.
        Returns a (N, N) array of visit-count-based move probabilities.
        """
        root = MCTSNode(prior=1.0)

        # Expand root immediately
        policy_probs, _ = self._evaluate(root_game)
        root.expand(policy_probs, root_game.legal_moves())

        for _ in range(self.num_simulations):
            node  = root
            game  = root_game.clone()
            path  = [node]

            # 1. Selection — walk down the tree
            while not node.is_leaf() and not game.is_terminal():
                action, node = node.select_child()
                game.make_move(*action)
                path.append(node)

            # 2. Expansion + Evaluation
            if game.is_terminal():
                # The player who just moved won; from the perspective of the
                # node we arrived at, the previous player won, so value = -1
                value = -1.0
            else:
                policy_probs, value = self._evaluate(game)
                node.expand(policy_probs, game.legal_moves())

            # 3. Backup — propagate value, negating at each level (two-player game)
            for node in reversed(path):
                node.backup(value)
                value = -value

        # Return move probabilities proportional to visit counts
        N = root_game.size
        visits = np.zeros((N, N), dtype=np.float32)
        for (r, c), child in root.children.items():
            visits[r, c] = child.visit_n
        visits /= visits.sum()
        return visits
```

### Why negate the value at each backup step?

At every edge in the tree, the active player switches. A position that is worth +0.8 to me is worth −0.8 to my opponent. Negating as we propagate upwards keeps the Q-values always relative to "the player whose turn it is at that node" — this is the **negamax** convention.

---

## 6. Self-Play Data Generation

Each self-play game produces a sequence of training examples:

```
(board_state, mcts_policy, outcome)
  (2×N×N)      (N×N)        ±1
```

The outcome is assigned after the game ends: +1 for the winner's moves, −1 for the loser's moves.

```python
# self_play.py

import numpy as np
from hex_env  import HexGame
from mcts     import MCTS


def self_play_game(net, board_size=11, num_simulations=200,
                   temperature=1.0, device='cpu'):
    """
    Play one game via MCTS self-play.
    Returns list of (encoded_state, mcts_policy, value_target) tuples.
    """
    game    = HexGame(size=board_size)
    mcts    = MCTS(net, num_simulations=num_simulations, device=device)
    records = []   # (state, policy)

    while not game.is_terminal():
        state  = game.encode()                    # (2, N, N)
        visits = mcts.search(game)                # (N, N)

        # Apply temperature to visit counts before sampling
        # High temperature (≈1) → explore broadly (early game)
        # Low temperature (→0)  → pick the best move (late game / evaluation)
        if temperature > 0:
            visits_t = visits ** (1.0 / temperature)
            visits_t /= visits_t.sum()
        else:
            visits_t = (visits == visits.max()).astype(np.float32)
            visits_t /= visits_t.sum()

        records.append((state, visits_t))

        # Sample a move from the visit distribution
        flat      = visits_t.flatten()
        move_idx  = np.random.choice(len(flat), p=flat)
        r, c      = divmod(move_idx, game.size)
        game.make_move(r, c)

    # Assign outcomes: winner's moves get +1, loser's moves get -1
    # records[i] was made by player whose turn it was at step i
    # game.winner is RED or BLUE; game.move_count tells us total moves made
    examples = []
    for i, (state, policy) in enumerate(records):
        # Player at step i: RED if i even (RED moves first), BLUE if i odd
        from hex_env import RED
        mover  = RED if i % 2 == 0 else (3 - RED)  # RED=1, BLUE=2
        outcome = 1.0 if mover == game.winner else -1.0
        examples.append((state, policy, outcome))

    return examples


def generate_dataset(net, num_games=100, board_size=11,
                     num_simulations=200, device='cpu'):
    dataset = []
    for i in range(num_games):
        examples = self_play_game(
            net, board_size=board_size,
            num_simulations=num_simulations,
            device=device
        )
        dataset.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  self-play game {i+1}/{num_games}  |  "
                  f"total examples: {len(dataset)}")
    return dataset
```

---

## 7. Training Loop

The loss has two components:
- **Policy loss:** cross-entropy between the MCTS visit distribution and the network's policy output.
- **Value loss:** mean squared error between the network's value output and the actual game outcome.

```
L = L_policy + L_value
  = CrossEntropy(π_mcts, π_net) + MSE(z, v_net)
```

```python
# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from hex_net   import HexNet
from self_play import generate_dataset


class HexDataset(Dataset):
    def __init__(self, examples):
        self.states   = torch.tensor(
            np.array([e[0] for e in examples]), dtype=torch.float32)
        self.policies = torch.tensor(
            np.array([e[1].flatten() for e in examples]), dtype=torch.float32)
        self.values   = torch.tensor(
            np.array([e[2] for e in examples]), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def train_epoch(net, loader, optimizer, device):
    net.train()
    total_loss = 0.0

    for states, target_policies, target_values in loader:
        states, target_policies, target_values = (
            states.to(device),
            target_policies.to(device),
            target_values.to(device),
        )

        policy_logits, value_pred = net(states)

        # Policy loss: cross-entropy
        # target_policies is a valid probability distribution from MCTS
        # log_softmax + (-target * log_pred).sum() == cross-entropy
        log_probs   = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, target_values)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def training_pipeline(
    board_size        = 5,      # use 5 for fast experiments; 11 for full Hex
    num_channels      = 64,
    num_res_blocks    = 4,
    num_iterations    = 10,     # outer loop: generate data → train → repeat
    games_per_iter    = 20,
    num_simulations   = 50,
    epochs_per_iter   = 5,
    batch_size        = 64,
    lr                = 1e-3,
    device_str        = 'cpu',
):
    device = torch.device(device_str)
    net    = HexNet(board_size, num_channels, num_res_blocks).to(device)
    opt    = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    for iteration in range(1, num_iterations + 1):
        print(f"\n=== Iteration {iteration}/{num_iterations} ===")

        # 1. Generate self-play data with the current network
        print("Generating self-play data...")
        examples = generate_dataset(
            net, num_games=games_per_iter,
            board_size=board_size,
            num_simulations=num_simulations,
            device=device_str,
        )

        # 2. Train on the generated data
        dataset = HexDataset(examples)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training...")
        for epoch in range(1, epochs_per_iter + 1):
            avg_loss = train_epoch(net, loader, opt, device)
            print(f"  epoch {epoch}/{epochs_per_iter}  loss={avg_loss:.4f}")

        # 3. Save a checkpoint
        torch.save(net.state_dict(), f"hex_net_iter{iteration:03d}.pt")
        print(f"Checkpoint saved: hex_net_iter{iteration:03d}.pt")

    return net


if __name__ == "__main__":
    trained_net = training_pipeline(
        board_size     = 5,
        num_channels   = 64,
        num_res_blocks = 4,
        num_iterations = 10,
        games_per_iter = 20,
        num_simulations= 50,
    )
```

---

## 8. Playing Against the Trained Network

```python
# play.py

import numpy as np
import torch
from hex_env import HexGame, RED, BLUE
from hex_net import HexNet
from mcts    import MCTS


def human_vs_ai(model_path, board_size=5, num_simulations=200):
    device = torch.device('cpu')
    net    = HexNet(board_size).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    mcts = MCTS(net, num_simulations=num_simulations)
    game = HexGame(size=board_size)

    print("You are RED (R). Connect TOP to BOTTOM.")
    print("AI is BLUE (B). Connect LEFT to RIGHT.")
    print("Enter moves as: row col  (0-indexed)\n")

    while not game.is_terminal():
        game.render()
        print()

        if game.current == RED:
            while True:
                try:
                    r, c = map(int, input("Your move (row col): ").split())
                    if (r, c) in game.legal_moves():
                        break
                    print("Illegal move, try again.")
                except (ValueError, IndexError):
                    print("Enter two integers.")
        else:
            print("AI is thinking...")
            visits = mcts.search(game)
            move   = np.unravel_index(visits.argmax(), visits.shape)
            r, c   = move
            print(f"AI plays: {r} {c}")

        game.make_move(r, c)

    game.render()
    winner_name = "RED (you)" if game.winner == RED else "BLUE (AI)"
    print(f"\nGame over — {winner_name} wins!")
```

---

## 9. File Structure

```
GoDeepNN/
├── hex_env.py       # HexGame + UnionFind
├── hex_net.py       # HexNet (residual trunk + policy head + value head)
├── mcts.py          # MCTSNode + MCTS
├── self_play.py     # self-play data generation
├── train.py         # training loop + pipeline entry point
└── play.py          # human vs AI interface
```

Run the full pipeline:

```bash
python train.py
```

Play against the result:

```bash
python play.py   # edit model_path inside play.py to point to the latest checkpoint
```

---

## 10. Practical Tips

| Topic | Recommendation |
|---|---|
| **Board size to start** | Use 5×5 — training converges in minutes. Move to 9×9 or 11×11 once the pipeline works. |
| **Simulations per move** | 50–100 for fast iteration, 400+ for strong play. The more simulations, the better the training signal but the slower data generation. |
| **GPU** | Training the network is fast even on CPU for small boards. If you have a GPU, pass `device_str='cuda'` to `training_pipeline`. Data generation (MCTS) remains CPU-bound regardless. |
| **Temperature schedule** | Use `temperature=1.0` for the first ~15 moves of each game (exploration), then `temperature=0` (greedy) for the rest. This mirrors AlphaGo's approach. |
| **Data augmentation** | The Hex board has a symmetry under 180° rotation and reflection along the main diagonal. Applying these to each training example doubles or quadruples your dataset for free. |
| **Evaluating progress** | Periodically pit the new network against the previous checkpoint (best of 10 games via MCTS). Replace the old network only if the new one wins ≥55% of games — this is the gating mechanism AlphaGo uses. |
| **Overfitting** | If value loss drops but policy loss stays high (or vice versa), try weighting the two losses differently: `loss = policy_loss + c * value_loss` with `c` between 0.5 and 2.0. |
