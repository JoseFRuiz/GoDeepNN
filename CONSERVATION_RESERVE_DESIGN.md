# Idea: AlphaGo-Style Search for Conservation Reserve Design

A parking-lot note on adapting the policy + value + MCTS self-play pipeline in this repo (see `HEX_IMPLEMENTATION.md`) to spatial conservation planning. Not started — just captured so it isn't lost.

## The core idea

Reserve site selection has the same shape as Hex: a grid (or graph) of discrete cells, one action per step, and a score for the resulting configuration. That structural match is what makes this worth exploring — the same policy/value/MCTS machinery could plausibly transfer with the environment swapped out.

| Hex | Reserve design |
|---|---|
| N×N board of cells | Grid (or graph) of land parcels |
| Place a stone at (r,c) | Add parcel (r,c) to the reserve |
| Win = connect top to bottom | "Win" = reserve network meets coverage/connectivity targets under budget |
| `HexGame.make_move()` (exact rules) | A species-distribution / connectivity model scoring a candidate parcel set |
| Policy: which cell to play next | Policy: which parcel to add next |
| Value: who's winning from this position | Value: how good does this partial reserve look, given budget remaining |
| Self-play games | Simulated design runs against the ecological model |

## Why MCTS specifically (vs. simpler heuristics)

Reserve selection is a classic combinatorial optimization problem (related to the "maximum coverage" / set-cover family), usually solved today with greedy heuristics or integer programming (e.g., Marxan-style tools). MCTS's advantage would be searching *sequences* of parcel choices where early choices change the marginal value of later ones (e.g., connectivity is combinatorial — parcel A is only valuable if parcel B is also chosen) — something greedy heuristics handle poorly and ILP handles well but doesn't easily learn from experience across many problem instances the way a trained policy/value net would.

## The key risk (carried over from the ecology discussion)

Hex's simulator is exact and free to run millions of times — that's what makes self-play work. An ecological scoring model (species distribution model, connectivity index, etc.) is approximate and uncertain, and there's no way to "replay the real ecosystem" to validate self-play the way Hex games can be replayed. Any real attempt at this needs to either:
- search against an ensemble of plausible ecological models rather than one fixed simulator, or
- treat the model's uncertainty as part of the value target (e.g., value = expected score under model uncertainty, not a point estimate).

## Open questions to resolve before prototyping

- What's the scoring/simulator model? (species distribution model? habitat connectivity index? existing tool like Marxan or Zonation as a callable "environment"?)
- Discrete grid or a graph of real parcels with irregular shapes/costs?
- Single "reserve design episode" per self-play game, or a longer sequential/adaptive management horizon (add parcels over multiple years, budget renews)?
- Is there existing benchmark data (real conservation planning datasets) to validate against instead of pure simulation?

## Possible starting points to research later

- Marxan / Zonation (existing spatial conservation prioritization tools) — could serve as the "rules engine" / scoring function, similar to how `HexGame` provides exact rules here.
- Literature on RL for spatial conservation planning and green security games (the adjacent anti-poaching patrol problem, which is more directly adversarial/two-player).
- Whether a single-agent reformulation (value head only, no negamax) is more appropriate than the two-player framing used for Hex, since reserve design isn't adversarial.
