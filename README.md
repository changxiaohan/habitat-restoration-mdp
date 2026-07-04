# Habitat Restoration under Uncertainty – MDP Models

This repository implements a **sequential decision problem for ecological restoration** on a small network of nine habitat sites.  
At each step we choose which fragile site to restore; afterwards, the remaining fragile sites may become permanently lost with known probabilities.  
The process is modelled as a finite‑horizon Markov Decision Process (MDP) and solved with value iteration / recursive dynamic programming.

## Problem Setting

- **Nine sites** (0 … 8) arranged in a spatial network.
- Each site can be in one of three states:
  - `1` – restored (healthy, hosts species)
  - `0` – fragile (still restorable, but may be lost)
  - `-1` – lost (cannot be restored)
- **Actions**: pick a fragile site and restore it (instant, deterministic).
- **Transitions**: after restoration, every remaining fragile site `k` becomes lost with probability `p_k`; all loss events are independent.
  - Constant loss probability: `p_k = 0.2` for all fragile sites.
  - Neighbour‑dependent loss: `p_k = 0.2 + 0.05 × (#lost neighbours of k)`.
- **Reward**: at the terminal state (no fragile sites left) we compute:
  - *Species richness* – number of distinct species hosted by restored sites (models `com12`, `com13`, `com14`).
  - *Valued‑species survival* – success (1) if all species from a predefined important set `{8,12,13,16}` are present; failure (0) otherwise (model `com15`).

## Implemented Policies

| Function | Type | Loss model | Objective |
|----------|------|------------|-----------|
| `com12`  | Greedy (fixed priority list) | constant (0.2) | max expected species richness |
| `com13`  | Optimal (value iteration)   | constant (0.2) | max expected species richness |
| `com14`  | Optimal (value iteration)   | neighbour‑dependent | max expected species richness |
| `com15`  | Optimal (value iteration)   | neighbour‑dependent | max probability that **all** valued species survive |

Each optimal function uses **memoisation** (caching) to avoid recomputing states. The corresponding policy (which site to restore next) is also stored.

## Code Structure

- **State representation**: 9‑tuple of integers `(-1,0,1)`.
- **`GenerateOutcomes(State, LossProb)`**  
  Returns all possible next states after a restoration step, together with their probabilities.  
  *This is the stochastic transition kernel.*
- **`com12(s)`** – greedy heuristic: restores the fragile site with the highest fixed priority.
- **`com13(s)`**, **`com14(s)`**, **`com15(s)`** – optimal policies computed by trying all fragile sites and picking the one that maximises the expected future reward.
- **`best_action(state, com_func, policy_dict)`** – helper to extract the optimal action for a given state after the value function has been computed.

## Example Usage

```python
# Compute value and best action for the optimal constant‑loss policy
print(com13((0,0,0,0,0,0,0,0,0)))               # expected species count
print(best_action((0,0,0,0,0,0,0,0,0), com13, _policy13))

# Maximum probability that all valued species survive
print(com15((0,0,0,0,0,0,0,0,0)))