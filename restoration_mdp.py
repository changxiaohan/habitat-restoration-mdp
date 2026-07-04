from itertools import combinations

Sites = range(9)

# The state is a 9-tuple where each element is
# -1 = lost, 0 = fragile, 1 = restored
# GenerateOutcomes returns a list of tuples with a probability
# in position 0 and a state as a tuple in position 1
def GenerateOutcomes(State, LossProb):
    ans = []
    tempSites = [j for j in Sites if State[j]==0]
    n = len(tempSites)
    for i in range(n+1):
        for tlist in combinations(tempSites, i):
            p = 1.0
            slist = list(State)
            for j in range(n):
                if tempSites[j] in tlist:
                    p *= LossProb[tempSites[j]]
                    slist[tempSites[j]] = -1
                else:
                    p *= 1-LossProb[tempSites[j]]
            ans.append((p, tuple(slist)))
    return ans

# example
outcomes = GenerateOutcomes((1,0,0,-1,-1,0,1,1,-1), [0.2 for j in Sites])


# Species hosted by each site (site index → list of species IDs)
Species = [
    [8, 9, 15, 16],
    [6, 19],
    [14, 18],
    [10, 11, 13, 17],
    [0, 14, 17, 18],
    [0, 2, 4, 5, 8, 10, 14, 16],
    [1, 3, 14],
    [12, 13, 18],
    [0, 7, 11, 16]
]

# Adjacency list: neighbours[i] = list of site indices adjacent to i
neighbors = [
    [1],
    [0, 2, 3],       # 1
    [1, 4],          # 2
    [1, 4, 7],       # 3
    [2, 3, 5, 8],    # 4
    [4, 6],          # 5
    [5],             # 6
    [3, 8],          # 7
    [4, 7]           # 8
]

# Fixed priority order for greedy restoration (lower index = higher priority)
pri = [5, 8, 4, 3, 0, 7, 6, 2, 1]

# communication12 : Greedy restoration using a fixed priority list
def com12(s):
    fragile = [j for j in Sites if s[j] == 0]
    if len(fragile) == 0:
        # No fragile sites left: count unique species covered by restored sites
        return len(set(sum((Species[j] for j in Sites if s[j] == 1), [])))
    else:
        # Among fragile sites, pick the one with highest priority (smallest index in pri)
        Sched = [j for j in pri if j in fragile]
        t = list(s)
        t[Sched[0]] = 1          # restore that site
        outcomes = GenerateOutcomes(t, [0.2 for _ in Sites])
        # Expected value = sum over outcomes: probability * value of next state
        return sum(p * com12(next_state) for p, next_state in outcomes)

# communication13 : Optimal restoration with constant loss probability (0.2)
_value13 = {}        # memoised values V_13(s)
_policy13 = {}       # optimal action pi*(s) for each state

def com13(s):
    # If state s has not been evaluated, compute its value and best action
    if s not in _value13:
        fragile = [j for j in Sites if s[j] == 0]   # F(s): fragile sites
        if len(fragile) == 0:
            # Terminal state: reward R(s) = number of distinct protected species
            result = len(set(sum((Species[j] for j in Sites if s[j] == 1), [])))
            _value13[s] = result
            _policy13[s] = None     # no action needed
        else:
            best = -1
            best_action = None
            # Try restoring each fragile site j and keep the one with highest expected value
            for j in fragile:
                t = list(s)
                t[j] = 1            # intermediate state after restoring j
                outcomes = GenerateOutcomes(t, [0.2 for _ in Sites])
                # Expected value = sum over random next states
                expected = sum(p * com13(next_state) for p, next_state in outcomes)
                if expected > best:
                    best = expected
                    best_action = j
            _value13[s] = best
            _policy13[s] = best_action
    return _value13[s]


# communication14 : Optimal restoration with neighbour‑dependent loss probability
_value14 = {}
_policy14 = {}

def com14(s):
    if s not in _value14:
        fragile = [j for j in Sites if s[j] == 0]
        if len(fragile) == 0:
            result = len(set(sum((Species[j] for j in Sites if s[j] == 1), [])))
            _value14[s] = result
            _policy14[s] = None
        else:
            best = -1
            best_action = None
            for j in fragile:
                t = list(s)
                t[j] = 1
                # Dynamic loss probability p_k based on number of lost neighbours
                loss_prob = [0] * 9
                for k in Sites:
                    if t[k] == 0:   # only fragile sites can become lost
                        n = sum(1 for nb in neighbors[k] if t[nb] == -1)
                        loss_prob[k] = 0.2 + 0.05 * n   # p_0 + Δp * n
                outcomes = GenerateOutcomes(t, loss_prob)
                expected = sum(p * com14(next_state) for p, next_state in outcomes)
                if expected > best:
                    best = expected
                    best_action = j
            _value14[s] = best
            _policy14[s] = best_action
    return _value14[s]


# communication15 : Maximise probability that all valued species survive
valued_species = {8, 12, 13, 16}    # important species set V
_value15 = {}
_policy15 = {}

def com15(s):
    if s not in _value15:
        fragile = [j for j in Sites if s[j] == 0]
        if len(fragile) == 0:
            # Terminal state: success = 1 if all valued species are protected
            survived = set(sum((Species[j] for j in Sites if s[j] == 1), []))
            _value15[s] = 1.0 if valued_species.issubset(survived) else 0.0
            _policy15[s] = None
        else:
            best_prob = -1.0
            best_action = None
            for j in fragile:
                t = list(s)
                t[j] = 1
                loss_prob = [0] * 9
                for k in Sites:
                    if t[k] == 0:
                        n = sum(1 for nb in neighbors[k] if t[nb] == -1)
                        loss_prob[k] = 0.2 + 0.05 * n
                outcomes = GenerateOutcomes(t, loss_prob)
                # Expected success probability (same as V_15 for next states)
                expected = sum(p * com15(next_state) for p, next_state in outcomes)
                if expected > best_prob:
                    best_prob = expected
                    best_action = j
            _value15[s] = best_prob
            _policy15[s] = best_action
    return _value15[s]

#Example how to use com15
print(com15((0,0,0,0,0,0,0,0,0)))

def best_action(state, com_func, policy_dict):
    if state not in policy_dict:
        com_func(state)   # triggers memoisation of both value and policy
    return policy_dict.get(state, None)

# Example: optimal first action from the initial all‑fragile state under com13
print(best_action((0,0,0,0,0,0,0,0,0), com13, _policy13))