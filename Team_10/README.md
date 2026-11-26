# Team 10

## Members

- Olivia Espiñeira 
- Iciar Adliño
- Laura Gonzalez
- Matteo Giorgetti
- Lara Iglesias Pérez

## Algorithms implemented

| Challenge | Algorithms | Where in the file |
|---|---|---|
| Road Trip | Dijkstra/UCS, A* (Haversine), iterated local search (2-opt + Or-opt + double-bridge), conflict-repairing decoder with negotiated-congestion routing, cheapest insertion on the real graph, restarts | `roadtrip_agent.py`: `ucs`, `astar`, `_best_first`, `local_search`, `_ils`, `_two_opt_or_opt`, `RoadTripSolver` (`decode`, `_negotiate`, `insertion`, `descend`, `run`) |
| Chess | TODO | `chess_agent.py`: TODO |
| Battleship | Belief-state filter (predict + filter) for moving ships, Bayes' rule for sonar and for the rival's move rate, CSP with backtracking + forward checking in target mode, CSP over SUNK events, value-of-information sonar rule | `battleship_agent.py`: `ShipBelief`, `Tracker` (`advance_to`, `apply_shot`, `apply_sonar`, `log_weight`, `attributions`, `_target_weights`), `predict_ship_belief`, `sonar_posterior`, `consistent_placements`, `_hunt_shot`, `_decide` |
| Catan | TODO | `catan_agent.py`: TODO |

## Training

Road Trip: no learned parameters; everything is computed during the 5 seconds.

TODO (Catan, and any other agent with learned values): method, number of games, how long it ran. Learned values are stored inside the `.py` file.

## Tests

**Road Trip.** `check_requirements("roadtrip_agent.py")`:

```
PASS  Road Trip  Uniform-cost search / Dijkstra     optimal on 8 city pairs, 384 nodes expanded
PASS  Road Trip  A* with a geographic heuristic     optimal on 8 city pairs, 176 nodes expanded
PASS  Road Trip  A* expands no more nodes than UCS  A* 176 vs UCS 384
PASS  Road Trip  Local search / optimization        tour improved from 16810 to 7480
4/4 requirements pass.
```

With the official judge (`run_game`), our agent beat `demo` and `demo_b` on the training instance and on random draws, every time, with 0 invalid submissions. Offline, we ran it on 301 instances made by the judge's own generator (the training set plus seeds 0 to 299). The clock was cut to **1.5 s** to imitate a Colab core about three times slower than our laptops, and in a stress test to **0.6 s**. Results are under "Baseline against improved version" below.

**Chess.** TODO: rivals, number of games, results.

**Battleship.** `check_requirements` (confirmed in Colab): 3/3 battleship requirements pass, and `consistent_placements` matched the judge's exact answer on 3,000 random cases. With the tournament engine, 200 to 300 games per pairing: the final version wins 92% against `demo_a`, 300–0 against `demo_b`, 94% against a rival that moves a ship every turn, and 251–49 against our v0 baseline. In Colab with the real judge, 20 games against `demo_a` and `demo_b` were all won with no incidents. Details under "Baseline against improved version" below.

**Catan.** TODO: rivals, number of games, results.

**Whole team.** TODO: paste the output of `check_requirements("Team_XX.zip")` (14/14) and the result of a full match against `demo_a` in Colab.

## Dependencies and known limitations

Road Trip uses only the Python standard library (`heapq`, `math`, `random`, `time`). Battleship uses `numpy` and `random`. Their limits are described in their technical explanations below.

TODO: libraries and known weaknesses of Chess and Catan.

## Technical explanation

### European Road Trip

#### Problem formulation
The map is an undirected weighted graph G = (V, E) with |V| = 105 cities, |E| = 262 roads and the weight w(e) in road km. Given the required set R ⊆ V (|R| = 25, with Madrid ∈ R), we look for a **simple cycle** C in G through Madrid with R ⊆ V(C) that minimises Σ w(e) over e ∈ C. Non-required cities may appear at most once. This is a Steiner-cycle problem, which is NP-hard. It is not a plain TSP, because joining the required cities by their shortest paths often makes two legs share a city, and then the route is invalid.

We split the problem into two layers:

- **Order layer.** A state is a permutation of R starting at Madrid. Actions are 2-opt reversals and Or-opt moves (moving a segment of 1 to 3 cities, possibly reversed).
- **Path layer (decoder).** It turns an order into a real cycle of cities, or reports that it cannot.

Key property: if D(a, b) is the shortest-path distance, then for any order π, LB(π) = Σ D(π_i, π_{i+1}) is a lower bound on every route that visits R in that order, because rerouting a leg can only lengthen it. The search uses this bound to prune: a move whose LB is not below the current route's km cannot improve it, so it is never decoded. If the final km equals the exact metric-TSP optimum, the route is a proven global optimum.

#### Algorithms
1. **UCS/Dijkstra and A\*** (`ucs`, `astar`, which share `_best_first`). A graph search with a closed set, where a node counts as expanded when it is popped. A\* uses f = g + h with h(n) = Haversine(n, goal), R = 6371 km. h is **admissible** because every road is at least as long as the straight line between its ends, so any path is at least as long as the straight line. It is also **consistent**, because the straight line obeys the triangle inequality: h(n) ≤ w(n, n') + h(n'). So A\* with a closed set is optimal and never expands more nodes than UCS (176 against 384 on the judge's test). Inside `solve`, Dijkstra from each required city builds the distance matrix D and its paths. Blocked-node Dijkstra (`_dijkstra_pair`) finds the shortest path that avoids cities already in the cycle.
2. **Decoder** (`RoadTripSolver.decode`). Each leg starts with its shortest path. Legs that collide (a shared city, or a required city crossed mid-leg) are repaired in three escalating levels:
   - (a) Reroute only the colliding legs around everything else, trying every priority order when there are 4 or fewer such legs.
   - (b) Also release their neighbouring legs.
   - (c) **Negotiated congestion routing**, as in the PathFinder algorithm used for FPGA routing (`_negotiate`). All legs are rerouted repeatedly, and entering a shared city costs extra km: a *present* penalty times its current occupancy, plus a *history* penalty that grows each round it stays shared. This continues until no city is shared.

   If the order is still infeasible, the decoder returns km + 20,000 per shared city. That gives the local search a gradient towards feasible orders instead of a flat wall of "infinity". Finally, each leg is re-optimised against the rest of the cycle (`_polish_legs`).
3. **Optimisation** (`descend`, `run`). This is a hill climb on real km over the full 2-opt + Or-opt neighbourhood:
   - Neighbours are sorted by lower bound, and the scan stops as soon as LB ≥ current km. This makes the move ordering exact *and* cheap.
   - The first improving move is accepted.
   - **Escaping local optima:** iterated local search with double-bridge kicks. The kicks alternate between two types: (A) re-optimise the kicked order on D first (fast, good for easy maps), and (B) descend on real km directly (keeps the search inside feasible regions on hard maps).
   - After 120 iterations without a gain, the search **restarts** from a fresh order: a random order snapped to a 2-opt optimum, or a randomised cheapest insertion. The global best is kept.
4. **Cheapest insertion on the real graph** (`insertion`). It grows a simple cycle one required city at a time, inserting each where it adds the fewest km along paths that avoid the cycle. It almost always yields a valid route even on maps where nearly all good orders are infeasible.
5. **`local_search`** (required function). The same 2-opt + Or-opt + double-bridge ILS (`_ils`) on the given distance matrix.

#### Baseline against improved version
Test: 301 judge-generated instances, 1.5 s per instance, every route validated with the judge's own `check_route`. The exact metric-TSP optimum, computed offline with an exact solver that is not part of the agent, gives a proven lower bound for each instance.

| Version | No valid route | Mean km | Proven optimal |
|---|---|---|---|
| B0: nearest-neighbour order, legs chained greedily | 296 / 301 | 14,561 (on the 5 it solved) | – |
| B1: + 2-opt on D, legs chained greedily | 75 / 301 | 13,086 (on the 226 it solved) | – |
| V1: + LB-pruned descent on real km, conflict repair (levels a-b) | 0 | 13,092 | 114 |
| V2: + negotiated routing, penalised decoding, cheapest insertion | 0 | 13,042 | 114 |
| **V3 (final): + restarts** | **0** | **13,041** | **150** (see below) |

- B1 *fails on a quarter of the draws*, which would lose the game outright. On the draws it does solve, it is 205 km longer than the final agent.
- V1 → V2 fixed the dangerous maps (bottlenecks in Romania, the Balkans and Iberia). There, V1 sent its first valid route only at 4.25 s, from its fallback, and up to 5,500 km too long. V2 sends it within 60 ms (seed 9: 18,245 → 12,884 km).
- V3 is never worse than V2, improves 4 instances by 278 km in total (seed 271: −533 km), and matches the best route found by any of our runs on all 301.
- **Optimality.** In 114 instances the route meets the metric-TSP lower bound, so it is optimal. For the rest, where the no-repeat rule makes that bound unreachable, we solved 36 instances exactly offline with a branch-and-cut MIP (HiGHS via SciPy, generalized subtour constraints separated by min-cut; not part of the agent). The agent's route equals the proven optimum on **all 36**. So 150 of the 301 routes are proven globally optimal, and no instance we checked showed a gap. On 19 instances we also ran OR-Tools CP-SAT for 25 s, and the agent matched it every time.
- **Robustness.** With the full 5 s instead of 1.5 s, no result changed on 20 slow-converging instances. With only 0.6 s (about 0.3 s of search), all 120 test instances still got a valid route, and only 3 were slightly longer.

#### Time and memory
`solve` reads `problem["time_limit_s"]` and stops at 0.9·limit − 0.25 s (4.25 s out of 5), because the judge's clock starts slightly before ours. The first valid route is submitted after a few milliseconds (66 ms on average, worst case under 1.1 s, all measured with the 1.5 s budget). After that, every strictly better route is submitted at once, and each one is first re-checked locally against all the judge's rules, so no invalid route is ever sent. Everything runs inside `try/except`. If the solver somehow produced nothing, an independent randomised insertion fallback (`_fallback`) still submits a valid route before the limit. Memory stays at a few MB: a 25×25 matrix plus a decode cache, which is cleared above 200,000 entries.

#### Limits
- The decoder is a heuristic. For a fixed order it does not guarantee the cheapest set of disjoint legs, and the agent itself does not prove optimality when the lower bound is unreachable. We only verified that offline.
- With more time we would add an exact branch-and-bound over the cycle, using the metric-TSP bound, to certify optimality inside the 5 s, and Or-opt moves with longer segments.

### Chess

#### Problem formulation
TODO

#### Algorithms
TODO

#### Baseline against improved version
TODO

#### Time and memory
TODO

#### Limits
TODO

### Battleship

#### Problem formulation
Battleship here is a partially observable, stochastic, sequential game [Session 2]. The **hidden state** is the position of every enemy ship. Each of our turns, the **actions** are to optionally move one of our undamaged ships one cell, then either shoot a cell or use sonar (2 per game). The **observations** are HIT / MISS / SUNK (with the sunk ship's length) and noisy sonar counts. We never see the rival's moves, its misses or its sonar.

The **goal** is to destroy all 17 enemy segments before the rival destroys ours. Since the game is a race, our objective is to minimise the number of turns we need. The **transition model** [Session 3] comes from the rules: an enemy ship moves at most one cell per rival turn, only one ship moves per turn, and a ship that has been hit can never move again.

What the agent can deduce **with certainty** [Session 11]:
- A hit ship stays where it is for the rest of the game.
- A miss proves only that the cell was empty at that moment.
- The number of hits not yet explained by sunk ships is always known exactly: total hits minus the lengths of the sunk ships.
- A SUNK shot is always the last hit on that ship. Combined with the rule that ships cannot overlap, this often identifies exactly which cells belonged to it.

#### Algorithms
All of them are in `battleship_agent.py`; the docstrings cite the course session of each one.

| Algorithm | Where | Course |
|---|---|---|
| Belief state: a probability for every position (top-left cell and orientation) of every enemy ship not yet sunk | `ShipBelief`, `Tracker` | Sessions 17 and 23 |
| PREDICT step with the transition model: with probability p the ship moves to a uniformly chosen legal neighbouring position, otherwise it stays | `predict_ship_belief`, `predict_np`, `Tracker.advance_to` | Session 17 |
| FILTER step: a miss removes every position covering that cell; a hit removes it for the ships that can still move | `zero_covering`, `Tracker.apply_shot` | Session 17 |
| Bayes' rule for sonar: prior over the count in the 3×3, likelihood from the sensor model, normalised posterior; the prior is obtained by marginalising over the ships | `sonar_posterior`, `Tracker.apply_sonar` | Sessions 23 and 25 |
| Bayes' rule over three hypotheses about the rival (moves never / 25% of turns / always): each hunting shot is evidence, and each hypothesis's belief is weighted by its posterior | `Tracker.log_weight`, `_hunt_shot` | Session 23 |
| Target mode as a CSP (details below), solved by backtracking with forward checking | `consistent_placements`, `Tracker._target_weights` | Sessions 8 and 9 |
| Logical inference over SUNK events, also a CSP: which hit cells belong to each sunk ship | `Tracker.attributions` | Sessions 8 and 11 |
| Value of information: use sonar only when no single cell looks promising | `_decide` | Session 17 |

The CSP in target mode has:
- **Variables:** the unsunk enemy ships.
- **Domains:** their positions through a hit cell.
- **Constraints:** every unexplained hit is covered exactly once; ships do not overlap; a ship cannot be fully hit unless it is sunk; and a ship cannot cover a cell that was missed *after* its first hit. A ship can move before its first hit, so earlier misses do not constrain it.

**Hunting** shoots the cell with the highest marginal probability of containing a ship. **Target mode** shoots the cell covered by the most consistent assignments.

**Why these choices.** The course's belief-state filter for a moving target (Session 17) extends directly to probabilities, which is what `predict_ship_belief` requires. Per-ship grids of 100 positions cost about 1 ms per turn. A joint belief over the whole fleet would not fit in 5 s.

**Defence.** Moving costs nothing, so the agent moves one undamaged ship every turn. It prefers sideways moves of short ships, which change all of their cells, and keeps away from our own damaged ships, which is where rivals aim next. Our ships are placed at random, with no two touching.

#### Baseline against improved version
**Method.** We ran games with the tournament's own engine (`games/battleship/engine.py`). Every observation and answer went through JSON, as in the real judge.
- **Rivals:** `demo_a`, `demo_b`, our own earlier versions, and a test rival that moves a ship on every turn. That last rival is the worst case for any belief filter.
- **Win rate:** 200 to 300 games per pairing, seeds 1–300.
- **Turns to sink the whole fleet:** win rate stops discriminating against weak rivals, so we also measured this against a passive rival that never threatens us. It moves never (q0), on 25% of turns (q25) or on every turn (q100). 150–300 seeds, from seed 1001.

Each cycle changed one thing, and the version files are kept in `versions/`.

| Version | Change | Turns to sink the fleet (q0 / q25 / q100) | Win % vs demo_a | Win % vs every-turn mover |
|---|---|---|---|---|
| v0 baseline | Probability map that assumes static ships, plus target mode | 45.4 / 64.0 / 67.2 | 66% | 50% |
| v1a | + filter with a fixed move rate of 0.4 | 53.4 / 69.3 / 66.4 (88 of 150 static games never finished) | — | — |
| v1 | + filter with Bayes over three move-rate hypotheses | 47.9 / 64.5 / 65.0 | 61% | 53% |
| v2 | + defence (move a ship every turn, no-touch placement) | 49.2 / 64.2 / 68.4 | 92% | 94% |
| **v3 (final)** | + sonar when the best cell is below 0.12 | **45.6 / 57.9 / 65.2** | **92%** | **94%** |

**Head-to-head results:**
- v1 against v0: 96–104, which is no significant difference.
- v2 against v1: 143–57.
- v3 against v2: 159–140 (1 draw).
- v3 against v0: 251–49.
- v3 against `demo_b`: 300–0.

We also tried sonar thresholds of 0.08 and 0.20. Both were worse than 0.12.

**What we learned:**
- **Defence was by far the largest gain.** Rivals that never re-shoot a cell they missed cannot find a ship that drifts into water they have already searched.
- **A fixed move rate was harmful.** Against a static rival, the filter spread the belief towards the centre and never searched an edge ship. Running several hypotheses and weighting them with Bayes fixed this. The filter is roughly neutral against these rivals: it costs about 2 turns against static ones and gains about 2 against movers. We keep it because it is the robust choice against unknown rivals.
- **Testing found one logic bug.** The agent assigned a sunk ship to the wrong cells when two ships touched, and then shot the same two cells forever. Solving the attribution as a CSP over all SUNK events fixed it.

`check_requirements` (confirmed in Colab): 3/3 battleship requirements pass (`consistent_placements` also matched the judge's exact answer on 3,000 random cases).

#### Time and memory
- **Clock:** all updates are incremental, so each turn processes only the new observations. Typical decisions take 1–10 ms. In Colab with the real judge, a whole game used 0.11 s of the 5 s clock on average and 0.22 s at most, over 20 games.
- **Expensive steps:** the backtracking in target mode is capped at 20,000 nodes per turn, and the transition structure is precomputed at import time, which is not timed.
- **Low clock:** below 0.6 s the agent switches to a cheap rule. It shoots next to unexplained hits, otherwise a random cell of the checkerboard pattern.
- **Safety:** every answer is validated against the judge's rules before it is returned. If the same observation is sent twice, the agent answers the same way.
- **Memory:** a few hundred small arrays.

#### Limits
- The beliefs of different ships are kept independently, so no-overlap between moving ships is ignored when hunting.
- The rival's behaviour is modelled with only three move rates. A rival that moves deliberately away from our shots is not modelled.
- The sonar rule is a simple threshold, not a full value-of-information calculation. Its gain is small.
- Our defence cannot use the rival's misses, because the game never reveals them.
- The large comparisons ran in-process on a laptop. Colab with the real judge was used to confirm them: `check_requirements` gave 3/3, and 20 games against `demo_a` and `demo_b` were all won with no incidents.

With more time we would:
- Model the no-overlap constraint jointly in the endgame, when few ships remain.
- Learn the rival's move rate continuously instead of choosing among three values.
- Choose sonar centres by expected information gain.


### Catan

#### Problem formulation
TODO

#### Algorithms
TODO

#### Baseline against improved version
TODO

#### Time and memory
TODO

#### Limits
TODO

## Use of generative AI

Road Trip: we used Claude (Anthropic) to help design, implement and benchmark the agent and to draft this section. We reviewed the code and can explain every algorithm in it.

Battleship: We used Claude (Anthropic) as a coding assistant for the Battleship agent. It helped design the belief-state filter, the CSP target mode and the Bayesian sonar update, write and debug the code, and run the comparison between versions. Every algorithm is based on material from the course sessions, and we reviewed and tested all of it ourselves in Colab with the official judge. We understand and can explain every part of the agent.

TODO: the other two agents.
## Catan

### Problem formulation
Two players, first to 7 victory points (VP). Dice and development cards are random, the rival's cards are hidden, and each agent has 5 s of decision time per game. The agent sees the board, its own cards, public information and the log, and it returns the id of one legal action. Goal: win.

The file has three policies:

| Policy | Function | What it is |
|---|---|---|
| Baseline | `heuristic_policy` | Fixed priority list. No search, no learned values. |
| Reinforcement learning | `q_policy` | Q-table learned offline. |
| Search (the one that plays) | `act` | Searches the actions of the turn and scores positions with a hand-written formula. |

`act` does **not** use the Q-table, because the learned policy is measurably much weaker (see the comparison).

### Markov decision process and reinforcement learning
A Markov decision process (MDP) is a model with states, actions, transitions and rewards. Ours is deliberately simplified.

**State** (`state_features`), 9 values:

| Value | Meaning |
|---|---|
| phase | Part of the game or turn given by the judge |
| stage | My points: 0 (up to 2), 1 (3 or 4), 2 (5 or more) |
| race | Visible points: ahead (1), level (0), behind (-1) |
| can road, can settlement, can city, can dev card | 1 if my cards pay for it, else 0 |
| open spot | 1 if I have a place to build a settlement now |
| big hand | 1 if I hold more than 7 cards |

**Action:** the action **type** (build road, bank trade, end turn and so on). Action ids change every turn, so they cannot be learned. A simple rule (`best_of_type`) then picks which road, trade or corner.

**Transitions:** not modeled. Q-learning learns from played games without a model of the dice or the rival.

**Reward:** VP gained (or lost) since my previous decision, plus 1 for winning or -1 for losing at the end. The VP term gives feedback during the game; the final term makes winning matter.

**Update rule** (`q_update`):

`Q_new = Q + alpha * (r + gamma * best_next - Q)`

- `Q`: current value of (state, action type), in discounted reward points.
- `r`: reward received since that action, in points.
- `best_next`: highest value among the action types available at my next decision (0 when the game ends), in points.
- `alpha`: learning rate, no unit; `max(0.02, n^-0.6)`, where `n` is the number of updates of that entry.
- `gamma`: discount, no unit, 0.99.

Worked example (illustrative numbers): `Q = 3.30`, `r = 1`, `best_next = 3.10`, `alpha = 0.02`, `gamma = 0.99`. Target = 1 + 0.99 * 3.10 = 4.069. `Q_new` = 3.30 + 0.02 * (4.069 - 3.30) = 3.315.

**Policy:** `q_policy` takes the action type with the highest learned value. If the baseline's action type was never learned in that state, it plays the baseline's move.

`action_value(observation, action_id)` returns the learned Q-value of the action's type (0 if never learned), plus 0.001 times a hand-written estimate to order actions of the same type. It is not a win probability.

### How we trained
- Opponent: `heuristic_policy`. The learner alternates seats.
- Exploration (epsilon-greedy: with probability epsilon, try a random action type): epsilon falls from 0.40 to 0.05 over the first 40,000 games, then stays at 0.05.
- 80,000 games in short batches with a saved checkpoint, about 20 minutes of computing on a 2-core Linux machine.
- Entries updated fewer than 30 times are discarded. The exported table has 938 entries over 259 states and is stored in `catan_agent.py`.
- Repeating a 2,000-game run, in one batch or two, gave an identical table.

To reproduce (`ai_agent_tournament` is the extracted course folder; `q_checkpoint.json` must not exist at the start):

```
python train_q_catan.py train  --root ai_agent_tournament --games 20000   # repeat until the total is 80000
python train_q_catan.py export --root ai_agent_tournament
```

### What we learned
| Training games | Q-learning against baseline (2,000 games, same boards) |
|---|---|
| 40,000 | 46.7% ± 1.7 |
| 80,000 | 50.4% ± 1.7 |

We chose the 80,000-game table from this result, so we checked it on new boards: **51.3% ± 1.2** (4,000 games). That is at most a very small gain over the baseline.

- The table does change behavior: in 500 test games it picked a different action type from the baseline in 2,988 of 11,807 decisions that offered a choice (25%). Most common: trading or building a road instead of ending the turn, and ending the turn instead of trading or playing Monopoly.
- Those changes barely affect the win rate.
- Our explanation (not proven): the state is too coarse, so positions needing different moves look the same; only the action type is learned; and dice noise is large compared with the difference between actions.
- More training helped from 40,000 to 80,000 games. We did not test beyond that.

### The search agent
**Value of a position** (`score`):

`V = sum_k(w_k * mine_k) - 0.55 * sum_k(w_k * rival_k) + 1.6 * (A_mine - 0.55 * A_rival)`

- `V`: score of the position, no unit.
- `k`: feature number, 34 features per player (listed in `FEATURES`): points, production per resource, expected turns until each purchase is affordable, reachable building spots, knights, ports and others.
- `mine_k`, `rival_k`: feature `k` for me and for the rival (counts, production, or 0 to 1 values, depending on the feature).
- `w_k`: hand-set weight of feature `k`, in score points per unit of the feature.
- `0.55`: how much the rival's position counts compared with mine (no unit).
- `1.6`: weight of progress toward Largest Army (score points).
- `A_mine`, `A_rival`: `(min(c, n) / n)^2`, no unit, where `c` is knights played plus held and `n` is knights needed to take Largest Army; 0 for its holder.

**Search** (`plan`): a beam search over my own turn. It keeps the 6 best partial sequences, up to 5 actions deep, and plays the first action of the best one. A drawn card is averaged over the cards not yet seen. Dice and the rival's reply are not searched.

### Planning: preconditions and effects
| Action | Preconditions | Effects |
|---|---|---|
| Road | 1 wood, 1 brick (or free); fewer than 15; empty edge connected to my network | Road added; Longest Road rechecked |
| Settlement | 1 wood, 1 brick, 1 sheep, 1 wheat; fewer than 5; on my road; no building here or next door | +1 VP; may cut the rival's road |
| City | 2 wheat, 3 ore; my settlement; fewer than 4 cities | +1 VP; that corner produces double |
| Development card | 1 sheep, 1 wheat, 1 ore; deck not empty | Unknown card (averaged) |
| Knight | Card held; no other card played this turn | Move robber; may take Largest Army |
| Year of Plenty / Monopoly | Same | 2 bank cards / all rival cards of one resource |
| Road Building | Same, and able to pay for a road (engine rule) | 2 free roads |
| Bank trade | 4, 3 or 2 equal cards, depending on ports | 1 card of another resource |

`next_actions` checks the preconditions and `apply_action` applies the effects.

### What can be deduced from public events
- Payouts, building costs, trades, Year of Plenty, Monopoly and cards stolen between the two players are all known to me.
- So the rival's hand is exact until some of its cards are discarded at random. After that it is an estimate, rescaled to the public card count. The bank running out of a resource is not modeled.
- Unseen development cards = deck composition minus my cards and all played cards.

### Baseline against improved versions
Each board is played twice with seats swapped. Boards differ from those used in training. ± is the half-width of the 95% confidence interval.

| Match | Games | Win rate of the first |
|---|---|---|
| Q-learning vs baseline | 4,000 | 51.3% ± 1.2 |
| Search vs baseline | 1,000 | **71.0% ± 2.8** |
| Search vs Q-learning | 1,000 | **71.9% ± 2.9** |
| Q-learning vs random moves | 1,000 | 100% |
| Baseline vs random moves | 1,000 | 99.7% ± 0.3 |

```
python train_q_catan.py evaluate --root ai_agent_tournament --games 4000 --seed0 900000 --pair qlearning baseline
python train_q_catan.py evaluate --root ai_agent_tournament --games 1000 --pair search baseline
python train_q_catan.py evaluate --root ai_agent_tournament --games 1000 --pair search qlearning
python train_q_catan.py evaluate --root ai_agent_tournament --games 1000 --pair qlearning random
python train_q_catan.py evaluate --root ai_agent_tournament --games 1000 --pair baseline random
```

**Course judge** (`run_game`, real clocks, match seeds 3000 onward): 52 wins in 60 games against `demo`, 20 in 20 against `demo_b`, no incidents. `check_requirements`: 4 of 4 Catan checks pass.

### How the search weights were chosen
The weights were set by hand, **not learned**. Changes were kept only if they won matches. These development matches used scripts and earlier versions that are not part of this submission, so the course folder alone cannot reproduce them.

| Change | Games | Win rate against previous best | Decision |
|---|---|---|---|
| Discount a hand above 7 cards by the risk of a 7 | 2,000 | 52.7% ± 1.5 | Kept |
| Same, plus a heavier penalty per card | 1,000 | 49.2% ± 2.4 | Rejected |
| Largest Army progress, weight 1.6 | 1,000 | 54.1% ± 2.1 | Kept |
| Same, weight 0.7 | 1,000 | 53.6% ± 1.8 | Not chosen (no clear difference from 1.6) |
| Three changes to the opening placement | 1,000 to 2,000 | 48.7% to 49.2% | Rejected |

### Time and memory
The search uses about 0.1 s of computation per game; in the judge games above at least 4.73 s of the 5 s remained. Below 0.8 s remaining, or after an internal error, `act` falls back to the baseline. Peak memory of a test process running the agent: about 36 MB.

### Limits
- No search over dice or the rival's reply.
- A knight is not considered in the middle of a planned turn, and there is no dedicated search for a winning sequence.
- Weights are hand-set, and no strong independent opponent was available.
- Q-learning is far weaker than the search. Next steps would be learning from numeric features (function approximation) instead of a small table, and training against stronger opponents.
- One engine behavior: cutting a rival's road can award Longest Road with fewer than 5 roads. The agent avoids giving those points away and does not try to obtain them.

### Dependencies
The agent uses only the Python standard library. The training script also needs the course folder (and its `catanatron` package). Tested on Linux with Python 3.13, not on Google Colab.

### Use of generative AI
Claude (Anthropic) designed, wrote, trained, tested and documented this agent and the training script, following the team's instructions. Before submitting, the team must describe here what it reviewed or changed itself, confirm that this complies with the course AI policy, and be able to explain every algorithm.