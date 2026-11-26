<!-- ===== Row for the "Algorithms implemented" table ===== -->

| Road Trip | Dijkstra/UCS, A* (Haversine), iterated local search (2-opt + Or-opt + double-bridge), conflict-repairing decoder with negotiated-congestion routing, cheapest insertion on the real graph, restarts | roadtrip_agent.py: ucs, astar, _best_first, local_search, _ils, _two_opt_or_opt, RoadTripSolver (decode, _negotiate, insertion, descend, run) |

<!-- ===== Lines for the "Tests" section ===== -->

Road Trip. check_requirements("roadtrip_agent.py"):

PASS  Road Trip  Uniform-cost search / Dijkstra     optimal on 8 city pairs, 384 nodes expanded
PASS  Road Trip  A* with a geographic heuristic     optimal on 8 city pairs, 176 nodes expanded
PASS  Road Trip  A* expands no more nodes than UCS  A* 176 vs UCS 384
PASS  Road Trip  Local search / optimization        tour improved from 16810 to 7480
4/4 requirements pass.

With the official judge (run_game), our agent beat demo and demo_b on the training instance and on random draws, every time, with 0 invalid submissions. Offline, we ran it on 301 instances made by the judge's own generator (the training set plus seeds 0 to 299), with the clock cut to 1.5 s to imitate a Colab core about three times slower than our laptops. Results are under "Baseline against improved version" below.

<!-- ===== Technical explanation ===== -->
European Road Trip
Problem formulation

The map is an undirected weighted graph G = (V, E) with |V| = 105 cities, |E| = 262 roads and the weight w(e) in road km. Given the required set R ⊆ V (|R| = 25, with Madrid ∈ R), we look for a simple cycle C in G through Madrid with R ⊆ V(C) that minimises Σ w(e) over e ∈ C. Non-required cities may appear at most once. This is a Steiner-cycle problem, which is NP-hard. It is not a plain TSP, because joining the required cities by their shortest paths often makes two legs share a city, and then the route is invalid.

We split the problem into two layers:

Order layer. A state is a permutation of R starting at Madrid. Actions are 2-opt reversals and Or-opt moves (moving a segment of 1 to 3 cities, possibly reversed).
Path layer (decoder). It turns an order into a real cycle of cities, or reports that it cannot.

Key property: if D(a, b) is the shortest-path distance, then for any order π, LB(π) = Σ D(π_i, π_{i+1}) is a lower bound on every route that visits R in that order, because rerouting a leg can only lengthen it. The search uses this bound to prune: a move whose LB is not below the current route's km cannot improve it, so it is never decoded. If the final km equals the exact metric-TSP optimum, the route is a proven global optimum.

Algorithms
UCS/Dijkstra and A* (ucs, astar, which share _best_first). A graph search with a closed set, where a node counts as expanded when it is popped. A* uses f = g + h with h(n) = Haversine(n, goal), R = 6371 km. h is admissible because every road is at least as long as the straight line between its ends, so any path is at least as long as the straight line. It is also consistent, because the straight line obeys the triangle inequality: h(n) ≤ w(n, n') + h(n'). So A* with a closed set is optimal and never expands more nodes than UCS (176 against 384 on the judge's test). Inside solve, Dijkstra from each required city builds the distance matrix D and its paths. Blocked-node Dijkstra (_dijkstra_pair) finds the shortest path that avoids cities already in the cycle.
Decoder (RoadTripSolver.decode). Each leg starts with its shortest path. Legs that collide (a shared city, or a required city crossed mid-leg) are repaired in three escalating levels:
(a) Reroute only the colliding legs around everything else, trying every priority order when there are 4 or fewer such legs.
(b) Also release their neighbouring legs.
(c) Negotiated congestion routing, as in the PathFinder algorithm used for FPGA routing (_negotiate). All legs are rerouted repeatedly, and entering a shared city costs extra km: a present penalty times its current occupancy, plus a history penalty that grows each round it stays shared. This continues until no city is shared.
If the order is still infeasible, the decoder returns km + 20,000 per shared city. That gives the local search a gradient towards feasible orders instead of a flat wall of "infinity". Finally, each leg is re-optimised against the rest of the cycle (_polish_legs).
Optimisation (descend, run). This is a hill climb on real km over the full 2-opt + Or-opt neighbourhood:
Neighbours are sorted by lower bound, and the scan stops as soon as LB ≥ current km. This makes the move ordering exact and cheap.
The first improving move is accepted.
Escaping local optima: iterated local search with double-bridge kicks. The kicks alternate between two types: (A) re-optimise the kicked order on D first (fast, good for easy maps), and (B) descend on real km directly (keeps the search inside feasible regions on hard maps).
After 120 iterations without a gain, the search restarts from a fresh order: a random order snapped to a 2-opt optimum, or a randomised cheapest insertion. The global best is kept.
Cheapest insertion on the real graph (insertion). It grows a simple cycle one required city at a time, inserting each where it adds the fewest km along paths that avoid the cycle. It almost always yields a valid route even on maps where nearly all good orders are infeasible.
local_search (required function). The same 2-opt + Or-opt + double-bridge ILS (_ils) on the given distance matrix.
Baseline against improved version

Test: 301 judge-generated instances, 1.5 s per instance, every route validated with the judge's own check_route. The exact metric-TSP optimum, computed offline with an exact solver that is not part of the agent, gives a proven lower bound for each instance.

Version	No valid route	Mean km	Proven optimal
B0: nearest-neighbour order, legs chained greedily	296 / 301	14,561 (on the 5 it solved)	–
B1: + 2-opt on D, legs chained greedily	75 / 301	13,086 (on the 226 it solved)	–
V1: + LB-pruned descent on real km, conflict repair (levels a-b)	0	13,092	114
V2: + negotiated routing, penalised decoding, cheapest insertion	0	13,042	114
V3 (final): + restarts	0	13,041	114
B1 fails on a quarter of the draws, which would lose the game outright. On the draws it does solve, it is 205 km longer than the final agent.
V1 → V2 fixed the dangerous maps (bottlenecks in Romania, the Balkans and Iberia). There, V1 sent its first valid route only at 4.25 s, from its fallback, and up to 5,500 km too long. V2 sends it within 60 ms (seed 9: 18,245 → 12,884 km).
V3 is never worse than V2, improves 4 instances by 278 km in total (seed 271: −533 km), and matches the best route found by any of our runs on all 301.
On the 19 instances we also solved with OR-Tools CP-SAT (25 s, 2 cores, offline only), the agent matched CP-SAT's best every time.
On 20 slow-converging instances, giving the agent the full 5 s instead of 1.5 s did not change a single result.

In 114 of 301 instances the route equals the proven lower bound, so it is the global optimum. In the rest, the no-repeat rule makes the bound unreachable, so the gap is not necessarily a loss.

Time and memory

solve reads problem["time_limit_s"] and stops at 0.9·limit − 0.25 s (4.25 s out of 5), because the judge's clock starts slightly before ours. The first valid route is submitted after a few milliseconds (66 ms on average, worst case under 1.1 s, all measured with the 1.5 s budget). After that, every strictly better route is submitted at once, and each one is first re-checked locally against all the judge's rules, so no invalid route is ever sent. Everything runs inside try/except. If the solver somehow produced nothing, an independent randomised insertion fallback (_fallback) still submits a valid route before the limit. Memory stays at a few MB: a 25×25 matrix plus a decode cache, which is cleared above 200,000 entries.

Limits
The decoder is a heuristic. For a fixed order it does not guarantee the cheapest set of disjoint legs, and the search does not prove optimality when the lower bound is unreachable.
With more time we would add an exact branch-and-bound over the cycle, using the metric-TSP bound, to certify optimality, and Or-opt moves with longer segments.
<!-- ===== Lines for the "Use of generative AI" section ===== -->

Road Trip: we used Claude (Anthropic) to help design, implement and benchmark the agent and to draft this section. We reviewed the code and can explain every algorithm in it.