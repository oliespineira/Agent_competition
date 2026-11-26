"""European Road Trip agent.

Strategy
--------
The map is a weighted undirected graph (105 cities, 262 roads). A route is a simple cycle
through Madrid that contains every required city. We split the problem in two layers:

1. ORDER layer: the visiting order of the 25 required cities. Dijkstra from every required
   city gives the shortest-path distance matrix D. For any order, sum(D) is a LOWER BOUND
   on the real length of any route that visits the cities in that order (rerouting a leg can
   only make it longer).
2. PATH layer ("decoder"): turns an order into a real simple cycle. Each leg first takes its
   shortest path. Legs that collide (two legs share a city, or a leg crosses a required city)
   are repaired in three escalating levels: reroute the colliding legs, also release their
   neighbours, and finally negotiated congestion routing (PathFinder). An order that stays
   infeasible gets a penalised cost, so local search can still move towards feasibility.
   A final pass re-optimises every leg against the rest of the cycle.

Search: hill climbing on real km over 2-opt + Or-opt moves, sorted by lower bound so that
moves that cannot improve are never decoded; iterated local search with double-bridge kicks;
restarts from random orders or a randomised cheapest insertion on the real graph when a run
stalls. Every improvement is validated locally and submitted at once; the first valid route
goes out within milliseconds.

Required functions (ucs, astar, local_search) are at the bottom and share the same cores.
"""
import heapq
import math
import random
import time

EARTH_RADIUS_KM = 6371.0  # slightly below the judge's 6371.0088, so h(n) stays admissible
INF = float("inf")
PENALTY = 20000  # km added per city shared by two legs (infeasible orders only)


# =====================================================================================
# Graph helpers
# =====================================================================================

def _build_graph(cities, roads):
    """Index the cities 0..n-1 and build adjacency lists [(neighbour, km), ...]."""
    names = sorted(cities)
    index = {c: i for i, c in enumerate(names)}
    adj = [[] for _ in names]
    for road in roads:
        a, b, km = index[road["from"]], index[road["to"]], road["km"]
        adj[a].append((b, km))
        if road.get("bidirectional", True):
            adj[b].append((a, km))
    # keep only the shortest road if the data ever had parallel roads
    for i, lst in enumerate(adj):
        best = {}
        for b, km in lst:
            if b not in best or km < best[b]:
                best[b] = km
        adj[i] = sorted(best.items())
    return names, index, adj


def _dijkstra_all(adj, source, blocked=None):
    """Single-source Dijkstra. Returns (dist list, parent list). blocked: set of nodes to avoid."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] and (blocked is None or v not in blocked):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, parent


def _path_from_parents(parent, source, goal):
    path = [goal]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _dijkstra_pair(adj, source, goal, blocked):
    """Shortest path source->goal that never enters a node in blocked (goal is always allowed).
    Returns (km, path) or (INF, None)."""
    dist = {source: 0}
    parent = {source: -1}
    heap = [(0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        if u == goal:
            path = [goal]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return d, path
        done.add(u)
        for v, w in adj[u]:
            if v in done or (v in blocked and v != goal):
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return INF, None


def _dijkstra_penalised(adj, source, goal, forbidden, occ, hist, pres):
    """Dijkstra where entering city v costs road km + hist[v] + pres * occ[v].
    Cities in forbidden (other than goal) are never entered. Returns a path or None."""
    dist = {source: 0.0}
    parent = {source: -1}
    heap = [(0.0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        if u == goal:
            path = [goal]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        done.add(u)
        for v, w in adj[u]:
            if v in done:
                continue
            if v == goal:
                nd = d + w
            elif v in forbidden:
                continue
            else:
                nd = d + w + hist.get(v, 0.0) + pres * occ.get(v, 0)
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return None


# =====================================================================================
# Order layer: tours on a distance matrix (shared by solve() and local_search())
# =====================================================================================

def _tour_len(t, D):
    return sum(D[t[i - 1]][t[i]] for i in range(len(t)))


def _two_opt_or_opt(t, D):
    """Hill climbing to a local optimum of 2-opt + Or-opt (segments of 1-3, both directions).
    t[0] stays fixed. Works in place-free style and returns (tour, length)."""
    n = len(t)
    t = list(t)
    if n < 4:
        return t, _tour_len(t, D)
    improved = True
    while improved:
        improved = False
        # 2-opt: reverse t[i..j]
        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            dab = D[a][b]
            for j in range(i + 1, n):
                c, e = t[j], t[(j + 1) % n]
                delta = D[a][c] + D[b][e] - dab - D[c][e]
                if delta < -1e-9:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    improved = True
                    a, b = t[i - 1], t[i]
                    dab = D[a][b]
        # Or-opt: move a segment of length L elsewhere, possibly reversed
        for L in (1, 2, 3):
            i = 1
            while i + L - 1 <= n - 1:
                seg = t[i:i + L]
                p, q = t[i - 1], t[(i + L) % n]
                remove_gain = D[p][seg[0]] + D[seg[-1]][q] - D[p][q]
                rest = t[:i] + t[i + L:]
                best_delta, best_pos, best_rev = -1e-9, None, False
                m = len(rest)
                for k in range(m):
                    x, y = rest[k], rest[(k + 1) % m]
                    if k == i - 1:
                        continue  # same place
                    add = D[x][seg[0]] + D[seg[-1]][y] - D[x][y]
                    add_r = D[x][seg[-1]] + D[seg[0]][y] - D[x][y]
                    if add - remove_gain < best_delta:
                        best_delta, best_pos, best_rev = add - remove_gain, k, False
                    if L > 1 and add_r - remove_gain < best_delta:
                        best_delta, best_pos, best_rev = add_r - remove_gain, k, True
                if best_pos is not None:
                    s = seg[::-1] if best_rev else seg
                    t = rest[:best_pos + 1] + s + rest[best_pos + 1:]
                    improved = True
                else:
                    i += 1
    return t, _tour_len(t, D)


def _double_bridge(t, rng):
    """Classic ILS kick: cut the tour in 4 parts A B C D and reconnect as A C B D."""
    n = len(t)
    if n < 8:
        t = list(t)
        i, j = sorted(rng.sample(range(1, n), 2))
        t[i], t[j] = t[j], t[i]
        return t
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return t[:a] + t[b:c] + t[a:b] + t[c:]


def _nearest_neighbour(D, start, nodes):
    left = set(nodes) - {start}
    t = [start]
    while left:
        nxt = min(left, key=lambda v: D[t[-1]][v])
        t.append(nxt)
        left.remove(nxt)
    return t


def _ils(t, D, deadline, rng):
    """Iterated local search on a distance matrix until the deadline."""
    best, best_len = _two_opt_or_opt(t, D)
    cur, cur_len = best, best_len
    while time.monotonic() < deadline:
        cand, cand_len = _two_opt_or_opt(_double_bridge(cur, rng), D)
        if cand_len < cur_len - 1e-9 or rng.random() < 0.05:
            cur, cur_len = cand, cand_len
        if cand_len < best_len - 1e-9:
            best, best_len = cand, cand_len
            cur, cur_len = cand, cand_len
    return best, best_len


# =====================================================================================
# Path layer: turning an order of required cities into a real simple cycle
# =====================================================================================

class RoadTripSolver:
    def __init__(self, problem, deadline, submit, seed=20261203):
        self.deadline = deadline
        self.submit_fn = submit
        self.rng = random.Random(seed)
        self.names, self.index, self.adj = _build_graph(problem["cities"], problem["roads"])
        self.start = self.index[problem["start"]]
        req = [self.index[c] for c in dict.fromkeys(problem["required"])]
        if self.start not in req:
            req.insert(0, self.start)
        req.remove(self.start)
        self.req = [self.start] + req           # local id k -> graph node
        self.k = len(self.req)
        self.req_set = set(self.req)
        self.adj_w = [dict(lst) for lst in self.adj]

        # Shortest paths between every pair of required cities (UCS/Dijkstra from each one)
        self.D = [[0] * self.k for _ in range(self.k)]
        self.P = [[None] * self.k for _ in range(self.k)]
        for a in range(self.k):
            dist, parent = _dijkstra_all(self.adj, self.req[a])
            for b in range(self.k):
                if a != b:
                    self.D[a][b] = dist[self.req[b]]
                    self.P[a][b] = _path_from_parents(parent, self.req[a], self.req[b])
        self.best_km = INF
        self.best_route = None
        self.best_lb = INF          # best metric (lower-bound) tour length seen
        self.cache = {}

    # ---------------------------------------------------------------- submission
    def _offer(self, cycle):
        """cycle: graph nodes, closed (first == last == start). Validate, then submit if better."""
        km = self._route_km(cycle)
        if km is None or km >= self.best_km:
            return False
        self.best_km = km
        self.best_route = cycle
        if time.monotonic() < self.deadline + 0.3:
            self.submit_fn([self.names[v] for v in cycle])
        return True

    def _route_km(self, cycle):
        """Independent re-check of every rule the judge enforces. None if invalid."""
        if len(cycle) < 3 or cycle[0] != self.start or cycle[-1] != self.start:
            return None
        body = cycle[:-1]
        if len(set(body)) != len(body) or not self.req_set.issubset(body):
            return None
        km = 0
        for a, b in zip(cycle, cycle[1:]):
            w = self.adj_w[a].get(b)
            if w is None:
                return None
            km += w
        return km

    # ---------------------------------------------------------------- decoder
    def decode(self, order):
        """order: local ids, order[0] == 0 (Madrid). Returns (km, legs) or (INF, None).
        legs[i] is the node path from order[i] to order[i+1] (cyclically)."""
        key = tuple(order)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res = self._decode(order)
        if len(self.cache) > 200000:
            self.cache.clear()
        self.cache[key] = res
        return res

    def _decode(self, order):
        k = len(order)
        P, D = self.P, self.D
        legs = [P[order[i]][order[(i + 1) % k]] for i in range(k)]
        # Who uses which intermediate node?
        users = {}
        for i, path in enumerate(legs):
            for v in path[1:-1]:
                users.setdefault(v, []).append(i)
        bad = set()
        for v, lst in users.items():
            if len(lst) > 1 or v in self.req_set:
                bad.update(lst)
        if not bad:
            return sum(D[order[i]][order[(i + 1) % k]] for i in range(k)), legs

        fixed_nodes = set(self.req_set)
        for i, path in enumerate(legs):
            if i not in bad:
                fixed_nodes.update(path[1:-1])
        bad = sorted(bad)
        attempts = [bad, sorted(bad, key=lambda i: -len(legs[i])), bad[::-1],
                    sorted(bad, key=lambda i: D[order[i]][order[(i + 1) % k]])]
        if len(bad) <= 4:
            # small conflict sets: try every priority order exhaustively (at most 24)
            attempts = list(_permutations(bad))
        else:
            for _ in range(4):
                p = bad[:]
                self.rng.shuffle(p)
                attempts.append(p)
        best_km, best_legs = INF, None
        seen = set()
        for prio in attempts:
            tp = tuple(prio)
            if tp in seen:
                continue
            seen.add(tp)
            res = self._reroute(order, legs, prio, fixed_nodes)
            if res is not None and res[0] < best_km:
                best_km, best_legs = res
        if best_legs is None:
            # level 2: widen the conflict set with the neighbouring legs and retry
            wide = set(bad)
            for i in bad:
                wide.update(((i - 1) % k, (i + 1) % k))
            fixed2 = set(self.req_set)
            for i, path in enumerate(legs):
                if i not in wide:
                    fixed2.update(path[1:-1])
            wide = sorted(wide)
            for r in range(6):
                prio = wide[:] if r == 0 else self.rng.sample(wide, len(wide))
                res = self._reroute(order, legs, prio, fixed2)
                if res is not None and res[0] < best_km:
                    best_km, best_legs = res
        if best_legs is None:
            # level 3: negotiated congestion routing over all legs
            neg, over = self._negotiate(order, legs)
            if neg is None:
                return INF, None
            if over:
                # infeasible order: finite penalised cost gives local search a gradient
                return self._legs_km(neg) + PENALTY * over, None
            best_legs = neg
        best_legs = self._polish_legs(order, best_legs)
        return self._legs_km(best_legs), best_legs

    def _reroute(self, order, legs, prio, fixed_nodes):
        k = len(order)
        taken = set(fixed_nodes)
        new = list(legs)
        for i in prio:
            a, b = self.req[order[i]], self.req[order[(i + 1) % k]]
            km, path = _dijkstra_pair(self.adj, a, b, taken)
            if path is None:
                return None
            new[i] = path
            taken.update(path[1:-1])
        return self._legs_km(new), new

    def _negotiate(self, order, legs, rounds=25):
        """PathFinder-style routing: every leg is rerouted repeatedly; a city shared by several
        legs gets more expensive each round (present + history penalty) until no city is shared.
        Required cities other than a leg's own ends are never entered."""
        k = len(order)
        legs = list(legs)
        occ = {}
        for p in legs:
            for v in p[1:-1]:
                occ[v] = occ.get(v, 0) + 1
        hist = {}
        pres = 100.0
        seq = list(range(k))
        best_state = None
        for _ in range(rounds):
            for i in seq:
                a, b = self.req[order[i]], self.req[order[(i + 1) % k]]
                for v in legs[i][1:-1]:
                    occ[v] -= 1
                path = _dijkstra_penalised(self.adj, a, b, self.req_set, occ, hist, pres)
                if path is None:
                    return None, 0
                legs[i] = path
                for v in path[1:-1]:
                    occ[v] = occ.get(v, 0) + 1
            shared = [v for v, c in occ.items() if c > 1]
            if not shared:
                return legs, 0
            over = sum(occ[v] - 1 for v in shared)
            km = self._legs_km(legs)
            if best_state is None or (over, km) < best_state[:2]:
                best_state = (over, km, list(legs))
            for v in shared:
                hist[v] = hist.get(v, 0.0) + 150.0 * (occ[v] - 1)
            pres *= 1.3
            self.rng.shuffle(seq)
            if time.monotonic() > self.deadline:
                break
        if best_state is None:
            return None, 0
        return best_state[2], best_state[0]

    def _legs_km(self, legs):
        w = self.adj_w
        return sum(w[p[j]][p[j + 1]] for p in legs for j in range(len(p) - 1))

    def _polish_legs(self, order, legs):
        """Re-optimise each leg against all other nodes of the cycle until nothing improves."""
        legs = list(legs)
        k = len(legs)
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            used = set(self.req_set)
            for p in legs:
                used.update(p[1:-1])
            for i in range(k):
                p = legs[i]
                inner = set(p[1:-1])
                old = sum(self.adj_w[p[j]][p[j + 1]] for j in range(len(p) - 1))
                km, path = _dijkstra_pair(self.adj, p[0], p[-1], used - inner)
                if path is not None and km < old:
                    used -= inner
                    used.update(path[1:-1])
                    legs[i] = path
                    improved = True
        return legs

    def cycle_of(self, legs):
        cyc = [legs[0][0]]
        for p in legs:
            cyc.extend(p[1:])
        return cyc

    # ---------------------------------------------------------------- order search
    def lb(self, order):
        D = self.D
        return sum(D[order[i - 1]][order[i]] for i in range(len(order)))

    def neighbours(self, t):
        """All 2-opt and Or-opt neighbours of t with their lower bound, cheapest first."""
        D, n = self.D, len(t)
        base = self.lb(t)
        out = []
        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            for j in range(i + 1, n):
                c, e = t[j], t[(j + 1) % n]
                delta = D[a][c] + D[b][e] - D[a][b] - D[c][e]
                out.append((base + delta, 0, i, j, False))
        for L in (1, 2, 3):
            for i in range(1, n - L + 1):
                seg0, seg1 = t[i], t[i + L - 1]
                p, q = t[i - 1], t[(i + L) % n]
                gain = D[p][seg0] + D[seg1][q] - D[p][q]
                for k in range(n):
                    if i - 1 <= k <= i + L - 1:
                        continue
                    x, y = t[k], t[(k + 1) % n]
                    if i <= (k + 1) % n <= i + L - 1:
                        continue
                    add = D[x][seg0] + D[seg1][y] - D[x][y]
                    out.append((base + add - gain, L, i, k, False))
                    if L > 1:
                        add_r = D[x][seg1] + D[seg0][y] - D[x][y]
                        out.append((base + add_r - gain, L, i, k, True))
        out.sort()
        return out

    @staticmethod
    def apply(t, move):
        _, L, i, j, rev = move
        if L == 0:
            return t[:i] + t[i:j + 1][::-1] + t[j + 1:]
        seg = t[i:i + L]
        if rev:
            seg = seg[::-1]
        x = t[j]
        rest = t[:i] + t[i + L:]
        pos = rest.index(x)
        return rest[:pos + 1] + seg + rest[pos + 1:]

    def descend(self, t, km):
        """Hill climbing on REAL km. Lower bound prunes every move that cannot improve."""
        while time.monotonic() < self.deadline:
            moved = False
            for move in self.neighbours(t):
                if move[0] >= km:
                    break          # sorted: no remaining move can beat the current route
                if time.monotonic() >= self.deadline:
                    return t, km
                cand = self.apply(t, move)
                ckm, legs = self.decode(cand)
                if ckm < km:
                    t, km = cand, ckm
                    if legs is not None:
                        self._offer(self.cycle_of(legs))
                    moved = True
                    break
            if not moved:
                break
        return t, km

    def run(self):
        k = self.k
        if k == 1:
            return
        # 1) instant route: nearest-neighbour order, decoded
        t0 = _nearest_neighbour(self.D, 0, range(k))
        km0, legs0 = self.decode(t0)
        if legs0 is not None:
            self._offer(self.cycle_of(legs0))
        # 2) lower-bound-optimal order (metric TSP on D), decoded
        t, lbv = _two_opt_or_opt(t0, self.D)
        self.best_lb = lbv
        km, legs = self.decode(t)
        if legs is not None:
            self._offer(self.cycle_of(legs))
        cur_t, cur_km = (t, km) if km <= km0 else (t0, km0)
        # 3) no valid route yet: build one city by city on the real graph
        if self.best_route is None:
            for attempt in range(60):
                if time.monotonic() >= self.deadline:
                    break
                ins = self.insertion(randomise=attempt > 0)
                if ins is not None:
                    ikm, ilegs = self.decode(ins)
                    if ilegs is not None:
                        self._offer(self.cycle_of(ilegs))
                    if ikm < cur_km:
                        cur_t, cur_km = ins, ikm
                    break
        cur_t, cur_km = self.descend(cur_t, cur_km)
        best_t, best_km = cur_t, cur_km
        # 4) iterated local search with restarts. Two kinds of kick, alternated:
        #    A: kick, snap to a lower-bound local optimum, decode, descend on real km
        #    B: kick and descend on real km directly (keeps hard regions feasible)
        #    When a run stalls, restart from a fresh order (random or randomised insertion),
        #    keeping the global best: the search converges in well under a second, so the
        #    rest of the clock is spent exploring other basins.
        it = 0
        last_gain = 0
        run_best_t, run_best_km = best_t, best_km
        stall_limit = 120
        while time.monotonic() < self.deadline:
            it += 1
            if it - last_gain > stall_limit:
                start_t = None
                if it % 2 and self.best_route is not None:
                    perm = [0] + self.rng.sample(range(1, k), k - 1)
                    start_t, _ = _two_opt_or_opt(perm, self.D)
                else:
                    start_t = self.insertion(randomise=True)
                if start_t is None:
                    last_gain = it
                    continue
                skm, legs = self.decode(start_t)
                if legs is not None:
                    self._offer(self.cycle_of(legs))
                if skm == INF:
                    last_gain = it
                    continue
                run_best_t, run_best_km = self.descend(start_t, skm)
                cur_t, cur_km = run_best_t, run_best_km
                last_gain = it
                continue
            base = run_best_t if it % 4 else cur_t
            kick = _double_bridge(base, self.rng)
            if it % 2 == 0 or run_best_km >= PENALTY:
                cand, cand_lb = _two_opt_or_opt(kick, self.D)
                if cand_lb < self.best_lb:
                    self.best_lb = cand_lb
                if cand_lb >= run_best_km:
                    continue          # cannot beat this run's best route: skip decoding
            else:
                cand = kick
            ckm, legs = self.decode(cand)
            if ckm == INF:
                continue
            if legs is not None:
                self._offer(self.cycle_of(legs))
            cand, ckm = self.descend(cand, ckm)
            if ckm < run_best_km:
                run_best_t, run_best_km = cand, ckm
                last_gain = it
            cur_t, cur_km = cand, ckm

    def insertion(self, randomise=False):
        """Cheapest insertion on the REAL graph: keep a simple cycle and add one required city at
        a time where it costs the fewest extra km, along paths avoiding the cycle. Returns an
        order (local ids) or None."""
        k, D, R = self.k, self.D, self.req
        others = list(range(1, k))
        if randomise:
            self.rng.shuffle(others)
        else:
            others.sort(key=lambda c: -D[0][c])   # farthest first
        first = others.pop(0)
        out = self.P[0][first]
        km_back, back = _dijkstra_pair(self.adj, R[first], R[0], (self.req_set | set(out[1:-1])) - {R[0]})
        if back is None:
            return None
        order = [0, first]
        legs = [out, back]
        while others:
            used = set()
            for p in legs:
                used.update(p)
            best = None
            pick = others if not randomise else others[:1]
            for x in pick:
                for i in range(len(order)):
                    a, b = order[i], order[(i + 1) % len(order)]
                    leg = legs[i]
                    leg_km = sum(self.adj_w[leg[j]][leg[j + 1]] for j in range(len(leg) - 1))
                    if best is not None and D[a][x] + D[x][b] - leg_km >= best[0]:
                        continue
                    blocked = (used - set(leg[1:-1])) | self.req_set
                    k1, p1 = _dijkstra_pair(self.adj, R[a], R[x], blocked - {R[x]})
                    if p1 is None:
                        continue
                    k2, p2 = _dijkstra_pair(self.adj, R[x], R[b], (blocked | set(p1)) - {R[b]})
                    if p2 is None:
                        continue
                    delta = k1 + k2 - leg_km
                    if best is None or delta < best[0]:
                        best = (delta, x, i, p1, p2)
            if best is None:
                return None
            _, x, i, p1, p2 = best
            order.insert(i + 1, x)
            legs[i:i + 1] = [p1, p2]
            others.remove(x)
        return order


def _permutations(items):
    if len(items) <= 1:
        yield list(items)
        return
    for i in range(len(items)):
        for rest in _permutations(items[:i] + items[i + 1:]):
            yield [items[i]] + rest


# =====================================================================================
# Entry point called by the judge
# =====================================================================================

def solve(problem, submit):
    t_start = time.monotonic()
    limit = float(problem.get("time_limit_s", 5))
    # keep a safety margin: the judge's clock started slightly before ours
    deadline = t_start + max(0.3, limit * 0.9 - 0.25)
    solver = None
    try:
        solver = RoadTripSolver(problem, deadline, submit)
        solver.run()
    except Exception:
        pass
    if solver is None or solver.best_route is None:
        # Safety net: should never be reached, but a valid route is worth a whole game.
        _fallback(problem, submit, t_start + limit * 0.95)


def _fallback(problem, submit, deadline):
    names, index, adj = _build_graph(problem["cities"], problem["roads"])
    start = index[problem["start"]]
    required = [index[c] for c in problem["required"] if index[c] != start]
    rng = random.Random(7)
    while time.monotonic() < deadline:
        route = [start, start]
        ok = True
        for city in rng.sample(required, len(required)):
            if city in route:
                continue
            used = set(route)
            inserted = False
            for i in rng.sample(range(len(route) - 1), len(route) - 1):
                _, there = _dijkstra_pair(adj, route[i], city, used)
                if there is None:
                    continue
                _, back = _dijkstra_pair(adj, city, route[i + 1], (used | set(there)) - {route[i + 1]})
                if back is None:
                    continue
                route = route[:i] + there + back[1:] + route[i + 2:]
                inserted = True
                break
            if not inserted:
                ok = False
                break
        if ok:
            submit([names[v] for v in route])
            return


# =====================================================================================
# Required functions (tested by check_requirements)
# =====================================================================================

def _haversine(a, b):
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def _best_first(graph, start, goal, heuristic):
    """Shared core of UCS (h = 0) and A* (h = straight-line km). Graph search with a closed set.
    A node counts as expanded when it is popped from the frontier to explore its neighbours."""
    adj = {c: {} for c in graph["cities"]}
    for road in graph["roads"]:
        a, b, km = road["from"], road["to"], road["km"]
        if km < adj[a].get(b, INF):
            adj[a][b] = km
        if road.get("bidirectional", True) and km < adj[b].get(a, INF):
            adj[b][a] = km
    g = {start: 0}
    parent = {start: None}
    frontier = [(heuristic(start), 0, start)]
    closed = set()
    expanded = 0
    while frontier:
        f, d, node = heapq.heappop(frontier)
        if node in closed or d > g[node]:
            continue
        if node == goal:
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return {"path": path[::-1], "km": d, "expanded": expanded}
        closed.add(node)
        expanded += 1
        for nxt, km in adj[node].items():
            nd = d + km
            if nxt not in closed and nd < g.get(nxt, INF):
                g[nxt] = nd
                parent[nxt] = node
                heapq.heappush(frontier, (nd + heuristic(nxt), nd, nxt))
    return {"path": [], "km": INF, "expanded": expanded}


def ucs(graph, start, goal):
    """Uniform-cost search: always expands the frontier node with the lowest g(n)."""
    return _best_first(graph, start, goal, lambda n: 0.0)


def astar(graph, start, goal):
    """A* with h(n) = Haversine distance to the goal (admissible and consistent: every road is
    at least as long as the straight line between its ends, and straight lines obey the
    triangle inequality)."""
    cities = graph["cities"]
    target = cities[goal]
    cache = {}

    def h(n):
        if n not in cache:
            cache[n] = _haversine(cities[n], target)
        return cache[n]
    return _best_first(graph, start, goal, h)


def local_search(tour, distance, time_limit_s):
    """Iterated local search: 2-opt + Or-opt hill climbing, escaping local optima with
    double-bridge kicks; accepts a kicked tour if it is not worse (plus 5% random acceptance).
    Uses at most a third of the time limit (and 1 s) because 12-25 cities converge fast."""
    tour = list(tour)
    if len(tour) < 4:
        return tour
    ids = list(range(len(tour)))
    D = [[0 if a == b else distance[tour[a]][tour[b]] for b in ids] for a in ids]
    deadline = time.monotonic() + min(1.0, max(0.05, time_limit_s / 3))
    best, _ = _ils(ids, D, deadline, random.Random(0))
    return [tour[i] for i in best]

