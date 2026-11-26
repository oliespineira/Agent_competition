"""Battleship agent (moving ships).

Course sessions used are cited in brackets, e.g. [Session 17].

Problem formulation [Sessions 2 and 3]
----------------------------
Partially observable, stochastic, sequential environment. The hidden state is where
each enemy ship is. Our actions are: optionally move one of our ships, then shoot or
use sonar. Observations are HIT / MISS / SUNK and noisy sonar counts. Goal: destroy
all 17 enemy segments in as few turns as possible.

How it decides
--------------
* Belief state [Session 17]: as in Mini Battleship, the agent keeps the possible positions of
  each enemy ship that is not sunk. Instead of a set it keeps a probability for each
  possible position (top-left cell and orientation) [Session 23]. Every turn:
    1. PREDICT [Session 17 "Hidden movement"]: the rival had one turn, so every belief goes
       through the transition model of predict_ship_belief (same model, numpy
       version for speed);
    2. FILTER [Session 17 "Belief-state filtering"]: a MISS removes every position covering
       that cell (it was empty at that moment); a HIT removes it for the ships that can
       still move; a sonar reading is a Bayes update with sonar_posterior [Session 23].
  How often the rival moves is unknown, so we keep three hypotheses (it never moves,
  it moves 25% of turns, it always moves), one belief per hypothesis. Bayes' rule
  [Session 23] updates P(hypothesis | shots so far): each shot result is evidence, the
  likelihood is the probability that hypothesis gave to that result, and the scores
  are normalised across hypotheses. The beliefs are then combined by those posterior
  probabilities.
* Hunt: shoot the cell most likely to contain a ship (the highest marginal [Session 23]).
* Target (CSP [Sessions 8 and 9]): while some hits are not explained by sunk ships. Variables:
  the enemy ships not sunk yet. Domains: their positions through a hit cell
  (consistent_placements, with forward checking). Constraints: every hit covered
  exactly once, no overlap, sunk cells excluded, and no cell that was missed AFTER
  that ship's first hit (a hit ship cannot move, but before its first hit it could).
  Backtracking enumerates every consistent assignment; we shoot the cell covered by
  the most of them.
* Logical inference [Session 11]: which cells belong to each sunk ship is deduced with
  certainty when only one assignment is consistent (attributions()), and the number
  of hits still to explain is always known exactly.
* Sonar, value of information [Session 17 "Why information matters"]: used only when no
  single cell looks promising, so information is worth more than a shot.
* Defence: move one undamaged ship every turn, so the rival's old misses stop being
  true; prefer sideways moves of short ships and move away from our damaged ships.
* Clock: work is incremental; below RESERVE_S seconds it falls back to a cheap rule.
  Every answer is validated before it is returned.
"""
import random

import numpy as np

# ---------------------------------------------------------------------------------
# Switches (each optimisation cycle changes one of them; see README).
# ---------------------------------------------------------------------------------
MOVE_RATES = (0.0, 0.25, 1.0)  # hypotheses for P(rival moves a ship on its turn) [Session 23]
DEFENSIVE_MOVES = True     # move one of our undamaged ships every turn
DESIGNED_PLACEMENT = True  # random layout with no two ships touching, all able to move
USE_SONAR = True
SONAR_THRESHOLD = 0.12     # scan only while hunting and the best cell is below this
RESERVE_S = 0.6            # below this clock, play the cheap fallback
TARGET_NODE_CAP = 20000    # limit on backtracking nodes per turn (time safety)
EXTRA_SHIP_WEIGHT = 0.3    # prior: fewer ships explaining a group of hits is more likely

DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


# ---------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------
def ship_cells(row, col, orientation, length):
    if orientation == "H":
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def fits(row, col, orientation, length, size):
    if orientation == "H":
        return 0 <= row < size and 0 <= col and col + length - 1 < size
    return 0 <= col < size and 0 <= row and row + length - 1 < size


def neighbours(cell, size):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS.values() if 0 <= r + dr < size and 0 <= c + dc < size]


# ---------------------------------------------------------------------------------
# Required algorithms (tested by check_requirements): also used by the agent.
# ---------------------------------------------------------------------------------
def consistent_placements(length, board_size, blocked, required):
    """CSP domain of one ship [Session 8]: every [row, col, orientation] consistent with the
    constraints (inside the board, avoids `blocked`, covers every `required` cell).

    Forward checking [Session 9]: with required cells, values that cannot cover all of them
    are pruned before testing: they must be in one line, and only top-left cells
    within `length` of them are tried.
    """
    blocked = {tuple(c) for c in blocked}
    required = {tuple(c) for c in required}
    found = []
    for orientation in ("H", "V"):
        if required:
            rows = [r for r, _ in required]
            cols = [c for _, c in required]
            if (orientation == "H" and len(set(rows)) > 1) or (orientation == "V" and len(set(cols)) > 1):
                continue
            if orientation == "H":
                row_range, col_range = [rows[0]], range(max(cols) - length + 1, min(cols) + 1)
            else:
                row_range, col_range = range(max(rows) - length + 1, min(rows) + 1), [cols[0]]
        else:
            row_range = col_range = range(board_size)
        for row in row_range:
            for col in col_range:
                if not fits(row, col, orientation, length, board_size):
                    continue
                cells = ship_cells(row, col, orientation, length)
                if blocked.isdisjoint(cells) and required.issubset(cells):
                    found.append([int(row), int(col), orientation])
    return found


def sonar_likelihood(reading, true_count, region_cells, correct_probability):
    """Likelihood P(reading | true count) [Session 23]: the published sensor model, with
    impossible readings removed and the rest renormalised."""
    error = (1 - correct_probability) / 2
    raw = {true_count: correct_probability, true_count - 1: error, true_count + 1: error}
    valid = {k: v for k, v in raw.items() if 0 <= k <= region_cells}
    total = sum(valid.values())
    return valid.get(reading, 0.0) / total if total > 0 else 0.0


def sonar_posterior(prior, reading, region_cells, correct_probability):
    """Bayes' rule [Session 23]: posterior P(count | reading) is proportional to
    likelihood P(reading | count) times prior P(count), then normalised."""
    joint = {key: float(p) * sonar_likelihood(reading, int(key), region_cells, correct_probability)
             for key, p in prior.items()}
    total = sum(joint.values())
    if total <= 0:
        return {key: float(p) for key, p in prior.items()}
    return {key: value / total for key, value in joint.items()}


def predict_ship_belief(belief, length, orientation, move_probability):
    """PREDICT step of the belief state [Session 17 "Hidden movement"], with probabilities.

    Transition model: with move_probability the ship moves one cell, chosen uniformly
    among the directions where it stays on the board; otherwise (or if it cannot
    move) it stays. Like predict_belief in Session 17, but each destination receives a share
    of the probability instead of just being added to a set."""
    size = len(belief)
    new = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            mass = belief[row][col]
            if not mass:
                continue
            options = [(row + dr, col + dc) for dr, dc in DIRS.values()
                       if fits(row + dr, col + dc, orientation, length, size)]
            if not options:
                new[row][col] += mass
                continue
            new[row][col] += mass * (1 - move_probability)
            share = mass * move_probability / len(options)
            for r, c in options:
                new[r][c] += share
    return new


# ---------------------------------------------------------------------------------
# Vectorised versions of the same models, for speed inside the game.
# ---------------------------------------------------------------------------------
_MODEL_CACHE = {}


def _model(length, orientation, size):
    """Valid top-left mask and the transition structure of predict_ship_belief."""
    key = (length, orientation, size)
    if key not in _MODEL_CACHE:
        valid = np.zeros((size, size), dtype=bool)
        for r in range(size):
            for c in range(size):
                valid[r, c] = fits(r, c, orientation, length, size)
        moves = {}
        options = np.zeros((size, size))
        for d, (dr, dc) in DIRS.items():
            ok = np.zeros((size, size), dtype=bool)
            for r in range(size):
                for c in range(size):
                    ok[r, c] = fits(r + dr, c + dc, orientation, length, size)
            moves[d] = ok
            options += ok
        share = np.divide(1.0, options, out=np.zeros_like(options), where=options > 0)
        _MODEL_CACHE[key] = (valid, moves, share, options > 0)
    return _MODEL_CACHE[key]


for _length in (2, 3, 4, 5):          # precompute at import time (not on the clock)
    for _o in ("H", "V"):
        _model(_length, _o, 10)


def _shift(array, dr, dc):
    """out[r + dr, c + dc] = array[r, c] (mass leaving the board is dropped; never happens
    here because the move masks only allow on-board destinations)."""
    out = np.zeros_like(array)
    size = array.shape[0]
    rs, rd = (slice(0, size - dr), slice(dr, size)) if dr >= 0 else (slice(-dr, size), slice(0, size + dr))
    cs, cd = (slice(0, size - dc), slice(dc, size)) if dc >= 0 else (slice(-dc, size), slice(0, size + dc))
    out[rd, cd] = array[rs, cs]
    return out


def predict_np(grid, length, orientation, p):
    """Same transition model as predict_ship_belief, on a numpy grid."""
    if p <= 0:
        return grid
    _, moves, share, can_move = _model(length, orientation, grid.shape[0])
    new = np.where(can_move, grid * (1 - p), grid)
    for d, (dr, dc) in DIRS.items():
        new += _shift(grid * moves[d] * share * p, dr, dc)
    return new


def occupancy(grid, length, orientation):
    """Marginal P(cell is covered by this ship) [Session 23]: add the probability of every
    position of the ship that covers the cell."""
    occ = np.zeros_like(grid)
    size = grid.shape[0]
    for k in range(length):
        if orientation == "H":
            occ[:, k:] += grid[:, :size - k]
        else:
            occ[k:, :] += grid[:size - k, :]
    return occ


def zero_covering(grid, length, orientation, cell):
    """FILTER step [Session 17]: remove every position of this ship that covers cell (they
    would have given a different observation)."""
    r, c = cell
    if orientation == "H":
        grid[r, max(0, c - length + 1):c + 1] = 0.0
    else:
        grid[max(0, r - length + 1):r + 1, c] = 0.0


def overlap_counts(length, orientation, size, region):
    """For each top-left, how many of the ship's cells fall inside the region box."""
    r0, r1, c0, c1 = region
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    if orientation == "H":
        in_rows = (rows >= r0) & (rows <= r1)
        span = np.clip(np.minimum(cols + length - 1, c1) - np.maximum(cols, c0) + 1, 0, None)
        return (in_rows * span).astype(int)
    in_cols = (cols >= c0) & (cols <= c1)
    span = np.clip(np.minimum(rows + length - 1, r1) - np.maximum(rows, r0) + 1, 0, None)
    return (in_cols * span).astype(int)


# ---------------------------------------------------------------------------------
# Belief over the enemy fleet
# ---------------------------------------------------------------------------------
class ShipBelief:
    def __init__(self, length, size, allowed):
        self.length = length
        self.grids = {}
        for o in ("H", "V"):
            self.grids[o] = allowed[(length, o)].astype(float)
        self.normalise(allowed)

    def mass(self):
        return float(self.grids["H"].sum() + self.grids["V"].sum())

    def normalise(self, allowed, recent_misses=()):
        total = self.mass()
        if not np.isfinite(total) or total < 1e-12:
            # Empty belief = contradiction [Session 17 "Filtering and contradictions"]: the
            # rival moved more than this hypothesis assumed. Re-check assumptions:
            # restart from every allowed position, keeping only the freshest misses.
            for o in ("H", "V"):
                self.grids[o] = allowed[(self.length, o)].astype(float)
                for cell in recent_misses:
                    zero_covering(self.grids[o], self.length, o, cell)
            total = self.mass()
            if total < 1e-12:
                for o in ("H", "V"):
                    self.grids[o] = allowed[(self.length, o)].astype(float)
                total = max(self.mass(), 1e-12)
        for o in ("H", "V"):
            self.grids[o] /= total

    def occupancy(self):
        return occupancy(self.grids["H"], self.length, "H") + occupancy(self.grids["V"], self.length, "V")


class Tracker:
    """The belief state [Session 17] under ONE hypothesis about how often the rival moves,
    updated incrementally from the observations."""

    def __init__(self, size, lengths, my_parity, move_rate):
        self.size = size
        self.move_rate = move_rate
        self.log_weight = 0.0               # log P(shot results so far | this hypothesis) [Session 23]
        self.my_parity = my_parity          # parity of the turn numbers of our own actions
        self.allowed = {}
        for length in set(lengths):
            for o in ("H", "V"):
                self.allowed[(length, o)] = _model(length, o, size)[0].copy()
        self.ships = [ShipBelief(length, size, self.allowed) for length in sorted(lengths, reverse=True)]
        self.unsunk_lengths = sorted(lengths, reverse=True)
        self.belief_turn = 0                # the belief describes the board after this turn
        self.shots_seen = 0
        self.sonar_seen = 0
        self.hits = {}                      # cell -> turn of the hit
        self.sunk_events = []               # (cell, length, turn) for every SUNK
        self._attributions = None           # cache of attributions()
        self.last_miss = {}                 # cell -> turn of the latest real miss there
        self.recent_misses = []

    # --- PREDICT [Session 17] ---------------------------------------------------------------
    def advance_to(self, turn):
        """One PREDICT step per rival turn since the belief was last updated. Only one
        enemy ship moves per turn, so the move probability is split among the ships
        that can still move."""
        steps = sum(1 for t in range(self.belief_turn + 1, turn + 1) if t % 2 != self.my_parity)
        self.belief_turn = max(self.belief_turn, turn)
        if steps == 0 or self.move_rate <= 0 or not self.ships:
            return
        mobile = max(1, len(self.ships) - self.anchored_unsunk())
        p = min(1.0, self.move_rate / mobile)
        for _ in range(steps):
            for ship in self.ships:
                for o in ("H", "V"):
                    ship.grids[o] = predict_np(ship.grids[o], ship.length, o, p) * self.allowed[(ship.length, o)]
                ship.normalise(self.allowed, self.recent_misses)

    def anchored_unsunk(self):
        """Lower bound on the enemy ships that are hit but not sunk (they cannot move)."""
        return 1 if self.unexplained_hits() > 0 else 0

    def unexplained_hits(self):
        """Hits not belonging to sunk ships. Deduced with certainty [Session 11]:
        total hits minus the lengths of the sunk ships, whatever the attribution."""
        return len(self.hits) - sum(length for _, length, _ in self.sunk_events)

    def attributions(self, cap=40):
        """Which hit cells belong to each sunk ship, as a CSP [Session 8] solved by
        backtracking. Variables: the SUNK events. Domain of each: lines of `length`
        hit cells through its SUNK cell, all hit before it (the SUNK shot is always
        the ship's last hit). Constraint: no two sunk ships share a cell. If only one
        assignment survives, the attribution is deduced with certainty [Session 11]."""
        if self._attributions is not None:
            return self._attributions
        events = sorted(self.sunk_events, key=lambda e: e[2])
        domains = []
        for cell, length, turn in events:
            options = []
            for r, c, o in consistent_placements(length, self.size, [], [cell]):
                cells = ship_cells(r, c, o, length)
                if all(x in self.hits and self.hits[x] <= turn for x in cells):
                    options.append(frozenset(cells))
            domains.append(options)
        found = []

        def search(i, used):
            if len(found) >= cap:
                return
            if i == len(domains):
                found.append(used)
                return
            for cells in domains[i]:
                if used.isdisjoint(cells):
                    search(i + 1, used | cells)

        search(0, frozenset())
        if not found:  # should not happen; fall back to the SUNK cells alone
            found = [frozenset(cell for cell, _, _ in events)]
        self._attributions = found
        return found

    # --- FILTER [Session 17] ---------------------------------------------------------------
    def apply_shot(self, entry):
        cell = tuple(entry["target"])
        turn = entry["turn"]
        if entry.get("already_damaged"):
            return                                  # no information
        if self.unexplained_hits() == 0 and self.ships:
            # Bayes over hypotheses [Session 23]: multiply in the likelihood this hypothesis
            # gave to the observed result (logs avoid underflow; clipped so a single
            # surprise cannot rule a hypothesis out for good).
            p_hit = min(0.98, max(0.02, float(self.hunt_map()[cell])))
            self.log_weight += np.log(p_hit if entry["result"] != "MISS" else 1 - p_hit)
        if entry["result"] == "MISS":
            self.last_miss[cell] = turn
            self.recent_misses = (self.recent_misses + [cell])[-3:]
            for ship in self.ships:
                for o in ("H", "V"):
                    zero_covering(ship.grids[o], ship.length, o, cell)
                ship.normalise(self.allowed, self.recent_misses)
            return
        self.hits[cell] = turn
        for key in self.allowed:
            zero_covering(self.allowed[key], key[0], key[1], cell)
        for ship in self.ships:
            for o in ("H", "V"):
                zero_covering(ship.grids[o], ship.length, o, cell)
            ship.normalise(self.allowed, self.recent_misses)
        if entry["result"] == "SUNK":
            self.sink(cell, int(entry["sunk_length"]))

    def sink(self, cell, length):
        self.sunk_events.append((cell, length, self.hits[cell]))
        self._attributions = None
        if length in self.unsunk_lengths:
            self.unsunk_lengths.remove(length)
        for i, ship in enumerate(self.ships):
            if ship.length == length:
                del self.ships[i]
                break

    def apply_sonar(self, entry, correct):
        """Bayes update of the belief with a sonar reading [Session 23].

        1. Prior over the true count in the 3x3: for each ship, P(k of its cells are in
           the square); adding the ships together is marginalisation, summing over
           every combination [Sessions 23 and 25] (np.convolve does exactly that sum).
        2. Posterior over the count with sonar_posterior.
        3. Each position of each ship is reweighted by how compatible it is with the
           posterior count, then normalised."""
        if not self.ships or self.unexplained_hits() > 0:
            return                                  # only modelled while every live ship is mobile
        r, c = entry["center"]
        region = (max(0, r - 1), min(self.size - 1, r + 1), max(0, c - 1), min(self.size - 1, c + 1))
        cells_in_region = (region[1] - region[0] + 1) * (region[3] - region[2] + 1)
        known = sum(1 for (hr, hc) in self.hits if region[0] <= hr <= region[1] and region[2] <= hc <= region[3])
        overlaps, dists = [], []
        for ship in self.ships:
            ov = {o: overlap_counts(ship.length, o, self.size, region) for o in ("H", "V")}
            dist = np.zeros(ship.length + 1)
            for o in ("H", "V"):
                np.add.at(dist, ov[o].ravel(), ship.grids[o].ravel())
            overlaps.append(ov)
            dists.append(dist)
        total = np.array([1.0])
        for dist in dists:
            total = np.convolve(total, dist)
        prior = {str(n + known): float(p) for n, p in enumerate(total) if p > 0 and n + known <= cells_in_region}
        if not prior:
            return
        posterior = sonar_posterior(prior, int(entry["reading"]), cells_in_region, correct)
        for i, ship in enumerate(self.ships):
            rest = np.array([1.0])
            for j, dist in enumerate(dists):
                if j != i:
                    rest = np.convolve(rest, dist)
            factor = np.zeros(ship.length + 1)
            for k in range(ship.length + 1):
                for key, post in posterior.items():
                    n = int(key) - known - k
                    if 0 <= n < len(rest) and prior[key] > 0:
                        factor[k] += post * rest[n] / prior[key]
            for o in ("H", "V"):
                ship.grids[o] = ship.grids[o] * factor[overlaps[i][o]]
            ship.normalise(self.allowed, self.recent_misses)

    def ingest(self, observation):
        """Process only the new observations, in turn order: PREDICT, then FILTER."""
        shots = observation["my_shots"][self.shots_seen:]
        sonar = observation["my_sonar"][self.sonar_seen:]
        self.shots_seen += len(shots)
        self.sonar_seen += len(sonar)
        entries = sorted([(e["turn"], "shot", e) for e in shots] + [(e["turn"], "sonar", e) for e in sonar],
                         key=lambda item: item[0])
        correct = observation.get("sonar_correct_probability", 0.8)
        for turn, kind, entry in entries:
            self.advance_to(turn - 1)
            if kind == "shot":
                self.apply_shot(entry)
            else:
                self.apply_sonar(entry, correct)
            self.belief_turn = turn
        self.advance_to(observation["turn"])

    # --- decisions ----------------------------------------------------------------
    def hunt_map(self):
        occ = np.zeros((self.size, self.size))
        for ship in self.ships:
            occ += ship.occupancy()
        return occ

    def target_map(self):
        """Target mode as a CSP [Sessions 8 and 9]: count, for each cell, the consistent
        assignments (over every possible sunk-ship attribution) that put a ship
        there. Returns None in hunt mode."""
        if self.unexplained_hits() <= 0:
            return None
        weights = {}
        budget = {"nodes": 0}                       # shared by every attribution
        for resolved in self.attributions():
            unresolved = [cell for cell in self.hits if cell not in resolved]
            if unresolved and budget["nodes"] < TARGET_NODE_CAP:
                self._target_weights(unresolved, resolved, weights, budget)
        if not weights:
            # Inconsistent evidence: shoot around the hits of the first attribution.
            resolved = self.attributions()[0]
            for cell in self.hits:
                if cell not in resolved:
                    for x in neighbours(cell, self.size):
                        if x not in self.hits:
                            weights[x] = weights.get(x, 0.0) + 1.0
        return weights

    def _target_weights(self, unresolved, resolved, weights, state):
        """Backtracking search [Session 8]: pick the first uncovered hit (variable), try each
        consistent ship position through it (domain, pruned by forward checking [Session 9]),
        recurse, and undo the choice when returning."""
        unresolved = sorted(unresolved, key=lambda x: (self.hits[x], x))
        hit_set = set(unresolved)
        blocked = sorted(resolved)
        lengths = sorted(set(self.unsunk_lengths), reverse=True)
        counts = {L: self.unsunk_lengths.count(L) for L in lengths}
        domains = {}
        for cell in unresolved:
            for L in lengths:
                options = []
                for r, c, o in consistent_placements(L, self.size, blocked, [cell]):
                    cells = ship_cells(r, c, o, L)
                    hit_cells = [x for x in cells if x in hit_set]
                    if len(hit_cells) == L:
                        continue                    # all hit would mean sunk
                    first = min(self.hits[x] for x in hit_cells)
                    if any(self.last_miss.get(x, -1) > first for x in cells if x not in hit_set):
                        continue                    # missed there after the ship was anchored
                    options.append((frozenset(cells), frozenset(hit_cells)))
                domains[(cell, L)] = options

        def search(covered, used, chosen):
            state["nodes"] += 1
            if state["nodes"] > TARGET_NODE_CAP:
                return
            pending = next((x for x in unresolved if x not in covered), None)
            if pending is None:
                w = EXTRA_SHIP_WEIGHT ** len(chosen)
                for cells in chosen:
                    for x in cells:
                        if x not in hit_set:
                            weights[x] = weights.get(x, 0.0) + w
                return
            for L in lengths:
                if counts[L] == 0:
                    continue
                for cells, hit_cells in domains[(pending, L)]:
                    if not cells.isdisjoint(used) or not hit_cells.isdisjoint(covered):
                        continue
                    counts[L] -= 1
                    chosen.append(cells)
                    search(covered | hit_cells, used | cells, chosen)
                    chosen.pop()
                    counts[L] += 1

        search(frozenset(), frozenset(), [])


# ---------------------------------------------------------------------------------
# Game state kept between calls (one process per game)
# ---------------------------------------------------------------------------------
STATE = {"trackers": None, "last_key": None, "last_answer": None}


def _argmax_cell(scores, exclude):
    best, best_value = None, -1.0
    for cell, value in scores.items():
        if cell in exclude:
            continue
        value = value + random.random() * 1e-9
        if value > best_value:
            best, best_value = cell, value
    return best


def _hunt_shot(trackers, exclude):
    """Combine the hypotheses [Session 23]: normalise their scores into posterior
    probabilities and weight each belief's map by them; shoot the best cell."""
    best = max(t.log_weight for t in trackers)
    weights = [np.exp(t.log_weight - best) for t in trackers]
    total = sum(weights)
    occ = sum(w / total * t.hunt_map() for w, t in zip(weights, trackers))
    tracker = trackers[0]
    scores = {(r, c): float(occ[r, c]) for r in range(tracker.size) for c in range(tracker.size)}
    return _argmax_cell(scores, exclude), occ


def _sonar_center(tracker, occ):
    """Scan where the 3x3 square holds the most expected ship cells."""
    size = tracker.size
    best, best_value = None, -1.0
    for r in range(size):
        for c in range(size):
            value = occ[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum()
            if value > best_value:
                best, best_value = (r, c), value
    return best


def _damaged(observation):
    cells = set()
    for shot in observation["my_shots"]:
        if shot["result"] in ("HIT", "SUNK") or shot.get("already_damaged"):
            cells.add(tuple(shot["target"]))
    return cells


def _cheap_shot(observation):
    """Fallback without the belief: around unexplained hits, else a random parity cell."""
    size = observation["board_size"]
    damaged = _damaged(observation)
    sunk_cells = 0
    for length in observation["enemy_ships_sunk"]:
        sunk_cells += length
    hits = [tuple(s["target"]) for s in observation["my_shots"] if s["result"] in ("HIT", "SUNK")]
    if len(hits) > sunk_cells:
        around = [x for cell in hits for x in neighbours(cell, size) if x not in damaged]
        if around:
            return random.choice(around)
    recent = {tuple(s["target"]) for s in observation["my_shots"][-20:]}
    cells = [(r, c) for r in range(size) for c in range(size) if (r + c) % 2 == 0 and (r, c) not in damaged | recent]
    if not cells:
        cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in damaged] or [(0, 0)]
    return random.choice(cells)


# ---------------------------------------------------------------------------------
# Our own fleet
# ---------------------------------------------------------------------------------
def _designed_placement(observation):
    size = observation["board_size"]
    fleet = observation["fleet"]
    for _ in range(2000):
        occupied, halo, placements = set(), set(), []
        ok = True
        for item in fleet:
            length = item["length"]
            options = []
            for o in ("H", "V"):
                for r in range(size):
                    for c in range(size):
                        if not fits(r, c, o, length, size):
                            continue
                        cells = ship_cells(r, c, o, length)
                        if any(x in occupied or x in halo for x in cells):
                            continue
                        options.append((r, c, o, cells))
            if not options:
                ok = False
                break
            r, c, o, cells = random.choice(options)
            placements.append({"ship": item["ship"], "row": int(r), "col": int(c), "orientation": o})
            occupied.update(cells)
            for x in cells:
                halo.update(neighbours(x, size))
        if ok:
            return placements
    return None


def _choose_move(observation):
    moves = observation["legal_moves"]
    if not moves:
        return None
    size = observation["board_size"]
    ships = {s["ship"]: s for s in observation["my_ships"]}
    danger = set()
    for s in observation["my_ships"]:
        if s["damaged_cells"] and not s["sunk"]:
            for cell in s["damaged_cells"]:
                danger.add(tuple(cell))
                danger.update(neighbours(tuple(cell), size))
    best, best_score = None, None
    for move in moves:
        ship = ships[move["ship"]]
        dr, dc = DIRS[move["direction"]]
        now = [tuple(x) for x in ship["cells"]]
        after = [(r + dr, c + dc) for r, c in now]
        score = 3.0 * (sum(x in danger for x in now) - sum(x in danger for x in after))
        score -= 2.0 * sum(x in danger for x in after)
        sideways = (ship["orientation"] == "H") == (move["direction"] in ("N", "S"))
        score += 1.0 if sideways else 0.0
        score += 0.3 * (6 - ship["length"])
        score += random.random()
        if best_score is None or score > best_score:
            best, best_score = move, score
    return {"ship": best["ship"], "direction": best["direction"]}


# ---------------------------------------------------------------------------------
# Answer validation (mirrors the judge's validator, so we never lose on format)
# ---------------------------------------------------------------------------------
def _valid_cell(cell, size):
    return (isinstance(cell, list) and len(cell) == 2 and all(type(v) is int for v in cell)
            and all(0 <= v < size for v in cell))


def _safe(answer, observation):
    size = observation["board_size"]
    move = answer.get("move")
    if move is not None and move not in observation["legal_moves"]:
        move = None
    if "sonar" in answer:
        if observation["sonar_remaining"] > 0 and _valid_cell(answer["sonar"], size):
            return {"move": move, "sonar": answer["sonar"]}
        answer = {"shot": list(_cheap_shot(observation))}
    shot = answer.get("shot")
    if not _valid_cell(shot, size):
        shot = [int(v) for v in _cheap_shot(observation)]
    return {"move": move, "shot": shot}


# ---------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------
def act(observation):
    if observation["phase"] == "placement":
        STATE.update(trackers=None, last_key=None, last_answer=None)
        if DESIGNED_PLACEMENT:
            try:
                return {"placements": _designed_placement(observation)}
            except Exception:
                pass
        return {"placements": None}

    # The judge may ask twice for the same state (after a rival failure): same answer.
    key = (observation["turn"], len(observation["my_shots"]), len(observation["my_sonar"]))
    if key == STATE["last_key"] and STATE["last_answer"] is not None:
        return _safe(dict(STATE["last_answer"]), observation)

    try:
        answer = _decide(observation)
    except Exception:
        answer = {"shot": list(_cheap_shot(observation))}
    answer = _safe(answer, observation)
    STATE["last_key"], STATE["last_answer"] = key, answer
    return answer


def _decide(observation):
    size = observation["board_size"]
    clock = observation.get("time_remaining_s")
    move = _choose_move(observation) if DEFENSIVE_MOVES else None
    if clock is not None and clock < RESERVE_S:
        return {"move": move, "shot": [int(v) for v in _cheap_shot(observation)]}

    trackers = STATE["trackers"]
    if trackers is None:
        lengths = [5, 4, 3, 3, 2]
        trackers = [Tracker(size, lengths, (observation["turn"] + 1) % 2, q) for q in MOVE_RATES]
        STATE["trackers"] = trackers
    for tracker in trackers:
        tracker.ingest(observation)
    tracker = trackers[0]  # hits, misses and sunk ships are the same in every filter

    exclude = set(tracker.hits)  # never shoot a damaged segment again
    weights = tracker.target_map()  # target mode (CSP) if some hits are unexplained
    if weights:
        cell = _argmax_cell(weights, exclude)
        if cell is not None:
            return {"move": move, "shot": [int(cell[0]), int(cell[1])]}

    cell, occ = _hunt_shot(trackers, exclude)
    # Value of information [Session 17]: when no cell is promising, a scan is worth more.
    if (USE_SONAR and observation["sonar_remaining"] > 0 and tracker.ships
            and float(occ.max()) < SONAR_THRESHOLD):
        center = _sonar_center(tracker, occ)
        return {"move": move, "sonar": [int(center[0]), int(center[1])]}
    if cell is None:
        cell = _cheap_shot(observation)
    return {"move": move, "shot": [int(cell[0]), int(cell[1])]}
