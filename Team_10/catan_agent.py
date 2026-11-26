"""Catan Tournament 1v1 agent (first to 7 victory points).

Course sessions used are cited in brackets, e.g. [Session 15].

Problem formulation [Sessions 2 and 3]
--------------------------------------
Catan is partially observable (we cannot see the rival's cards), multiagent,
stochastic (dice, development cards, the card stolen by the robber), sequential and
discrete. Our agent is a utility-based agent [Session 2]: before acting, it simulates
where each of its actions would lead and picks the one whose resulting position looks
best.

Two policies live in this file:
  * act()      the policy that plays. It searches the actions of the current turn and
               scores each resulting position with a hand-written weighted sum of
               features [Session 15].
  * q_policy() the reinforcement-learning policy [Session 20]: a Q-table learned
               offline (train_q_catan.py). It is the learned value required by the
               assignment; in our tests it is much weaker than act().

How it decides
--------------
* State and transition model [Session 3]: State is a copy of the position that knows
  just enough of the rules to simulate our own actions (build, buy, trade, play a
  card). apply_action() plays the role of RESULT(s, a).
* Hidden information [Session 17]: we cannot see the rival's hand, so the Tracker
  replays the public log (dice, builds, trades) to keep a belief about which cards
  they hold. Development cards we have not seen are treated as equally likely
  [Session 23].
* Evaluation [Session 15]: score() adds up our features, each multiplied by its
  weight, and subtracts the rival's, scaled down so that our own progress counts more
  than blocking. The weights were chosen by hand and kept or rejected by playing
  matches; they are not learned.
* Search over one turn [Session 6]: a turn can be several actions in a row (trade,
  then build, then buy). plan() is a local beam search: at each step it keeps only the
  BEAM_WIDTH best positions instead of all of them. Two orders of actions that reach
  the same position are searched only once, like the "reached" set of [Session 4].
* Expected utility [Sessions 2 and 23]: when the result of an action is random
  (buying a development card, stealing with the robber), we average the score over
  every possible outcome, weighted by its probability.
* Longest Road: depth-first search with backtracking over our roads [Session 4].
* Baseline: heuristic_policy() is a simple reflex agent [Session 2], a fixed list of
  condition-action rules with no search and no learned values.

Layout
  1. Board       the map: which numbers, resources, ports and neighbours each corner has
  2. State       a copyable position with the rules needed to simulate our own actions
  3. Tracker     replays the public log to estimate which cards the rival holds
  4. Evaluation  scores a position: weighted sum of features
  5. Search      beam search over the action sequences of one turn
  6. Placement   the two opening settlements and roads
  7. Required functions: state_features, action_value, q_update, heuristic_policy
  8. Q-learning policy and the learned table

Words used: "node" = a corner of a tile (where settlements go); "edge" = a side
(where roads go); "pips" = how many of the 36 two-dice results give a number (6 and 8
have 5 pips, 2 and 12 have 1); player 0 = us, player 1 = the rival.
"""
import math

# ---------------------------------------------------------------- constants
# the five resources; every hand or cost is a list of 5 numbers in this order
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
RIDX = {r: i for i, r in enumerate(RES)}
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)
# building costs: (wood, brick, sheep, wheat, ore)
COST_ROAD = (1, 1, 0, 0, 0)
COST_SETTLE = (1, 1, 1, 1, 0)
COST_CITY = (0, 0, 0, 2, 3)
COST_DEV = (0, 0, 1, 1, 1)
# dice number -> pips (ways out of 36 to roll it)
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
DEV = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
DEV_TOTAL = (14, 2, 2, 2, 5)  # how many of each development card exist in the deck
# turns at which eta() checks "can we afford it yet?"
ETA_STEPS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25)



# ---------------------------------------------------------------- 1. board
class Board:
    """Facts about the map that never change during a game."""

    def __init__(self, board):
        tiles = board["tiles"]
        self.tile_id = {tuple(t["tile"]): i for i, t in enumerate(tiles)}
        self.tile_res = [RIDX[t["resource"]] if t["resource"] else None for t in tiles]
        self.tile_num = [t["number"] if t["resource"] else None for t in tiles]
        self.tile_pips = [PIPS.get(t["number"], 0) if t["resource"] else 0 for t in tiles]
        self.tile_nodes = [tuple(t["nodes"]) for t in tiles]
        self.node_tiles, self.adj = {}, {}  # node -> tiles touching it; node -> neighbor nodes
        for i, t in enumerate(tiles):
            for n in t["nodes"]:
                self.node_tiles.setdefault(n, []).append(i)
            for a, b in t["edges"]:
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
        self.nodes = sorted(self.node_tiles)
        self.adj = {n: tuple(sorted(v)) for n, v in self.adj.items()}
        self.edges = sorted({(min(a, b), max(a, b)) for a in self.adj for b in self.adj[a]})
        self.port = {}  # node -> 0..4 (2:1 port of that resource) or 5 (3:1 port)
        for p in board["ports"]:
            kind = 5 if p["resource"] is None else RIDX[p["resource"]]
            for n in p["nodes"]:
                self.port[n] = kind
        # production of a corner: pips of each resource on the tiles around it
        self.node_prod = {n: [0] * 5 for n in self.nodes}
        for n, ts in self.node_tiles.items():
            for t in ts:
                if self.tile_res[t] is not None:
                    self.node_prod[n][self.tile_res[t]] += self.tile_pips[t]
        self.node_pips = {n: sum(v) for n, v in self.node_prod.items()}
        # quick quality of a corner: pips, plus 0.7 per different resource, plus 1 for a port (hand-chosen)
        self.spot_value = {}
        for n in self.nodes:
            diversity = sum(1 for v in self.node_prod[n] if v)
            self.spot_value[n] = self.node_pips[n] + 0.7 * diversity + (1.0 if n in self.port else 0.0)


# ---------------------------------------------------------------- 2. state
def longest_road(board, roads, blocked):
    """Number of roads in the longest path that never uses the same road twice.

    It is a depth-first search with backtracking [Session 4]: from a starting corner we
    try every way of extending the path. When a branch ends, we undo the last road so
    other branches can use it.
    `blocked` holds the corners with a rival building. As in the engine, a path cannot
    start at, enter or pass through one of them.
    """
    if not roads:
        return 0
    graph = {}  # corner -> corners connected to it by our roads
    for a, b in roads:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    best, used = 0, set()

    def walk(node, length):  # depth-first search: try every way of extending the path
        nonlocal best
        best = max(best, length)
        for nxt in graph[node]:
            edge = (node, nxt) if node < nxt else (nxt, node)
            if edge not in used and nxt not in blocked:
                used.add(edge)
                walk(nxt, length + 1)
                used.discard(edge)  # undo, so other branches may use this road

    # A longest path usually starts at a loose end or a fork (a corner without exactly 2 of our roads): try those first.
    for n, neighbors in graph.items():
        if len(neighbors) != 2 and n not in blocked:
            walk(n, 0)
    # If that did not use every road, there may be a loop. A loop has no loose end, so the best path can start
    # at a corner with exactly 2 roads: try every corner.
    if best < len(roads):
        for n in graph:
            if n not in blocked:
                walk(n, 0)
    return best


class Player:
    """One player's pieces and cards.

    hand: 5 card counts (fractions for the rival, whose cards are estimated). sett / city: corners.
    roads: edges. dev: development cards held by type (order of DEV). knights: knights played.
    road_len: longest road. ndev: number of development cards held. hidden_vp: expected hidden points.
    """
    __slots__ = ("hand", "sett", "city", "roads", "dev", "knights", "road_len", "ndev", "hidden_vp")

    def copy(self):
        p = Player()
        p.hand = self.hand[:]
        p.sett = set(self.sett)
        p.city = set(self.city)
        p.roads = set(self.roads)
        p.dev = self.dev[:]
        p.knights, p.road_len, p.ndev, p.hidden_vp = self.knights, self.road_len, self.ndev, self.hidden_vp
        return p


class State:
    """A position that can be copied and changed to try actions.

    b: Board. p: [me, rival]. robber: tile index. road_owner / army_owner: who holds the 2-point bonus.
    dev_left: cards in the deck. played_dev: we already played a card this turn. free_roads: from Road Building.
    unseen: development cards, by type, that we have not seen (deck or rival's hand).
    """
    __slots__ = ("b", "p", "robber", "road_owner", "army_owner", "dev_left", "played_dev", "free_roads", "unseen")

    def copy(self):
        s = State()
        s.b = self.b
        s.p = [self.p[0].copy(), self.p[1].copy()]
        s.robber, s.road_owner, s.army_owner = self.robber, self.road_owner, self.army_owner
        s.dev_left, s.played_dev, s.free_roads, s.unseen = self.dev_left, self.played_dev, self.free_roads, self.unseen
        return s

    # -- derived facts
    def vp(self, i):
        p = self.p[i]
        return (len(p.sett) + 2 * len(p.city) + 2 * (self.road_owner == i) + 2 * (self.army_owner == i)
                + p.dev[4] + p.hidden_vp)

    def occupied(self, n):
        return n in self.p[0].sett or n in self.p[0].city or n in self.p[1].sett or n in self.p[1].city

    def free_spot(self, n):  # distance rule: the corner and all its neighbors are empty
        return not self.occupied(n) and not any(self.occupied(m) for m in self.b.adj[n])

    def network(self, i):
        """Nodes from which player i may extend a road."""
        p, q = self.p[i], self.p[1 - i]
        nodes = set(p.sett) | p.city
        for a, b in p.roads:
            nodes.add(a)
            nodes.add(b)
        return {n for n in nodes if n not in q.sett and n not in q.city}

    def road_options(self, i):
        taken = self.p[0].roads | self.p[1].roads
        out = set()
        for n in self.network(i):
            for m in self.b.adj[n]:
                e = (n, m) if n < m else (m, n)
                if e not in taken:
                    out.add(e)
        return out

    def settle_options(self, i):
        nodes = set()
        for a, b in self.p[i].roads:
            nodes.add(a)
            nodes.add(b)
        return [n for n in nodes if self.free_spot(n)]

    def ratios(self, i):  # bank trade rate per resource: 4, or 3 / 2 with the right port
        p = self.p[i]
        ports = {self.b.port[n] for n in (p.sett | p.city) if n in self.b.port}
        base = 3 if 5 in ports else 4
        return [2 if r in ports else base for r in range(5)]

    def production(self, i, robber=True):
        """Pips per resource that player i collects (a city counts twice). robber=False ignores the robber."""
        out = [0] * 5
        p = self.p[i]
        for nodes, mult in ((p.sett, 1), (p.city, 2)):
            for n in nodes:
                for t in self.b.node_tiles[n]:
                    r = self.b.tile_res[t]
                    if r is not None and not (robber and t == self.robber):
                        out[r] += mult * self.b.tile_pips[t]
        return out

    # -- rules for the actions we simulate
    def update_longest_road(self, i):  # Longest Road needs 5 or more, and strictly more than the holder
        q = self.p[1 - i]
        n = longest_road(self.b, self.p[i].roads, q.sett | q.city)
        self.p[i].road_len = max(self.p[i].road_len, n)
        if n >= 5 and (self.road_owner is None or n > self.p[self.road_owner].road_len):
            self.road_owner = i

    def pay(self, i, cost):
        h = self.p[i].hand
        for r in range(5):
            h[r] -= cost[r]

    def can_pay(self, i, cost):
        h = self.p[i].hand
        return all(h[r] >= cost[r] - 1e-6 for r in range(5))

    def build_road(self, i, edge, free=False):
        if not free:
            self.pay(i, COST_ROAD)
        self.p[i].roads.add(edge)
        self.update_longest_road(i)

    def build_settlement(self, i, node):
        self.pay(i, COST_SETTLE)
        self.p[i].sett.add(node)
        j = 1 - i
        q = self.p[j]
        # Building on a corner with 2 rival roads can cut their road. The engine may then award Longest Road in a
        # surprising way (even below 5 roads); we copy it so we do not hand it to the rival by accident.
        if sum(1 for e in q.roads if node in e) >= 2:
            q.road_len = longest_road(self.b, q.roads, self.p[i].sett | self.p[i].city)
            if self.road_owner is None and q.road_len >= self.p[i].road_len and q.road_len < 5:
                self.road_owner = j
            elif self.road_owner == j:
                mine = self.p[i].road_len
                if q.road_len < 5:
                    self.road_owner = i if mine >= 5 else None
                elif mine > q.road_len:
                    self.road_owner = i

    def build_city(self, i, node):
        self.pay(i, COST_CITY)
        self.p[i].sett.discard(node)
        self.p[i].city.add(node)

    def play_knight(self, i):
        p = self.p[i]
        p.dev[0] -= 1
        p.knights += 1
        self.played_dev = True
        if p.knights >= 3 and (self.army_owner is None or p.knights > self.p[self.army_owner].knights):
            self.army_owner = i

    def move_robber(self, i, tile, steal):
        """The stolen card is random, so move the average card: 1/total of each resource."""
        self.robber = tile
        if steal:
            q = self.p[1 - i]
            total = sum(q.hand)
            if total > 0:
                frac = min(1.0, 1.0 / total)
                for r in range(5):
                    x = q.hand[r] * frac
                    q.hand[r] -= x
                    self.p[i].hand[r] += x


# ---------------------------------------------------------------- 3. tracker
class Tracker:
    """Keep a belief about the rival's hidden cards [Session 17], built from the public log.

    We cannot see the rival's hand, but every card they get or spend is public: dice
    rolls give resources, and every build or trade removes them. By replaying the log
    we know their hand exactly, until they discard or lose cards we cannot see. From
    then on it is an estimate.
    Our own hand does not come from here: the observation gives it to us.
    """

    def __init__(self, board):
        self.b = board
        self.seen = 0
        self.history = []  # entries already replayed
        self.hand = {"me": [0.0] * 5, "opponent": [0.0] * 5}
        self.sett = {"me": set(), "opponent": set()}
        self.city = {"me": set(), "opponent": set()}
        self.n_sett = {"me": 0, "opponent": 0}
        self.n_road = {"me": 0, "opponent": 0}
        self.robber = next((i for i, r in enumerate(board.tile_res) if r is None), 0)
        self.free = {"me": 0, "opponent": 0}
        self.played_this_turn = False
        self.bought_unseen = 0

    def _add(self, who, cards, sign=1):
        h = self.hand[who]
        for r in range(5):
            h[r] = max(0.0, h[r] + sign * cards[r])

    def update(self, log):  # replay only the new entries; start again if this is a different game
        if len(log) < self.seen:
            self.__init__(self.b)
        if log[:self.seen] != self.history:  # another game, or another player's view of it
            self.__init__(self.b)
        for e in log[self.seen:]:
            self._apply(e)
        self.seen = len(log)
        self.history = list(log)

    def _apply(self, e):
        who, kind = e["player"], e["type"]
        other = "opponent" if who == "me" else "me"
        one = lambda r: [1 if i == RIDX[r] else 0 for i in range(5)]
        if kind == "ROLL":  # every building next to a tile with this number collects
            number = sum(e["dice"])
            if number == 7:
                return
            for t, num in enumerate(self.b.tile_num):
                if num != number or t == self.robber:
                    continue
                r = self.b.tile_res[t]
                for pl in ("me", "opponent"):
                    for n in self.b.tile_nodes[t]:
                        if n in self.sett[pl]:
                            self.hand[pl][r] += 1
                        elif n in self.city[pl]:
                            self.hand[pl][r] += 2
        elif kind == "BUILD_SETTLEMENT":
            n = e["node"]
            self.n_sett[who] += 1
            self.sett[who].add(n)
            if self.n_sett[who] > 2:  # the first two settlements are free
                self._add(who, COST_SETTLE, -1)
            elif self.n_sett[who] == 2:  # the second one gives starting cards
                for t in self.b.node_tiles[n]:
                    if self.b.tile_res[t] is not None:
                        self.hand[who][self.b.tile_res[t]] += 1
        elif kind == "BUILD_ROAD":
            self.n_road[who] += 1
            if self.n_road[who] > 2:
                if self.free[who] > 0:
                    self.free[who] -= 1
                else:
                    self._add(who, COST_ROAD, -1)
        elif kind == "BUILD_CITY":
            self.sett[who].discard(e["node"])
            self.city[who].add(e["node"])
            self._add(who, COST_CITY, -1)
        elif kind == "BUY_DEVELOPMENT_CARD":
            self._add(who, COST_DEV, -1)
        elif kind == "MOVE_ROBBER":
            self.robber = self.b.tile_id[tuple(e["tile"])]
            victim = e.get("stole_from")
            if victim is not None and e.get("stolen_resource"):
                card = one(e["stolen_resource"])
                self._add(victim, card, -1)
                self._add(who, card, 1)
        elif kind == "DISCARD":
            if "cards" in e:
                for c in e["cards"]:
                    self._add(who, one(c), -1)
            else:
                h = self.hand[who]
                total = sum(h)
                if total > 0:
                    keep = max(0.0, total - e["count"]) / total
                    self.hand[who] = [x * keep for x in h]
        elif kind == "MARITIME_TRADE":
            self.hand[who][RIDX[e["give"]]] = max(0.0, self.hand[who][RIDX[e["give"]]] - e["give_count"])
            self.hand[who][RIDX[e["receive"]]] += 1
        elif kind == "PLAY_YEAR_OF_PLENTY":
            self.played_this_turn = True
            for c in e["resources"]:
                self.hand[who][RIDX[c]] += 1
        elif kind == "PLAY_MONOPOLY":
            self.played_this_turn = True
            r = RIDX[e["resource"]]
            self.hand[who][r] += self.hand[other][r]
            self.hand[other][r] = 0.0
        elif kind == "PLAY_ROAD_BUILDING":
            self.played_this_turn = True
            self.free[who] = 2
        elif kind == "PLAY_KNIGHT_CARD":
            self.played_this_turn = True
        elif kind == "END_TURN":
            self.played_this_turn = False
            self.free[who] = 0

    def opponent_hand(self, count):
        """Expected rival hand, rescaled to the public card count."""
        h = self.hand["opponent"]
        total = sum(h)
        if total <= 1e-9:
            out = [count / 5.0] * 5
        else:
            out = [x * count / total for x in h]
        return out  # the replay state is never modified, so the estimate depends only on the log


_cache = {}


def get_context(obs):
    """Board and tracker live as long as the game; a new board or a shorter log resets them."""
    key = (tuple((t["resource"], t["number"]) for t in obs["board"]["tiles"]),
           tuple((p["resource"], tuple(p["nodes"])) for p in obs["board"]["ports"]))
    ctx = _cache.get("ctx")
    if ctx is None or ctx[0] != key or len(obs["log"]) < ctx[2].seen:
        board = Board(obs["board"])
        ctx = _cache["ctx"] = (key, board, Tracker(board))
    ctx[2].update(obs["log"])
    return ctx[1], ctx[2]


def make_state(obs):
    """Turn the judge's observation into a State we can copy and try actions on.

    Everything about us is known exactly: our cards come from the observation.
    The rival's hand is hidden, so we use the Tracker's estimate from the public log [Session 17].
    The same goes for their development cards: we only know how many they hold, so we
    assume each one is a random card among those we have not seen yet [Session 23]. That
    gives an expected number of hidden victory points and knights for the rival."""
    board, tracker = get_context(obs)
    s = State()
    s.b = board
    s.p = []
    for info in (obs["me"], obs["opponent"]):
        p = Player()
        p.sett = set(info["settlements"])
        p.city = set(info["cities"])
        p.roads = {(min(a, b), max(a, b)) for a, b in info["roads"]}
        p.knights = info["played_development_cards"].get("KNIGHT", 0)
        p.road_len = info["longest_road_length"]
        p.ndev = info["development_cards"]
        p.hidden_vp = 0
        p.dev = [0, 0, 0, 0, 0]
        s.p.append(p)
    me, opp = obs["me"], obs["opponent"]
    s.p[0].hand = [me["resources"][r] for r in RES]
    s.p[0].dev = [me["development_cards_in_hand"][d] for d in DEV]
    s.p[1].hand = tracker.opponent_hand(opp["resource_cards"])
    s.robber = board.tile_id[tuple(obs["robber_tile"])]
    s.road_owner = 0 if me["has_longest_road"] else 1 if opp["has_longest_road"] else None
    s.army_owner = 0 if me["has_largest_army"] else 1 if opp["has_largest_army"] else None
    s.dev_left = obs["development_cards_left"]
    s.played_dev = tracker.played_this_turn
    s.free_roads = 0
    # cards we cannot see: still in the deck or in the rival's hand
    unseen = []
    for k, d in enumerate(DEV):
        seen = s.p[0].dev[k] + me["played_development_cards"].get(d, 0) + opp["played_development_cards"].get(d, 0)
        unseen.append(max(0, DEV_TOTAL[k] - seen))
    s.unseen = tuple(unseen)
    total = sum(unseen)
    if total:  # expected hidden victory points of the rival
        s.p[1].hidden_vp = s.p[1].ndev * unseen[4] / total
        s.p[1].dev = [s.p[1].ndev * unseen[0] / total, 0, 0, 0, 0]
    return s


# ---------------------------------------------------------------- 4. evaluation
# One number per name, computed for each player by features().
FEATURES = (
    "vp", "vp_sq",                                  # victory points and their square (late points matter more)
    "prod_wood", "prod_brick", "prod_sheep", "prod_wheat", "prod_ore",  # expected production per resource (pips)
    "prod_total", "prod_sqrt",                      # all pips; sum of square roots (rewards balance)
    "diversity", "numbers",                         # resources produced; different dice numbers touched
    "port3", "port2_syn",                           # has a 3:1 port; production that has its own 2:1 port
    "cards", "excess",                              # cards in hand up to 7; cards above 7 (discard risk)
    "eta_city", "eta_settle", "eta_dev", "eta_road",  # 1/(1+turns) until each can be paid: nearer = bigger
    "spot0", "spot0b", "spot1", "spot2",            # best and 2nd best open spot now; best with 1 and 2 new roads
    "road_len", "has_road", "knights", "has_army",
    "knights_held", "yop_held", "mono_held", "rb_held",  # development cards held (yop = Year of Plenty,
                                                         # mono = Monopoly, rb = Road Building)
    "settlements", "cities", "hand_div",            # pieces on the board; different resources in hand
)


# Weight of each feature, chosen by hand and then kept or rejected by playing matches. They are not learned.
W_ME = {
    "vp": 1.6, "vp_sq": 0.04, "prod_wood": 0.045, "prod_brick": 0.045, "prod_sheep": 0.025, "prod_wheat": 0.06,
    "prod_ore": 0.055, "prod_total": 0.02, "prod_sqrt": 0.12, "diversity": 0.16, "numbers": 0.03, "port3": 0.12,
    "port2_syn": 0.025, "cards": 0.07, "excess": -0.09, "eta_city": 0.9, "eta_settle": 0.8, "eta_dev": 0.3,
    "eta_road": 0.2, "spot0": 0.035, "spot0b": 0.012, "spot1": 0.02, "spot2": 0.008, "road_len": 0.05,
    "has_road": 0.2, "knights": 0.18, "has_army": 0.2, "knights_held": 0.3, "yop_held": 0.35, "mono_held": 0.35,
    "rb_held": 0.35, "settlements": 0.25, "cities": 0.2, "hand_div": 0.04,
}
# The rival's features count against us, multiplied by this. Hand-chosen below 1 so that our own progress matters
# more than blocking. A small test (96 games, 0.55 against 1.0) found no clear difference.
W_OPP_SCALE = 0.55


def _vec(d, scale=1.0):  # weights as a list in FEATURES order
    return [d.get(f, 0.0) * scale for f in FEATURES]


_w = {"me": _vec(W_ME), "opp": _vec(W_ME, -W_OPP_SCALE)}


def eta(cost, hand, rate, ratio):
    """Estimate how many of our turns it will take until we can pay `cost`.

    cost: cards needed per resource. hand: cards held now. rate: expected cards gained per turn.
    ratio: bank trade rate (how many of a spare resource buy 1 card).

    Each turn we expect to collect `rate` cards of each resource. So after t turns our
    hand should be hand + rate * t. Whatever we are still missing can also be covered
    by trading our surplus with the bank, at `ratio` cards for one.
    We check this at the turns listed in ETA_STEPS: every turn up to 6, then with wider
    gaps. Far-off estimates do not need to be precise, and this function runs thousands
    of times per game. When the cost becomes affordable between two checked turns, we
    draw a straight line between them to estimate the exact moment.
    If it is still not affordable after 25 turns, we return 30.
    """
    prev_t, prev_gap = 0, None
    for t in ETA_STEPS:
        need = spare = 0.0
        for r in range(5):
            d = hand[r] + rate[r] * t - cost[r]
            if d < 0:
                need -= d
            else:
                spare += d / ratio[r]
        gap = need - spare
        if gap <= 1e-9:  # affordable at turn t
            if prev_gap is None:
                return float(t)
            return prev_t + (t - prev_t) * prev_gap / (prev_gap - gap)  # where the straight line crosses zero
        prev_t, prev_gap = t, gap
    return 30.0


def expansion(s, i):
    """Looks for places to expand.

    Returns ([best spot now, second best now, best with 1 new road, best with 2 new roads], spots open now).
    A value of 0 means there is none. The search goes outward from my roads, one road at a time.
    """
    b, p, q = s.b, s.p[i], s.p[1 - i]
    enemy_nodes = q.sett | q.city
    taken = p.roads | q.roads
    occupied = p.sett | p.city | enemy_nodes
    blocked = set(occupied)
    for n in occupied:
        blocked.update(b.adj[n])
    frontier = set(p.sett) | p.city
    for a, c in p.roads:
        frontier.add(a)
        frontier.add(c)
    on_road = set()
    for a, c in p.roads:
        on_road.add(a)
        on_road.add(c)
    values = b.spot_value
    now = sorted((values[n] for n in on_road if n not in blocked), reverse=True)
    best = [now[0] if now else 0.0, now[1] if len(now) > 1 else 0.0, 0.0, 0.0]
    seen = set(frontier)
    level = [n for n in frontier if n not in enemy_nodes]
    for depth in (2, 3):
        nxt = []
        for n in level:
            for m in b.adj[n]:
                if m in seen or ((n, m) if n < m else (m, n)) in taken:
                    continue
                seen.add(m)
                if m in enemy_nodes:
                    continue
                nxt.append(m)
                if m not in blocked and values[m] > best[depth]:
                    best[depth] = values[m]
        level = nxt
    return best, len(now)


# chance of at least one 7 in two rolls: a 7 has probability 6/36 = 1/6, so 1 - (5/6)**2 = 0.31
DISCARD_P = 0.31


def features(s, i):
    """The FEATURES of player i in position s [Session 15], as a list in FEATURES order."""
    b, p = s.b, s.p[i]
    prod = s.production(i)  # with the robber where it is now
    long_run = s.production(i, robber=False)
    mixed = [0.4 * prod[r] + 0.6 * long_run[r] for r in range(5)]  # the robber will not stay forever
    rate = [x * 2.0 / 36.0 for x in mixed]  # cards per turn of ours: pips/36 per roll, 2 rolls (ours and the rival's)
    ratio = s.ratios(i)
    hand = p.hand
    n_cards = sum(hand)
    vp = s.vp(i)
    ports = {b.port[n] for n in (p.sett | p.city) if n in b.port}
    numbers = set()
    for n in p.sett | p.city:
        for t in b.node_tiles[n]:
            if b.tile_num[t]:
                numbers.add(b.tile_num[t])
    best, n_open = expansion(s, i)
    if n_cards > 7.0:
        # a 7 in the two rolls before we can spend again costs half the hand: use the hand we expect to keep
        keep = 1.0 - DISCARD_P * (n_cards // 2) / n_cards
        hand = [x * keep for x in hand]

    f = dict.fromkeys(FEATURES, 0.0)  # anything not set below stays 0
    f["vp"] = vp
    f["vp_sq"] = vp * vp
    for r in range(5):
        f["prod_" + RES[r].lower()] = mixed[r]
    f["prod_total"] = sum(mixed)
    f["prod_sqrt"] = sum(math.sqrt(x) for x in mixed)
    f["diversity"] = sum(1.0 for x in mixed if x > 0.5)
    f["numbers"] = len(numbers)
    f["port3"] = 1.0 if 5 in ports else 0.0
    f["port2_syn"] = sum(mixed[r] for r in range(5) if r in ports)
    f["cards"] = min(n_cards, 7.0)
    f["excess"] = max(0.0, n_cards - 7.0)

    can_settle = len(p.sett) < 5  # each player has 5 settlements, 4 cities, 15 roads
    if p.sett and len(p.city) < 4:
        f["eta_city"] = 1.0 / (1.0 + eta(COST_CITY, hand, rate, ratio))
    if can_settle and n_open:
        f["eta_settle"] = 1.0 / (1.0 + eta(COST_SETTLE, hand, rate, ratio))
    elif can_settle:  # no open spot: the cost also includes 1 or 2 roads to reach one
        extra = 1 if best[2] else 2
        cost = (1 + extra, 1 + extra, 1, 1, 0)
        f["eta_settle"] = 1.0 / (1.0 + eta(cost, hand, rate, ratio))
    if s.dev_left > 0:
        f["eta_dev"] = 1.0 / (1.0 + eta(COST_DEV, hand, rate, ratio))
    if len(p.roads) < 15 and not n_open:  # a road only matters here when there is nowhere to build
        f["eta_road"] = 1.0 / (1.0 + eta(COST_ROAD, hand, rate, ratio))
    if can_settle:
        f["spot0"], f["spot0b"], f["spot1"], f["spot2"] = best

    f["road_len"] = min(p.road_len, 9)
    f["has_road"] = 1.0 if s.road_owner == i else 0.0
    f["knights"] = min(p.knights, 4)
    f["has_army"] = 1.0 if s.army_owner == i else 0.0
    f["knights_held"] = min(p.dev[0], 3)
    f["yop_held"], f["mono_held"], f["rb_held"] = p.dev[1], p.dev[2], p.dev[3]
    f["settlements"] = len(p.sett)
    f["cities"] = len(p.city)
    f["hand_div"] = sum(min(1.0, x) for x in hand)
    return [f[name] for name in FEATURES]


# Weight of progress towards Largest Army. Chosen by matches: 1.6 won 54.1% +/- 2.1 against the version without the
# term, 0.7 won 53.6% +/- 1.8 (so the term helps; 1.6 and 0.7 are not clearly different from each other).
ARMY_W = 1.6


def army_progress(s, i):
    """0..1: how close player i is to taking Largest Army (needs 3 knights, or 1 more than the holder).

    Knights played plus held, divided by the number needed, then squared: the last knight is worth the most.
    """
    if s.army_owner == i:
        return 0.0
    p, q = s.p[i], s.p[1 - i]
    need = 3 if s.army_owner is None else q.knights + 1
    return (min(p.knights + p.dev[0], need) / need) ** 2


def score(s):
    """How good position s is for us [Session 15]. Higher is better.

    We add up our features, each multiplied by its weight in W_ME, and subtract the
    rival's features scaled down by W_OPP_SCALE. Our own progress counts more than
    blocking theirs. Progress towards Largest Army is added separately, weighted by
    ARMY_W. The result has no unit: it is only used to compare positions with each
    other. A position where we already have 7 points beats everything.
    """
    if s.vp(0) >= 7:  # a winning position beats everything
        return 1000.0 + sum(s.p[0].hand) * 0.01
    mine, theirs = features(s, 0), features(s, 1)
    total = (sum(w * x for w, x in zip(_w["me"], mine))
             + sum(w * x for w, x in zip(_w["opp"], theirs)))  # these weights are already negative
    total += ARMY_W * (army_progress(s, 0) - W_OPP_SCALE * army_progress(s, 1))
    return total


def squash(logit):
    """Map a score to 0..1. It orders positions; it is NOT a calibrated win probability."""
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit / SCALE))))


SCALE = 3.0  # only stretches the squash() curve; it does not change which action is best


# ---------------------------------------------------------------- 5. search
BEAM_WIDTH = 6  # positions kept at each step of the search
MAX_DEPTH = 5   # longest sequence of actions looked at within one turn


def apply_action(s, a):
    """Transition model RESULT(s, a) [Session 3]: the position after one of OUR actions.

    The action uses the same format as observation["actions"]. s is not changed: we
    work on a copy, so the search can try many actions from the same position."""
    t = s.copy()
    kind = a["type"]
    me = t.p[0]
    if kind == "BUILD_ROAD":
        edge = tuple(a["edge"])
        if t.free_roads > 0:
            t.build_road(0, edge, free=True)
            t.free_roads -= 1
        else:
            t.build_road(0, edge)
    elif kind == "BUILD_SETTLEMENT":
        t.build_settlement(0, a["node"])
    elif kind == "BUILD_CITY":
        t.build_city(0, a["node"])
    elif kind == "MARITIME_TRADE":
        me.hand[RIDX[a["give"]]] -= a["give_count"]
        me.hand[RIDX[a["receive"]]] += 1
    elif kind == "PLAY_YEAR_OF_PLENTY":
        me.dev[1] -= 1
        t.played_dev = True
        for c in a["resources"]:
            me.hand[RIDX[c]] += 1
    elif kind == "PLAY_MONOPOLY":
        me.dev[2] -= 1
        t.played_dev = True
        r = RIDX[a["resource"]]
        me.hand[r] += t.p[1].hand[r]
        t.p[1].hand[r] = 0.0
    elif kind == "PLAY_ROAD_BUILDING":
        me.dev[3] -= 1
        t.played_dev = True
        t.free_roads = 2
    return t


def next_actions(s):
    """Our legal follow-up actions inside the same turn."""
    me = s.p[0]
    out = []
    if s.free_roads > 0 and len(me.roads) < 15:
        out = [{"type": "BUILD_ROAD", "edge": list(e)} for e in sorted(s.road_options(0))]
        if out:
            return out
    if len(me.sett) < 5 and s.can_pay(0, COST_SETTLE):
        out += [{"type": "BUILD_SETTLEMENT", "node": n} for n in sorted(s.settle_options(0))]
    if len(me.city) < 4 and s.can_pay(0, COST_CITY):
        out += [{"type": "BUILD_CITY", "node": n} for n in sorted(me.sett)]
    if len(me.roads) < 15 and s.can_pay(0, COST_ROAD):
        out += [{"type": "BUILD_ROAD", "edge": list(e)} for e in sorted(s.road_options(0))]
    if s.dev_left > 0 and s.can_pay(0, COST_DEV):
        out.append({"type": "BUY_DEVELOPMENT_CARD"})
    if not s.played_dev:
        if me.dev[1] > 0:
            out += [{"type": "PLAY_YEAR_OF_PLENTY", "resources": [RES[i], RES[j]]}
                    for i in range(5) for j in range(i, 5)]
        if me.dev[2] > 0:
            out += [{"type": "PLAY_MONOPOLY", "resource": r} for r in RES]
        if me.dev[3] > 0 and len(me.roads) < 15 and s.can_pay(0, COST_ROAD):  # engine precondition
            out.append({"type": "PLAY_ROAD_BUILDING"})
    ratio = s.ratios(0)
    for i in range(5):
        if me.hand[i] >= ratio[i] - 1e-6:
            out += [{"type": "MARITIME_TRADE", "give": RES[i], "give_count": ratio[i], "receive": RES[j]}
                    for j in range(5) if j != i]
    return out


def buy_value(s):
    """Expected score of buying a development card [Sessions 2 and 23].

    We do not know which card we will get, so we try each type and average the
    results. Each type is weighted by how likely it is: the cards of that type we have
    not seen yet, divided by all the cards we have not seen.
    """
    total = sum(s.unseen)
    base = s.copy()
    base.pay(0, COST_DEV)
    base.dev_left -= 1
    base.p[0].ndev += 1
    if not total:
        return score(base)
    value = 0.0
    for k, n in enumerate(s.unseen):
        if n:
            t = base.copy()
            t.p[0].dev[k] += 1
            value += n / total * score(t)
    return value


def knight_value(s):
    """Best score after playing a knight and moving the robber."""
    t = s.copy()
    t.play_knight(0)
    return max(score(u) for u, _ in robber_outcomes(t, 0))


def robber_outcomes(s, i):
    """Every position player i can reach by moving the robber."""
    for tile in range(len(s.b.tile_res)):
        if tile == s.robber:
            continue
        owners = any(n in s.p[1 - i].sett or n in s.p[1 - i].city for n in s.b.tile_nodes[tile])
        u = s.copy()
        u.move_robber(i, tile, owners and sum(u.p[1 - i].hand) > 0.5)
        yield u, tile


def state_key(s):  # two action orders that reach the same position are searched once
    me = s.p[0]
    return (tuple(round(x, 2) for x in me.hand), frozenset(me.roads), frozenset(me.sett), frozenset(me.city),
            tuple(me.dev), s.free_roads, s.played_dev)


def plan(s, actions, deadline=None):
    """Choose the best first action of our turn with a local beam search [Session 6].

    A turn can be several actions in a row (trade, then build, then buy), and trying
    every possible sequence would take too long. So, as in local beam search, we keep
    only the BEAM_WIDTH best positions at each step:
      1. Try every legal action from the current position.
      2. Score each resulting position with score() [Session 15].
      3. Keep the BEAM_WIDTH best ones and extend them with further actions, up to
         MAX_DEPTH actions.
    Positions reached by two different orders of actions are only searched once
    (state_key), which also keeps the beam from filling up with copies of the same
    position. Ending the turn right now is the value to beat.
    We remember which FIRST action led to the best position seen, and play only that
    one. The judge then asks again, and we search again from the new position.
    The rival's reply is not searched."""
    
    best_value, best_id = -1e18, None
    seen = set()
    frontier = []
    end = next((a for a in actions if a["type"] == "END_TURN"), None)
    if end is not None:
        best_value, best_id = score(s), end["id"]
    for a in actions:
        kind = a["type"]
        if kind == "END_TURN":
            continue
        if kind == "BUY_DEVELOPMENT_CARD":
            value = buy_value(s)
        elif kind == "PLAY_KNIGHT_CARD":
            value = knight_value(s)
        else:
            t = apply_action(s, a)
            value = score(t)
            # Two action orders that reach the same position are searched once,
            # like the reached set [Session 4].
            key = state_key(t)
            if key in seen:
                continue
            seen.add(key)
            frontier.append((value, a["id"], t))
            if kind == "PLAY_ROAD_BUILDING" or (t.free_roads and end is None):
                value = -1e17  # the free roads must be placed before this state counts
        if value > best_value or best_id is None:
            best_value, best_id = value, a["id"]
    for _ in range(MAX_DEPTH - 1):
        if not frontier:
            break
        frontier.sort(key=lambda x: (not x[2].free_roads, -x[0]))
        nxt = []
        for _, first, s2 in frontier[:BEAM_WIDTH]:
            for a in next_actions(s2):
                if a["type"] == "BUY_DEVELOPMENT_CARD":
                    value = buy_value(s2)
                else:
                    t = apply_action(s2, a)
                    key = state_key(t)
                    if key in seen:
                        continue
                    seen.add(key)
                    value = score(t)
                    nxt.append((value, first, t))
                    if t.free_roads:
                        continue
                if value > best_value:
                    best_value, best_id = value, first
        frontier = nxt
    return best_id


def choose_robber(s, actions):  # the robber move whose resulting position scores highest
    best, best_id = -1e18, actions[0]["id"]
    for a in actions:
        u = s.copy()
        u.move_robber(0, s.b.tile_id[tuple(a["tile"])], a.get("steal_from") == "opponent")
        v = score(u)
        if v > best:
            best, best_id = v, a["id"]
    return best_id


def pre_roll(s, actions):
    """ROLL, or play a knight first: if it wins, or if the robber blocks us and moving it improves the score."""
    roll = next(a for a in actions if a["type"] == "ROLL")
    knight = next((a for a in actions if a["type"] == "PLAY_KNIGHT_CARD"), None)
    if knight is None:
        return roll["id"]
    blocked = sum(s.production(0, robber=False)) - sum(s.production(0))
    if knight_value(s) >= 1000 or (blocked > 0 and knight_value(s) > score(s) - KNIGHT_MARGIN):
        return knight["id"]
    return roll["id"]


KNIGHT_MARGIN = 0.0


# ---------------------------------------------------------------- 6. placement
# opening weights (hand-chosen): value of a pip of each resource, then bonuses
P_W = {"wood": 1.0, "brick": 1.0, "sheep": 0.7, "wheat": 1.15, "ore": 1.05, "new_res": 1.6, "new_num": 0.35,
       "port3": 1.0, "port2": 0.45, "expand": 0.18, "cards": 0.35, "deny": 0.15}


def node_score(s, i, node):
    """Value of corner `node` for player i's opening settlement."""
    b = s.b
    mine = s.p[i].sett | s.p[i].city
    have = s.production(i, robber=False)
    prod = b.node_prod[node]
    names = ("wood", "brick", "sheep", "wheat", "ore")
    value = sum(P_W[names[r]] * prod[r] / (1.0 + 0.04 * have[r]) for r in range(5))
    value += P_W["new_res"] * sum(1 for r in range(5) if prod[r] and not have[r])
    nums = {b.tile_num[t] for n in mine for t in b.node_tiles[n]}
    value += P_W["new_num"] * len({b.tile_num[t] for t in b.node_tiles[node] if b.tile_num[t]} - nums)
    port = b.port.get(node)
    if port == 5:
        value += P_W["port3"]
    elif port is not None:
        value += P_W["port2"] * 0.3 * (prod[port] + have[port])
    ring = set()
    for m in b.adj[node]:
        for k in b.adj[m]:
            if k != node and s.free_spot(k):
                ring.add(b.spot_value[k])
    value += P_W["expand"] * sum(sorted(ring, reverse=True)[:2])
    if mine:  # second settlement: the starting cards
        value += P_W["cards"] * sum(1 for r in range(5) if prod[r])
    return value


def initial_settlement(s, actions):  # best corner for us, minus a little for what it leaves the rival
    best, best_id = -1e18, actions[0]["id"]
    for a in actions:
        node = a["node"]
        value = node_score(s, 0, node)
        t = s.copy()
        t.p[0].sett.add(node)
        reply = max((node_score(t, 1, n) for n in t.b.nodes if t.free_spot(n)), default=0.0)
        value -= P_W["deny"] * reply
        if value > best:
            best, best_id = value, a["id"]
    return best_id


def initial_road(s, actions):  # road pointing at the best corners we could settle later
    b = s.b
    own = s.p[0].sett | s.p[0].city
    best, best_id = -1e18, actions[0]["id"]
    for a in actions:
        x, y = a["edge"]
        start, end = (x, y) if x in own else (y, x)
        value = 0.0
        for m in b.adj[end]:
            if m == start:
                continue
            if s.free_spot(m):
                value = max(value, b.spot_value[m])
            for k in b.adj[m]:
                if k != end and s.free_spot(k):
                    value = max(value, 0.6 * b.spot_value[k])
        if value > best:
            best, best_id = value, a["id"]
    return best_id


# ---------------------------------------------------------------- the policy that plays
def decide(obs):
    """Pick the method that fits the phase of the game; returns an action id."""
    actions = obs["actions"]
    s = make_state(obs)
    kinds = {a["type"] for a in actions}
    if obs["phase"] == "initial_placement":
        if "BUILD_SETTLEMENT" in kinds:
            return initial_settlement(s, actions)
        return initial_road(s, actions)
    if "MOVE_ROBBER" in kinds:
        return choose_robber(s, actions)
    if "ROLL" in kinds:
        return pre_roll(s, actions)
    if obs["phase"] == "free_roads":
        s.free_roads = max(1, get_context(obs)[1].free["me"])
    return plan(s, actions)


def act(observation):
    """Entry point. An exception or illegal id loses the game, so every path ends in a legal id."""
    legal = [a["id"] for a in observation["actions"]]
    remaining = observation.get("time_remaining_s")
    policies = (decide, heuristic_policy)
    if remaining is not None and remaining < LOW_CLOCK_S:
        policies = (heuristic_policy,)
    for policy in policies:
        try:
            choice = policy(observation)
        except Exception as error:  # report it, then fall back
            print("agent error:", repr(error))
            continue
        if choice in legal:
            return {"action": choice}
    return {"action": legal[0]}


LOW_CLOCK_S = 0.8  # below this, stop searching and use the baseline


# ---------------------------------------------------------------- 7. required functions
def state_features(observation):
    """Describe the position in 9 small values: the state s of our Q-table [Session 20].

    A Q-table stores one value per (state, action) pair, so a real Catan position, with its
    board, hands and pieces, would never repeat and nothing could be learned. Instead we keep
    only what matters for choosing the next type of action, an abstraction as in [Session 3]:
    [phase, our points bucket, race, can road, can settlement, can city, can dev card,
     open spot, big hand]
    With so few possible states, each one is seen often enough during training to learn
    a reliable value.
    """
    me, opp = observation["me"], observation["opponent"]
    hand = me["resources"]

    def affords(cost):  # 1 if our hand covers this cost, else 0
        return int(all(hand[RES[r]] >= cost[r] for r in range(5)))

    points = me["victory_points"]
    stage = 0 if points <= 2 else 1 if points <= 4 else 2  # early, middle, close to the 7-point target
    lead = me["visible_victory_points"] - opp["visible_victory_points"]
    race = 1 if lead > 0 else -1 if lead < 0 else 0  # ahead, behind or level
    open_spot = int(bool(make_state(observation).settle_options(0)))  # a place to build a settlement now
    big_hand = int(sum(hand.values()) > 7)  # a roll of 7 would cost half of these cards
    return [observation["phase"], stage, race, affords(COST_ROAD), affords(COST_SETTLE), affords(COST_CITY),
            affords(COST_DEV), open_spot, big_hand]


def action_value(observation, action_id):
    """Value of one legal action: its learned Q-value Q(s, a) from the table [Session 20].

    The table is indexed by the action TYPE (build a road, trade...), so all the roads we
    could build share the same Q-value. To choose between them we add a very small
    hand-set estimate, which only breaks ties and never overrides the learned value.
    A (state, action) pair that never appeared in training counts as 0.
    """
    # find the action by its id; do not assume that the id equals the list position
    action = next(a for a in observation["actions"] if a["id"] == action_id)
    learned = Q.get(q_key(state_features(observation), action["type"]), 0.0)  # 0 if never learned
    return learned + 0.001 * estimated_value(observation, action)


def estimated_value(observation, a):
    """Hand-set estimate (0..1) of the position after action `a`. It orders actions; it is not a probability."""
    s = make_state(observation)
    kind = a["type"]
    if observation["phase"] == "initial_placement":
        t = s.copy()
        if kind == "BUILD_SETTLEMENT":
            t.p[0].sett.add(a["node"])
        else:
            t.p[0].roads.add(tuple(a["edge"]))
        return squash(score(t))
    if kind in ("END_TURN", "ROLL"):
        return squash(score(s))
    if kind == "BUY_DEVELOPMENT_CARD":
        return squash(buy_value(s))
    if kind == "PLAY_KNIGHT_CARD":
        return squash(knight_value(s))
    if kind == "PLAY_ROAD_BUILDING":
        return squash(road_building_value(s))
    if kind == "MOVE_ROBBER":
        t = s.copy()
        t.move_robber(0, s.b.tile_id[tuple(a["tile"])], a.get("steal_from") == "opponent")
        return squash(score(t))
    if observation["phase"] == "free_roads":
        s.free_roads = max(1, get_context(observation)[1].free["me"])
    return squash(score(apply_action(s, a)))


def road_building_value(s):
    """Score after playing Road Building and placing the free roads greedily, one at a time."""
    t = apply_action(s, {"type": "PLAY_ROAD_BUILDING"})
    while t.free_roads > 0:
        options = [a for a in next_actions(t) if a["type"] == "BUILD_ROAD"]
        if not options or len(t.p[0].roads) >= 15:
            break
        t = max((apply_action(t, a) for a in options), key=score)
    return score(t)


def q_update(q, reward, best_next_q, alpha, gamma):
    """One Q-learning update, the temporal-difference rule of [Session 20]:

        new_q = q + alpha * (reward + gamma * best_next_q - q)

    The target, reward + gamma * best_next_q, is what this action seems to be worth now:
    the points it just earned plus the discounted value of the best action in the next
    state. We move the old estimate q a fraction alpha of the way towards that target.
    q: old value of (state, action). reward: points received after the action.
    best_next_q: highest Q-value in the next state, 0 when the game has ended.
    alpha: learning rate, 0..1. gamma: discount for later rewards, 0..1.
    """
    return q + alpha * (reward + gamma * best_next_q - q)


def heuristic_policy(observation):
    """Baseline: a simple reflex agent [Session 2] with fixed condition-action rules.

    No search and no learned values. It tries, in this order: build a settlement,
    then a city, then a road (only if there is nowhere to settle), then buy a card,
    then a trade that makes something affordable. Otherwise it ends the turn.
    """
    s = make_state(observation)
    actions = observation["actions"]
    b = s.b
    by = {}
    for a in actions:
        by.setdefault(a["type"], []).append(a)
    if "BUILD_SETTLEMENT" in by:
        return max(by["BUILD_SETTLEMENT"], key=lambda a: b.spot_value[a["node"]])["id"]
    if observation["phase"] == "initial_placement":
        return initial_road(s, actions)
    if "MOVE_ROBBER" in by:
        def harm(a):  # rival pips blocked, minus 3 times our own
            t = b.tile_id[tuple(a["tile"])]
            total = 0
            for n in b.tile_nodes[t]:
                total += b.tile_pips[t] * ((n in s.p[1].sett) + 2 * (n in s.p[1].city)
                                           - 3 * ((n in s.p[0].sett) + 2 * (n in s.p[0].city)))
            return total + (1 if a.get("steal_from") else 0)
        return max(by["MOVE_ROBBER"], key=harm)["id"]
    if "ROLL" in by:
        blocked = sum(s.production(0, robber=False)) > sum(s.production(0))
        if blocked and "PLAY_KNIGHT_CARD" in by:
            return by["PLAY_KNIGHT_CARD"][0]["id"]
        return by["ROLL"][0]["id"]
    if "BUILD_CITY" in by:
        return max(by["BUILD_CITY"], key=lambda a: b.node_pips[a["node"]])["id"]
    if "BUILD_ROAD" in by and (observation["phase"] == "free_roads" or not s.settle_options(0)):
        def reach(a):
            t = s.copy()
            t.p[0].roads.add(tuple(a["edge"]))
            best, _ = expansion(t, 0)
            return best[0] * 3 + best[2] + 0.3 * best[3]
        return max(by["BUILD_ROAD"], key=reach)["id"]
    for kind in ("BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"):
        if kind in by:
            return by[kind][0]["id"]
    for kind in ("PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY"):
        if kind in by:
            def useful(a):
                t = apply_action(s, a)
                return sum(t.can_pay(0, c) for c in (COST_CITY, COST_SETTLE, COST_DEV))
            return max(by[kind], key=useful)["id"]
    targets = [c for c, ok in ((COST_CITY, bool(s.p[0].sett)), (COST_SETTLE, True), (COST_DEV, s.dev_left > 0)) if ok]
    for a in by.get("MARITIME_TRADE", []):
        t = apply_action(s, a)
        if any(t.can_pay(0, c) and not s.can_pay(0, c) for c in targets):
            return a["id"]
    return by.get("END_TURN", actions)[0]["id"]


# ---------------------------------------------------------------- 8. Q-learning policy
# Learned offline by train_q_catan.py (80,000 games against heuristic_policy; entries seen at least 30 times).
# Key: "state|ACTION_TYPE". Value: learned Q-value (expected discounted reward from there on).
Q = {"['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|END_TURN": 1.8496, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.7376, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.0861, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 1.8985, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 2.109, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 0]|ROLL": 2.0497, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 1]|END_TURN": 1.9797, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.0561, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 2.2895, "['play_turn', 0, -1, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 2.1969, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|END_TURN": 1.8261, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.7337, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.317, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.8942, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.4147, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 0]|ROLL": 2.2667, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 1]|END_TURN": 1.9605, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 2.2693, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 3.194, "['play_turn', 0, -1, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 2.3288, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.079, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 0]|END_TURN": 2.0013, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 1.8532, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 2.59, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 2.1771, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.1875, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 1]|END_TURN": 2.0394, "['play_turn', 0, -1, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 2.0417, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.3096, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|END_TURN": 2.2311, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 2.146, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 2.6841, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 2.4578, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.8447, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.4186, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 1]|END_TURN": 2.2527, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 2.4362, "['play_turn', 0, -1, 0, 0, 0, 1, 1, 1]|PLAY_MONOPOLY": 2.74, "['play_turn', 0, -1, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 3.0151, "['play_turn', 0, -1, 0, 0, 1, 0, 0, 0]|END_TURN": 2.6998, "['play_turn', 0, -1, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 2.9118, "['play_turn', 0, -1, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 2.9135, "['play_turn', 0, -1, 0, 0, 1, 0, 1, 0]|END_TURN": 2.4342, "['play_turn', 0, -1, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 2.9735, "['play_turn', 0, -1, 0, 0, 1, 0, 1, 1]|END_TURN": 2.695, "['play_turn', 0, -1, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 2.5701, "['play_turn', 0, -1, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 3.1357, "['play_turn', 0, -1, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 3.1603, "['play_turn', 0, -1, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 2.6953, "['play_turn', 0, -1, 0, 0, 1, 1, 1, 0]|END_TURN": 2.3448, "['play_turn', 0, -1, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 3.0769, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 2.0645, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|END_TURN": 1.8515, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.8556, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 2.0472, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.0854, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 0]|ROLL": 1.9921, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 1.9682, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 1]|END_TURN": 1.9304, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.0603, "['play_turn', 0, -1, 1, 0, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.2893, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 2.1647, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|END_TURN": 1.8758, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.9485, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.476, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 2.2948, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.3574, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 0]|ROLL": 2.5015, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 2.2674, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 1]|END_TURN": 1.9436, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 2.2194, "['play_turn', 0, -1, 1, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 2.6075, "['play_turn', 0, -1, 1, 0, 1, 0, 0, 1]|BUILD_ROAD": 2.8732, "['play_turn', 0, -1, 1, 0, 1, 0, 1, 0]|BUILD_CITY": 3.1443, "['play_turn', 0, -1, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 3.1645, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 1.9529, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 0]|END_TURN": 2.0385, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 1.9929, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.606, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 2.2649, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 1]|END_TURN": 2.133, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 2.1503, "['play_turn', 0, -1, 1, 1, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.9259, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 1.9759, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 2.4814, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 0]|END_TURN": 2.6019, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.7896, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 2.3662, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 2.8372, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 1]|END_TURN": 2.4373, "['play_turn', 0, -1, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 2.2737, "['play_turn', 0, -1, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 2.2869, "['play_turn', 0, -1, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.0646, "['play_turn', 0, -1, 1, 1, 0, 1, 0, 0]|END_TURN": 2.2591, "['play_turn', 0, -1, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 2.3269, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 2.4884, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 2.8365, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.3946, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 0]|END_TURN": 2.564, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 2.4968, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 3.0457, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.678, "['play_turn', 0, -1, 1, 1, 0, 1, 1, 1]|END_TURN": 2.7468, "['play_turn', 0, -1, 1, 1, 1, 1, 1, 1]|BUILD_ROAD": 2.9673, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|END_TURN": 2.876, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 3.0239, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 3.1338, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 2.9533, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 3.0741, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 0]|ROLL": 3.01, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 1]|END_TURN": 2.996, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 3.0742, "['play_turn', 0, 0, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 3.159, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|END_TURN": 3.1554, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 3.0827, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 3.3809, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 3.1429, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.4194, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 0]|ROLL": 3.2462, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 1]|END_TURN": 3.2051, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 3.3129, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 3.7662, "['play_turn', 0, 0, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 3.3954, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 3.099, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|END_TURN": 2.9484, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 2.9516, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 3.1428, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 3.1551, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 0]|PLAY_YEAR_OF_PLENTY": 3.3533, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.066, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 1]|END_TURN": 3.0249, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 3.1098, "['play_turn', 0, 0, 0, 0, 0, 1, 0, 1]|PLAY_MONOPOLY": 3.3759, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 3.318, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|END_TURN": 3.2569, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 3.1384, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 3.5878, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 3.4732, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.9471, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.4299, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 1]|END_TURN": 3.2815, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 3.5546, "['play_turn', 0, 0, 0, 0, 0, 1, 1, 1]|PLAY_MONOPOLY": 3.793, "['play_turn', 0, 0, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 3.9439, "['play_turn', 0, 0, 0, 0, 1, 0, 0, 0]|END_TURN": 3.7355, "['play_turn', 0, 0, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 4.0703, "['play_turn', 0, 0, 0, 0, 1, 0, 0, 1]|END_TURN": 3.6376, "['play_turn', 0, 0, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 3.3729, "['play_turn', 0, 0, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 4.0825, "['play_turn', 0, 0, 0, 0, 1, 0, 1, 0]|END_TURN": 3.8439, "['play_turn', 0, 0, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 4.1573, "['play_turn', 0, 0, 0, 0, 1, 0, 1, 1]|END_TURN": 3.7761, "['play_turn', 0, 0, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 3.306, "['play_turn', 0, 0, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 4.01, "['play_turn', 0, 0, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 4.0017, "['play_turn', 0, 0, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.2741, "['play_turn', 0, 0, 0, 0, 1, 1, 0, 1]|END_TURN": 3.5294, "['play_turn', 0, 0, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 4.0364, "['play_turn', 0, 0, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 4.1446, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 3.0558, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|END_TURN": 2.9947, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 2.9686, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 3.1639, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 3.0649, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 3.169, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 3.211, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 1]|END_TURN": 2.9927, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 3.1048, "['play_turn', 0, 0, 1, 0, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 3.3205, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 3.1012, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|END_TURN": 3.3004, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 3.1104, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 3.4509, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 3.407, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 3.2212, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 3.1702, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 1]|END_TURN": 3.2553, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 3.5262, "['play_turn', 0, 0, 1, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 3.7216, "['play_turn', 0, 0, 1, 0, 1, 0, 0, 0]|BUILD_CITY": 4.044, "['play_turn', 0, 0, 1, 0, 1, 0, 0, 1]|BUILD_CITY": 4.0597, "['play_turn', 0, 0, 1, 0, 1, 0, 1, 0]|BUILD_CITY": 4.169, "['play_turn', 0, 0, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 4.1823, "['play_turn', 0, 0, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 4.1257, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 3.2453, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 0]|END_TURN": 3.1189, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 3.886, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 3.3661, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 1]|END_TURN": 3.1288, "['play_turn', 0, 0, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 3.1438, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 3.3518, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 3.9663, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 0]|END_TURN": 3.7346, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 3.8599, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 3.611, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 4.0234, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 1]|END_TURN": 3.73, "['play_turn', 0, 0, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 3.3896, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 3.2921, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 3.2164, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 0]|END_TURN": 3.2488, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 3.9146, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 3.348, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.2586, "['play_turn', 0, 0, 1, 1, 0, 1, 0, 1]|END_TURN": 3.2282, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 3.331, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 3.9997, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 3.478, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 0]|END_TURN": 3.8016, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 3.5087, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 4.0608, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.6152, "['play_turn', 0, 0, 1, 1, 0, 1, 1, 1]|END_TURN": 3.7706, "['play_turn', 0, 0, 1, 1, 1, 1, 0, 1]|BUILD_CITY": 4.0861, "['play_turn', 0, 0, 1, 1, 1, 1, 1, 1]|BUILD_CITY": 4.2247, "['play_turn', 0, 0, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 4.1428, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|END_TURN": 1.4128, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.1212, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 1.8615, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 1.3192, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 1.5308, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 0]|ROLL": 1.5045, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 1]|END_TURN": 1.3897, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 1.4258, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 1.9591, "['play_turn', 1, -1, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 1.755, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|END_TURN": 1.4716, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.3523, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 1.9384, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.4566, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 1.9691, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 0]|ROLL": 1.7338, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 1]|END_TURN": 1.5466, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.8106, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 2.2998, "['play_turn', 1, -1, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 2.041, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 1.6607, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|END_TURN": 1.4611, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 1.4971, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 2.2602, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 1.7652, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|PLAY_YEAR_OF_PLENTY": 1.9633, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 0]|ROLL": 2.1099, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.699, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 1]|END_TURN": 1.3766, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 1.6058, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 1]|PLAY_KNIGHT_CARD": 2.4203, "['play_turn', 1, -1, 0, 0, 0, 1, 0, 1]|PLAY_MONOPOLY": 1.8936, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.8466, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 0]|END_TURN": 1.6064, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 1.7781, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 2.2474, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 2.1108, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.9664, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 1]|END_TURN": 1.7514, "['play_turn', 1, -1, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 2.055, "['play_turn', 1, -1, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 2.5404, "['play_turn', 1, -1, 0, 0, 1, 0, 0, 0]|END_TURN": 1.5637, "['play_turn', 1, -1, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 2.6533, "['play_turn', 1, -1, 0, 0, 1, 0, 0, 1]|END_TURN": 1.5073, "['play_turn', 1, -1, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 1.6251, "['play_turn', 1, -1, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 2.2484, "['play_turn', 1, -1, 0, 0, 1, 0, 1, 0]|END_TURN": 1.9198, "['play_turn', 1, -1, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 2.5198, "['play_turn', 1, -1, 0, 0, 1, 0, 1, 1]|END_TURN": 1.7381, "['play_turn', 1, -1, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 1.798, "['play_turn', 1, -1, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 2.1839, "['play_turn', 1, -1, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 2.5004, "['play_turn', 1, -1, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.6566, "['play_turn', 1, -1, 0, 0, 1, 1, 0, 1]|MARITIME_TRADE": 1.7955, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 2.7895, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.878, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 0]|END_TURN": 1.6498, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 2.773, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.9663, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 1]|END_TURN": 1.8865, "['play_turn', 1, -1, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 2.1763, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 1.5136, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|END_TURN": 1.2465, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.0045, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.022, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 1.368, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 1.789, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 1.7301, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 1]|END_TURN": 1.5163, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 1.4499, "['play_turn', 1, -1, 1, 0, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 1.9618, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 1.7953, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|END_TURN": 1.3175, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.4386, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.0991, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.9988, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 1.9894, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.4058, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 0]|ROLL": 1.8759, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 1.7392, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 1]|END_TURN": 1.6199, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.8965, "['play_turn', 1, -1, 1, 0, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 2.3379, "['play_turn', 1, -1, 1, 0, 1, 0, 0, 1]|BUILD_CITY": 2.6935, "['play_turn', 1, -1, 1, 0, 1, 0, 1, 0]|BUILD_CITY": 2.8975, "['play_turn', 1, -1, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 3.0292, "['play_turn', 1, -1, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 2.0216, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 1.5468, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 0]|END_TURN": 1.3347, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 1.4617, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.4286, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 1.7811, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 1]|END_TURN": 1.6141, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 1.6006, "['play_turn', 1, -1, 1, 1, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.9259, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 1.7059, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 2.1946, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 0]|END_TURN": 2.0145, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 0]|PLAY_MONOPOLY": 2.414, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.7858, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 1.9524, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 2.5451, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 1]|END_TURN": 2.3209, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 2.0282, "['play_turn', 1, -1, 1, 1, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 2.9821, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 1.8614, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 1.6986, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 0]|END_TURN": 1.5063, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 2.5467, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 2.0862, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.8938, "['play_turn', 1, -1, 1, 1, 0, 1, 0, 1]|END_TURN": 1.881, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 1.9691, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 2.4487, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.0188, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 0]|END_TURN": 1.9183, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 0]|PLAY_ROAD_BUILDING": 2.7701, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 2.044, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 2.4908, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.2109, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|END_TURN": 2.2661, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|MARITIME_TRADE": 2.2194, "['play_turn', 1, -1, 1, 1, 0, 1, 1, 1]|PLAY_ROAD_BUILDING": 2.8889, "['play_turn', 1, -1, 1, 1, 1, 1, 0, 1]|BUILD_CITY": 2.618, "['play_turn', 1, -1, 1, 1, 1, 1, 1, 1]|BUILD_ROAD": 2.4891, "['play_turn', 1, -1, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 2.5251, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|END_TURN": 2.3221, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 2.1948, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.6496, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 2.3194, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 2.5694, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 0]|ROLL": 2.4466, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 1]|END_TURN": 2.3632, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.4723, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 2.9911, "['play_turn', 1, 0, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 2.7154, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|END_TURN": 2.4653, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 2.6458, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.8164, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 2.4789, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.8482, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 0]|ROLL": 2.7919, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 1]|END_TURN": 2.612, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 2.791, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 3.2685, "['play_turn', 1, 0, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 3.1477, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.5652, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|END_TURN": 2.4519, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 2.3943, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 2.8497, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 2.7337, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 0]|PLAY_YEAR_OF_PLENTY": 3.0036, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.66, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 1]|END_TURN": 2.2829, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 2.6458, "['play_turn', 1, 0, 0, 0, 0, 1, 0, 1]|PLAY_MONOPOLY": 2.8323, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.7349, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|END_TURN": 2.681, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 2.5237, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 3.1731, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 2.9843, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.4895, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.841, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 1]|END_TURN": 2.7813, "['play_turn', 1, 0, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 3.0704, "['play_turn', 1, 0, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 3.3347, "['play_turn', 1, 0, 0, 0, 1, 0, 0, 0]|END_TURN": 2.5801, "['play_turn', 1, 0, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 3.5635, "['play_turn', 1, 0, 0, 0, 1, 0, 0, 1]|END_TURN": 2.6238, "['play_turn', 1, 0, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 2.6916, "['play_turn', 1, 0, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 3.435, "['play_turn', 1, 0, 0, 0, 1, 0, 1, 0]|END_TURN": 2.7773, "['play_turn', 1, 0, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 3.6053, "['play_turn', 1, 0, 0, 0, 1, 0, 1, 1]|END_TURN": 2.6681, "['play_turn', 1, 0, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 2.8457, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 3.2048, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 0]|END_TURN": 2.7995, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 3.6689, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.6124, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 1]|END_TURN": 2.5287, "['play_turn', 1, 0, 0, 0, 1, 1, 0, 1]|MARITIME_TRADE": 2.8071, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 3.3934, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.8232, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 0]|END_TURN": 2.5344, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 3.6126, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.9862, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 1]|END_TURN": 2.7524, "['play_turn', 1, 0, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 3.3358, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 2.5007, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 0]|END_TURN": 2.3491, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 2.3686, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.7778, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.7193, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 2.6542, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 1]|END_TURN": 2.4457, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.5324, "['play_turn', 1, 0, 1, 0, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.9323, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 2.6291, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|END_TURN": 2.7655, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 2.488, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 3.1075, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 2.9644, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.8758, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.4126, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 2.7993, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 1]|END_TURN": 2.6918, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 3.0245, "['play_turn', 1, 0, 1, 0, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 3.1391, "['play_turn', 1, 0, 1, 0, 1, 0, 0, 1]|BUILD_CITY": 3.5545, "['play_turn', 1, 0, 1, 0, 1, 0, 1, 0]|BUILD_ROAD": 3.577, "['play_turn', 1, 0, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 3.7833, "['play_turn', 1, 0, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 2.7783, "['play_turn', 1, 0, 1, 0, 1, 0, 1, 1]|END_TURN": 3.0073, "['play_turn', 1, 0, 1, 0, 1, 0, 1, 1]|MARITIME_TRADE": 3.3557, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 2.4947, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 0]|END_TURN": 2.5529, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 2.4792, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 3.5273, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 2.851, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 1]|END_TURN": 2.4199, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 2.6474, "['play_turn', 1, 0, 1, 1, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 3.5248, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 2.7015, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 3.4518, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 0]|END_TURN": 3.1394, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 3.2105, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 3.6304, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 1]|END_TURN": 3.2652, "['play_turn', 1, 0, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 3.0303, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 2.8016, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.6867, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 0]|END_TURN": 2.4792, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 3.4352, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 2.9096, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.8397, "['play_turn', 1, 0, 1, 1, 0, 1, 0, 1]|END_TURN": 2.8263, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 2.915, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 3.5165, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.9151, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 0]|END_TURN": 3.0429, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 3.1223, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 3.6188, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.1128, "['play_turn', 1, 0, 1, 1, 0, 1, 1, 1]|END_TURN": 3.0651, "['play_turn', 1, 0, 1, 1, 1, 1, 0, 1]|BUILD_CITY": 3.5516, "['play_turn', 1, 0, 1, 1, 1, 1, 1, 1]|BUILD_CITY": 3.6929, "['play_turn', 1, 0, 1, 1, 1, 1, 1, 1]|BUILD_ROAD": 3.2474, "['play_turn', 1, 0, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 3.3521, "['play_turn', 1, 0, 1, 1, 1, 1, 1, 1]|MARITIME_TRADE": 3.4463, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|END_TURN": 2.9422, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 2.9315, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 3.2733, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 3.065, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 3.1735, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 0]|ROLL": 3.1117, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 1]|END_TURN": 3.0365, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 3.1655, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 3.2957, "['play_turn', 1, 1, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 2.7872, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|END_TURN": 3.2203, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 3.0404, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 3.34, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 3.1636, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.3404, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 0]|ROLL": 3.246, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 1]|END_TURN": 3.1839, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 3.3086, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 3.509, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 3.2675, "['play_turn', 1, 1, 0, 0, 0, 0, 1, 1]|PLAY_YEAR_OF_PLENTY": 3.4008, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 3.2351, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|END_TURN": 3.0938, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 2.9916, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 3.3026, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 3.1596, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 0]|PLAY_YEAR_OF_PLENTY": 3.3573, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.2309, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 1]|END_TURN": 3.0988, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 3.1357, "['play_turn', 1, 1, 0, 0, 0, 1, 0, 1]|PLAY_KNIGHT_CARD": 3.4028, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 3.334, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|END_TURN": 3.2303, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 3.0197, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 3.5534, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 3.3609, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.4534, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.3768, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 1]|END_TURN": 3.2388, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 3.2445, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 1]|PLAY_KNIGHT_CARD": 3.692, "['play_turn', 1, 1, 0, 0, 0, 1, 1, 1]|PLAY_MONOPOLY": 3.3574, "['play_turn', 1, 1, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 3.6973, "['play_turn', 1, 1, 0, 0, 1, 0, 0, 0]|END_TURN": 3.2198, "['play_turn', 1, 1, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 3.6351, "['play_turn', 1, 1, 0, 0, 1, 0, 0, 1]|END_TURN": 3.0887, "['play_turn', 1, 1, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 3.1004, "['play_turn', 1, 1, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 3.7995, "['play_turn', 1, 1, 0, 0, 1, 0, 1, 0]|END_TURN": 3.1389, "['play_turn', 1, 1, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 3.7704, "['play_turn', 1, 1, 0, 0, 1, 0, 1, 1]|END_TURN": 3.1434, "['play_turn', 1, 1, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 3.2708, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 3.6429, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 3.1615, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 3.6733, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.1406, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 1]|END_TURN": 3.1787, "['play_turn', 1, 1, 0, 0, 1, 1, 0, 1]|MARITIME_TRADE": 3.2769, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 3.712, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 3.2724, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 3.8058, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.3581, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 1]|END_TURN": 3.2575, "['play_turn', 1, 1, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 3.428, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 3.1585, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|END_TURN": 3.0054, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 2.9769, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 3.1118, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 3.0851, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 3.2136, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 3.1956, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 1]|END_TURN": 3.0996, "['play_turn', 1, 1, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 3.1087, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 3.1648, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|END_TURN": 3.0773, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 3.1352, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 3.4224, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 3.2238, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 3.3222, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 3.475, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 3.3077, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 1]|END_TURN": 3.132, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 3.3656, "['play_turn', 1, 1, 1, 0, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 3.4416, "['play_turn', 1, 1, 1, 0, 1, 0, 0, 1]|BUILD_ROAD": 3.5753, "['play_turn', 1, 1, 1, 0, 1, 0, 1, 0]|BUILD_ROAD": 3.7021, "['play_turn', 1, 1, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 3.6033, "['play_turn', 1, 1, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 3.2945, "['play_turn', 1, 1, 1, 0, 1, 0, 1, 1]|END_TURN": 3.3518, "['play_turn', 1, 1, 1, 0, 1, 0, 1, 1]|MARITIME_TRADE": 3.4242, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 3.1942, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 0]|END_TURN": 2.9697, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 3.0863, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 3.4836, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 3.2494, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 1]|END_TURN": 3.1592, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 3.1126, "['play_turn', 1, 1, 1, 1, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 3.582, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 3.2965, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 3.5221, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 0]|END_TURN": 3.3142, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 3.4249, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 3.4185, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 1]|END_TURN": 3.3855, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 3.3532, "['play_turn', 1, 1, 1, 1, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 3.6084, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 3.3218, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 3.1755, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 0]|END_TURN": 3.1766, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 3.4671, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 3.3404, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 3.2063, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 1]|END_TURN": 3.1621, "['play_turn', 1, 1, 1, 1, 0, 1, 0, 1]|PLAY_ROAD_BUILDING": 3.4985, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 3.3107, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 3.5839, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 3.3025, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 0]|END_TURN": 3.2838, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 3.4396, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 3.383, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 3.3923, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 1]|END_TURN": 3.3501, "['play_turn', 1, 1, 1, 1, 0, 1, 1, 1]|MARITIME_TRADE": 3.3056, "['play_turn', 1, 1, 1, 1, 1, 1, 0, 1]|BUILD_CITY": 3.2966, "['play_turn', 1, 1, 1, 1, 1, 1, 0, 1]|BUILD_ROAD": 3.5, "['play_turn', 1, 1, 1, 1, 1, 1, 1, 1]|BUILD_ROAD": 3.6551, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|END_TURN": 0.529, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 0.6132, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 1.0614, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 0.5094, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 1.059, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 0]|ROLL": 0.9882, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 1]|END_TURN": 0.7942, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 0.9984, "['play_turn', 2, -1, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 1.2142, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|END_TURN": 0.9975, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 0.6714, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 1.209, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 0.5534, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 1.373, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 0]|ROLL": 1.1874, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 1]|END_TURN": 0.9797, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.1567, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 1.8145, "['play_turn', 2, -1, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 1.463, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 0.9873, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 0]|END_TURN": 0.742, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 0.9619, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 1.3882, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.1828, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 1]|END_TURN": 0.8164, "['play_turn', 2, -1, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 1.2169, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.2238, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 0]|END_TURN": 1.0851, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 1.1355, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 1.6162, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 1.5645, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.3644, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 1]|END_TURN": 1.1438, "['play_turn', 2, -1, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 1.5272, "['play_turn', 2, -1, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 2.0703, "['play_turn', 2, -1, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 1.9944, "['play_turn', 2, -1, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 1.137, "['play_turn', 2, -1, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 1.985, "['play_turn', 2, -1, 0, 0, 1, 0, 1, 0]|END_TURN": 1.211, "['play_turn', 2, -1, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 2.1422, "['play_turn', 2, -1, 0, 0, 1, 0, 1, 1]|END_TURN": 0.8012, "['play_turn', 2, -1, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 1.4183, "['play_turn', 2, -1, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 2.1097, "['play_turn', 2, -1, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.2722, "['play_turn', 2, -1, 0, 0, 1, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.3029, "['play_turn', 2, -1, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 2.1684, "['play_turn', 2, -1, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.4272, "['play_turn', 2, -1, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 1.9632, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 0.7755, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|END_TURN": 0.6007, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 0.5423, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 1.1441, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 0.9161, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 1.2325, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 0.9933, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 1]|END_TURN": 0.5826, "['play_turn', 2, -1, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 0.8852, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 0.6592, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|END_TURN": 0.9144, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 0.7092, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 1.4188, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.2468, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 1.5071, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 1.3353, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 1]|END_TURN": 1.0607, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.4559, "['play_turn', 2, -1, 1, 0, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 1.6647, "['play_turn', 2, -1, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 2.3174, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 0.9649, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 0]|END_TURN": 0.6995, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 1.9596, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 1.3677, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 1]|END_TURN": 0.7453, "['play_turn', 2, -1, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 1.1065, "['play_turn', 2, -1, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 1.5726, "['play_turn', 2, -1, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 2.0305, "['play_turn', 2, -1, 1, 1, 0, 0, 1, 0]|END_TURN": 1.1362, "['play_turn', 2, -1, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 1.2973, "['play_turn', 2, -1, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 2.0226, "['play_turn', 2, -1, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 1.0576, "['play_turn', 2, -1, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 1.1697, "['play_turn', 2, -1, 1, 1, 0, 1, 0, 0]|END_TURN": 0.8516, "['play_turn', 2, -1, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 2.051, "['play_turn', 2, -1, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 1.4231, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 1.6942, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 1.8695, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.4448, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 0]|END_TURN": 1.1078, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 1.991, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 2.1542, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.5356, "['play_turn', 2, -1, 1, 1, 0, 1, 1, 1]|END_TURN": 1.2494, "['play_turn', 2, -1, 1, 1, 1, 1, 1, 1]|BUILD_CITY": 2.2949, "['play_turn', 2, -1, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 2.2255, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|END_TURN": 0.9523, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.1817, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 1.5966, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 1.2209, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 1.4538, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 0]|ROLL": 1.3482, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 1]|END_TURN": 1.3863, "['play_turn', 2, 0, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 1.4848, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|END_TURN": 1.2246, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.1282, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 1.6104, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.0534, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 1.7888, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 0]|ROLL": 1.5597, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 1]|END_TURN": 1.4374, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.7872, "['play_turn', 2, 0, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 1.8515, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 1.5649, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 0]|END_TURN": 1.4342, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 1.1613, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 2.1303, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 1.5691, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.6746, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 1]|END_TURN": 1.5507, "['play_turn', 2, 0, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 1.5266, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.7154, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 0]|END_TURN": 1.4464, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 1.4999, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 1.8828, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 1.9093, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.6612, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 1]|END_TURN": 1.3716, "['play_turn', 2, 0, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 1.9798, "['play_turn', 2, 0, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 2.2261, "['play_turn', 2, 0, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 2.2815, "['play_turn', 2, 0, 0, 0, 1, 0, 0, 1]|END_TURN": 1.3492, "['play_turn', 2, 0, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 1.5761, "['play_turn', 2, 0, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 2.5569, "['play_turn', 2, 0, 0, 0, 1, 0, 1, 0]|END_TURN": 1.7463, "['play_turn', 2, 0, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 2.6928, "['play_turn', 2, 0, 0, 0, 1, 0, 1, 1]|END_TURN": 1.7279, "['play_turn', 2, 0, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 1.8174, "['play_turn', 2, 0, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 2.5409, "['play_turn', 2, 0, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 2.4588, "['play_turn', 2, 0, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.7854, "['play_turn', 2, 0, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 2.3736, "['play_turn', 2, 0, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 2.5038, "['play_turn', 2, 0, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.761, "['play_turn', 2, 0, 0, 0, 1, 1, 1, 1]|END_TURN": 1.5475, "['play_turn', 2, 0, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 2.3277, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 1.237, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 0]|END_TURN": 1.1151, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 0.9728, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 1.4395, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 1.7305, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 1.6275, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 1]|END_TURN": 1.3019, "['play_turn', 2, 0, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 1.4619, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 1.2872, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|END_TURN": 1.7351, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.2387, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 1.8065, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.6756, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 1.8972, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 0]|ROLL": 1.9036, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 1.5913, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 1]|END_TURN": 1.2867, "['play_turn', 2, 0, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 1.9048, "['play_turn', 2, 0, 1, 0, 1, 0, 0, 1]|BUILD_CITY": 2.7024, "['play_turn', 2, 0, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 2.6628, "['play_turn', 2, 0, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 2.0479, "['play_turn', 2, 0, 1, 0, 1, 0, 1, 1]|MARITIME_TRADE": 2.4796, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 1.6229, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 0]|END_TURN": 1.1296, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 1.3265, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.5231, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 1.7909, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 1]|END_TURN": 1.4449, "['play_turn', 2, 0, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 1.6336, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 1.7975, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 2.5468, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 0]|END_TURN": 2.0447, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.5195, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 2.3333, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 2.6569, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 1]|END_TURN": 1.559, "['play_turn', 2, 0, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 1.9177, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 1.6512, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 1.5741, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 2.3795, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 1.8912, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 1.7842, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 1]|END_TURN": 1.7675, "['play_turn', 2, 0, 1, 1, 0, 1, 0, 1]|PLAY_ROAD_BUILDING": 2.5602, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 2.1525, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 2.6044, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 1.9401, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 0]|END_TURN": 1.4983, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 2.1198, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 2.652, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 1.999, "['play_turn', 2, 0, 1, 1, 0, 1, 1, 1]|END_TURN": 1.6208, "['play_turn', 2, 0, 1, 1, 1, 1, 0, 1]|BUILD_ROAD": 2.5551, "['play_turn', 2, 0, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 2.7918, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|END_TURN": 2.0337, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.9126, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.1873, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 1.7287, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 2.1335, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 0]|ROLL": 2.0594, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 1]|END_TURN": 2.0041, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.1153, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 2.251, "['play_turn', 2, 1, 0, 0, 0, 0, 0, 1]|PLAY_MONOPOLY": 1.7963, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|END_TURN": 2.1453, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.9768, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.2637, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 1.9793, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.2807, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 0]|ROLL": 2.1174, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 1]|END_TURN": 2.0736, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 1]|MARITIME_TRADE": 2.1978, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 2.3411, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 2.0897, "['play_turn', 2, 1, 0, 0, 0, 0, 1, 1]|PLAY_YEAR_OF_PLENTY": 2.3436, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.1973, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|END_TURN": 2.0378, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|MARITIME_TRADE": 2.0388, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|PLAY_KNIGHT_CARD": 2.3172, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|PLAY_MONOPOLY": 2.1435, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 0]|PLAY_YEAR_OF_PLENTY": 2.2712, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.2017, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 1]|END_TURN": 1.97, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 1]|MARITIME_TRADE": 2.0889, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 1]|PLAY_KNIGHT_CARD": 2.283, "['play_turn', 2, 1, 0, 0, 0, 1, 0, 1]|PLAY_YEAR_OF_PLENTY": 2.3659, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.2958, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|END_TURN": 2.1584, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|MARITIME_TRADE": 2.0411, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|PLAY_KNIGHT_CARD": 2.3534, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|PLAY_MONOPOLY": 2.1562, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.3864, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 0]|ROLL": 2.3339, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.2191, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 1]|END_TURN": 1.9548, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 1]|MARITIME_TRADE": 2.2102, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 1]|PLAY_KNIGHT_CARD": 2.4633, "['play_turn', 2, 1, 0, 0, 0, 1, 1, 1]|PLAY_MONOPOLY": 2.3017, "['play_turn', 2, 1, 0, 0, 1, 0, 0, 0]|BUILD_CITY": 2.5049, "['play_turn', 2, 1, 0, 0, 1, 0, 0, 0]|END_TURN": 2.1313, "['play_turn', 2, 1, 0, 0, 1, 0, 0, 1]|BUILD_CITY": 2.5881, "['play_turn', 2, 1, 0, 0, 1, 0, 0, 1]|END_TURN": 2.1132, "['play_turn', 2, 1, 0, 0, 1, 0, 0, 1]|MARITIME_TRADE": 2.1669, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 0]|BUILD_CITY": 2.5869, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 0]|END_TURN": 2.135, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 0]|MARITIME_TRADE": 1.7378, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 1]|BUILD_CITY": 2.5675, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 1]|END_TURN": 2.0369, "['play_turn', 2, 1, 0, 0, 1, 0, 1, 1]|MARITIME_TRADE": 2.1955, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 0]|BUILD_CITY": 2.4697, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.1662, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 1]|BUILD_CITY": 2.5666, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.1821, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 1]|END_TURN": 2.1024, "['play_turn', 2, 1, 0, 0, 1, 1, 0, 1]|MARITIME_TRADE": 2.2156, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 0]|BUILD_CITY": 2.4731, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.0002, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 0]|END_TURN": 2.1905, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 1]|BUILD_CITY": 2.5608, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.2023, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 1]|END_TURN": 2.0619, "['play_turn', 2, 1, 0, 0, 1, 1, 1, 1]|MARITIME_TRADE": 2.3443, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|BUILD_ROAD": 2.147, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|END_TURN": 1.8523, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|MARITIME_TRADE": 1.9247, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.12, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|PLAY_MONOPOLY": 2.0552, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.1962, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|PLAY_YEAR_OF_PLENTY": 2.25, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 0]|ROLL": 2.1497, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 1]|BUILD_ROAD": 2.1531, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 1]|END_TURN": 1.8303, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 1]|MARITIME_TRADE": 2.0782, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 2.4134, "['play_turn', 2, 1, 1, 0, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.3812, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|BUILD_ROAD": 1.9497, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|END_TURN": 2.1048, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|MARITIME_TRADE": 1.9259, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.2454, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|PLAY_MONOPOLY": 2.1645, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.3789, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|PLAY_YEAR_OF_PLENTY": 2.39, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 0]|ROLL": 2.2582, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|BUILD_ROAD": 2.1264, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|END_TURN": 2.0598, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|MARITIME_TRADE": 2.268, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 2.4516, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|PLAY_MONOPOLY": 2.2225, "['play_turn', 2, 1, 1, 0, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 2.4452, "['play_turn', 2, 1, 1, 0, 1, 0, 0, 1]|BUILD_ROAD": 2.4653, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 0]|BUILD_CITY": 2.7061, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 0]|BUILD_ROAD": 2.1385, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 1]|BUILD_CITY": 2.447, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 1]|BUILD_ROAD": 2.5504, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 1]|END_TURN": 1.9899, "['play_turn', 2, 1, 1, 0, 1, 0, 1, 1]|MARITIME_TRADE": 2.3477, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 0]|BUILD_ROAD": 2.0887, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 0]|END_TURN": 1.919, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 0]|MARITIME_TRADE": 2.0806, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 0]|PLAY_KNIGHT_CARD": 2.3002, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 0]|PLAY_ROAD_BUILDING": 2.5726, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 1]|BUILD_ROAD": 2.1929, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 1]|END_TURN": 2.0753, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 1]|MARITIME_TRADE": 2.1254, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 1]|PLAY_KNIGHT_CARD": 2.3394, "['play_turn', 2, 1, 1, 1, 0, 0, 0, 1]|PLAY_ROAD_BUILDING": 2.5927, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|BUILD_ROAD": 2.2177, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|BUILD_SETTLEMENT": 2.4466, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|END_TURN": 2.2627, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|MARITIME_TRADE": 2.1521, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|PLAY_KNIGHT_CARD": 2.4982, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 0]|PLAY_ROAD_BUILDING": 2.5867, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|BUILD_ROAD": 2.4375, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|BUILD_SETTLEMENT": 2.4182, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|END_TURN": 2.3198, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|MARITIME_TRADE": 2.2081, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|PLAY_KNIGHT_CARD": 2.5749, "['play_turn', 2, 1, 1, 1, 0, 0, 1, 1]|PLAY_ROAD_BUILDING": 2.6369, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 0]|BUILD_ROAD": 2.24, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 0]|BUY_DEVELOPMENT_CARD": 2.1478, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 0]|END_TURN": 2.1765, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 0]|PLAY_ROAD_BUILDING": 2.5104, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|BUILD_ROAD": 2.2862, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|BUY_DEVELOPMENT_CARD": 2.2376, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|END_TURN": 2.1601, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|MARITIME_TRADE": 2.1577, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|PLAY_KNIGHT_CARD": 2.4025, "['play_turn', 2, 1, 1, 1, 0, 1, 0, 1]|PLAY_ROAD_BUILDING": 2.5558, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 0]|BUILD_ROAD": 2.3538, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 0]|BUILD_SETTLEMENT": 2.5138, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 0]|BUY_DEVELOPMENT_CARD": 2.2211, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 0]|END_TURN": 2.2156, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 0]|PLAY_ROAD_BUILDING": 2.5171, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 1]|BUILD_ROAD": 2.4206, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 1]|BUILD_SETTLEMENT": 2.4297, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.2766, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 1]|END_TURN": 2.2339, "['play_turn', 2, 1, 1, 1, 0, 1, 1, 1]|MARITIME_TRADE": 2.2718, "['play_turn', 2, 1, 1, 1, 1, 1, 0, 1]|BUILD_ROAD": 2.4317, "['play_turn', 2, 1, 1, 1, 1, 1, 1, 1]|BUILD_CITY": 2.37, "['play_turn', 2, 1, 1, 1, 1, 1, 1, 1]|BUILD_ROAD": 2.5709, "['play_turn', 2, 1, 1, 1, 1, 1, 1, 1]|BUILD_SETTLEMENT": 2.3488, "['play_turn', 2, 1, 1, 1, 1, 1, 1, 1]|BUY_DEVELOPMENT_CARD": 2.3619}  # <Q-TABLE>


def q_key(features, kind):  # the table key, e.g. "['play_turn', 0, 0, 1, 0, 0, 0, 1, 0]|BUILD_ROAD"
    return str(features) + "|" + kind


def best_road(s, roads):
    """Road that gets closest to good building spots (same rule as the baseline)."""
    def reach(a):
        t = s.copy()
        t.p[0].roads.add(tuple(a["edge"]))
        best, _ = expansion(t, 0)  # [best now, 2nd best now, best with 1 more road, best with 2 more]
        return best[0] * 3 + best[2] + 0.3 * best[3]
    return max(roads, key=reach)["id"]


def best_of_type(observation, kind):
    """Q-learning chooses the TYPE of action; this simple rule chooses which one of that type."""
    options = [a for a in observation["actions"] if a["type"] == kind]
    if len(options) == 1:
        return options[0]["id"]
    if kind == "BUILD_ROAD" and observation["phase"] != "initial_placement":
        return best_road(make_state(observation), options)
    if kind == "MARITIME_TRADE":
        hand = observation["me"]["resources"]
        # give what we have most of, receive what we have least of
        return max(options, key=lambda a: hand[a["give"]] - a["give_count"] - hand[a["receive"]])["id"]
    return heuristic_policy(dict(observation, actions=options))  # the baseline, restricted to this type


def q_policy(observation, table=None):
    """Play with the learned Q-table, always picking the best-valued action type [Session 20].

    During training (train_q_catan.py) the agent explored with epsilon-greedy, with epsilon
    shrinking over the games [Session 21]. Here we only exploit: as the course recommends
    for evaluation, epsilon = 0 and we take the action type with the highest Q-value.
    The table only chooses the TYPE of action; best_of_type() picks the concrete one.
    Where the table has no value for the current state, we fall back to heuristic_policy().
    """
    table = Q if table is None else table
    baseline_id = heuristic_policy(observation)
    baseline_kind = next(a["type"] for a in observation["actions"] if a["id"] == baseline_id)
    kinds = sorted({a["type"] for a in observation["actions"]})
    if len(kinds) == 1:
        return baseline_id
    features = state_features(observation)
    known = {k: table[q_key(features, k)] for k in kinds if q_key(features, k) in table}
    if baseline_kind not in known:
        return baseline_id
    kind = max(known, key=known.get)  # the action type with the highest learned value
    return baseline_id if kind == baseline_kind else best_of_type(observation, kind)