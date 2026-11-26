"""Tabular Q-learning for the Catan agent, trained offline in batches.


    python train_q_catan.py train    --root ai_agent_tournament --games 20000  # one batch; run 4 times (80,000 games)
    python train_q_catan.py export   --root ai_agent_tournament                # write the table into the agent
    python train_q_catan.py evaluate --root ai_agent_tournament --games 1000   # compare the versions

Training saves q_checkpoint.json after every batch, so a stopped session loses at most the batch in progress.
A batch also stops by itself after --time-limit seconds (240 by default); then simply run it again until the
total says 80000 games. The same total gives the same table however it is split into batches.

Q-learning update (done by q_update in the agent):
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
    Q(s,a) : learned value of taking action type a in abstract state s
    r      : reward = victory points gained since my last decision, plus +1 (win) or -1 (loss) at the end
    s', a' : next state where I decide, and the action types legal there
    alpha  : learning rate, 0..1 (how far one game moves the value)
    gamma  : discount, 0..1 (how much later rewards count)
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

GAMMA = 0.99              # rewards one decision later are worth 99 % as much
EPSILON_START = 0.40      # share of decisions that explore a random action type at the beginning
EPSILON_END = 0.05        # share at the end of the planned training
PLANNED_GAMES = 40000     # exploration shrinks linearly over this many games, then stays at EPSILON_END
MIN_VISITS = 30           # entries seen fewer times are not exported: too noisy to trust
CHECKPOINT = "q_checkpoint.json"
HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.join(HERE, "catan_agent.py")


def load_agent():
    """Import catan_agent.py as a fresh module (its own card tracker and memory)."""
    spec = importlib.util.spec_from_file_location("agent_%d" % random.randrange(10 ** 9), AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def play_one_game(engine, policy_a, policy_b, seed):
    """The loop from docs/students/TRAINING.md. Returns (winner "A"/"B"/None, points, game)."""
    game = engine.new_game(seed, vps_to_win=7)
    policies = {"A": policy_a, "B": policy_b}
    while game.winning_color() is None and game.state.num_turns // 2 < 100:
        actions = engine.legal_actions(game)
        if len(actions) == 1:  # forced: the judge plays it without asking
            game.execute(actions[0])
            continue
        side = engine.SIDE_OF[game.state.current_color()]
        observation = engine.observation_for(game, side, actions, None, {"max_rounds": 100})
        game.execute(actions[policies[side](observation)])
    champion = game.winning_color()
    winner = engine.SIDE_OF[champion] if champion is not None else engine.adjudicate(game)[0]
    return winner, engine.victory_points(game), game


# ------------------------------------------------------------------ training
class Learner:
    """Plays one side with epsilon-greedy exploration and updates Q after every decision."""

    def __init__(self, agent, q, visits, epsilon, rng):
        self.agent, self.q, self.visits, self.epsilon, self.rng = agent, q, visits, epsilon, rng
        self.last = None  # (key of my previous decision, my points at that moment)

    def update(self, reward, best_next):
        """Apply the Q-learning rule to my previous decision."""
        if self.last is None:
            return
        key = self.last[0]
        self.visits[key] = self.visits.get(key, 0) + 1
        alpha = max(0.02, self.visits[key] ** -0.6)  # big steps at first, small steps once well explored
        self.q[key] = self.agent.q_update(self.q.get(key, 0.0), reward, best_next, alpha, GAMMA)

    def __call__(self, observation):
        agent = self.agent
        kinds = sorted({a["type"] for a in observation["actions"]})
        if len(kinds) == 1:  # no choice of action type, so nothing to learn here
            return agent.best_of_type(observation, kinds[0])
        features = agent.state_features(observation)
        keys = {k: agent.q_key(features, k) for k in kinds}
        points = observation["me"]["victory_points"]
        if self.last is not None:  # learn from what happened since my previous decision
            self.update(points - self.last[1], max(self.q.get(keys[k], 0.0) for k in kinds))
        if self.rng.random() < self.epsilon:  # explore
            kind = self.rng.choice(kinds)
            choice = agent.best_of_type(observation, kind)
        else:  # exploit what has been learned so far (baseline where nothing is known)
            choice = agent.q_policy(observation, self.q)
            kind = next(a["type"] for a in observation["actions"] if a["id"] == choice)
        self.last = (keys[kind], points)
        return choice


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"games": 0, "seconds": 0.0, "q": {}, "visits": {}}


def train(engine, games, checkpoint, time_limit):
    """Play one batch against the baseline and save. Run it again to continue."""
    data = load_checkpoint(checkpoint)
    start = time.time()
    agent, rival = load_agent(), load_agent()
    played = wins = 0
    for _ in range(games):
        if time.time() - start > time_limit:  # stop early, keeping everything learned
            break
        n = data["games"]  # global game number: seeds never repeat across batches
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * n / PLANNED_GAMES)
        learner = Learner(agent, data["q"], data["visits"], epsilon, random.Random(10 ** 6 + n))
        side = "A" if n % 2 == 0 else "B"  # learn from both seats
        policies = (learner, rival.heuristic_policy) if side == "A" else (rival.heuristic_policy, learner)
        winner, points, _ = play_one_game(engine, policies[0], policies[1], n + 1)
        result = 0.0 if winner is None else 1.0 if winner == side else -1.0
        if learner.last is not None:  # close the last step with the final reward
            learner.update(points[side] - learner.last[1] + result, 0.0)
        data["games"] += 1
        played += 1
        wins += result > 0
    data["seconds"] += time.time() - start
    with open(checkpoint + ".tmp", "w") as f:  # write fully, then rename: a stop cannot corrupt it
        json.dump(data, f)
    os.replace(checkpoint + ".tmp", checkpoint)
    print(f"batch: {played} games, learner won {wins / max(1, played):.1%} (while exploring) | total "
          f"{data['games']} games, {data['seconds'] / 60:.1f} min, {len(data['q'])} table entries", flush=True)


def export(checkpoint):
    """Write the well-visited entries into the 'Q = {...}  # <Q-TABLE>' line of the agent."""
    data = load_checkpoint(checkpoint)
    table = {k: round(v, 4) for k, v in sorted(data["q"].items()) if data["visits"].get(k, 0) >= MIN_VISITS}
    with open(AGENT_PATH) as f:
        lines = f.read().split("\n")
    index = next(i for i, line in enumerate(lines) if line.endswith("# <Q-TABLE>"))
    lines[index] = f"Q = {table!r}  # <Q-TABLE>"
    with open(AGENT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"exported {len(table)} of {len(data['q'])} entries (visited at least {MIN_VISITS} times) "
          f"after {data['games']} games")


# ------------------------------------------------------------------ evaluation
def duel(engine, name_a, name_b, games, seed0):
    """Win rate of version a against version b. Each board is played twice with seats swapped."""
    scores = []
    for i in range(games // 2):
        pair = 0.0
        for seat in "AB":
            a, b = load_agent(), load_agent()
            pa, pb = make_policy(a, name_a), make_policy(b, name_b)
            first, second = (pa, pb) if seat == "A" else (pb, pa)
            winner, _, _ = play_one_game(engine, first, second, seed0 + i)
            pair += 0.5 if winner is None else float(winner == seat)
        scores.append(pair / 2)
    mean = sum(scores) / len(scores)
    spread = math.sqrt(sum((x - mean) ** 2 for x in scores) / (len(scores) - 1))
    half = 1.96 * spread / math.sqrt(len(scores))  # half-width of the 95 % confidence interval
    print(f"{name_a:9s} vs {name_b:9s}: {mean:.1%} +/- {half:.1%} over {2 * len(scores)} games", flush=True)


def make_policy(agent, name):
    if name == "baseline":
        return agent.heuristic_policy
    if name == "qlearning":
        return agent.q_policy
    if name == "search":
        return lambda observation: agent.act(observation)["action"]
    rng = random.Random(7)  # "random": own generator, so the dice are not disturbed
    return lambda observation: rng.choice(observation["actions"])["id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "export", "evaluate"))
    parser.add_argument("--root", default=".", help="extracted ai_agent_tournament folder")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--time-limit", type=float, default=240.0, help="seconds per training batch")
    parser.add_argument("--checkpoint", default=os.path.join(HERE, CHECKPOINT))
    parser.add_argument("--pair", nargs=2, help="evaluate only this pair, e.g. qlearning baseline")
    parser.add_argument("--seed0", type=int, default=500000, help="first evaluation board; training uses 1, 2, 3...")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.root))
    from games.catan import engine
    if args.mode == "train":
        train(engine, args.games, args.checkpoint, args.time_limit)
    elif args.mode == "export":
        export(args.checkpoint)
    else:
        pairs = [args.pair] if args.pair else [("qlearning", "baseline"), ("search", "baseline"),
                                               ("search", "qlearning")]
        for a, b in pairs:
            duel(engine, a, b, args.games, args.seed0)


if __name__ == "__main__":
    main()