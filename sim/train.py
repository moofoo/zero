"""Self-play PPO for duels in zero's headless simulator (bot-ai ticket 42).

    .venv/bin/python train.py --envs 16 --steps 2000000 --run local1

The learner flies side 0 of every environment. Each episode's opponent (side 1) is, by turns: a frozen copy of the
latest learner (40%), a random earlier checkpoint (the league, 35%), or a scripted aimer (25%, --scripted-share; all
of them until checkpoints exist), which keeps self-play honest against close-range aimed fire. Before the league come
the aiming drills (DRILLS): a still target, a drifting one, then the aimer, each until it wins 80% (--no-drills skips). The policy
sees only the fair observation (duelsim.DuelEnv.fair: 150 ms old, what a player could see) and chooses turn, thrust
and fire ten times a second. The network: two layers of 256, a 256-unit GRU, a value head and three action heads.
--league-fixed keeps earlier runs' networks in the league for good, so the learner keeps beating old styles (each
picked in proportion to its weight, path:weight, 1 by default); with --league-fixed-greedy they play their likeliest
choices, as they're judged head to head, rather than sampling.
Every --eval-every iterations it plays the scripted aimer for a score that doesn't drift with the league.
With --dial (ticket 50), an aggression dial is an extra input, and the reward charges range and reversing and pays
for bullets in proportion to it.
With --clone (ticket 46's second stage), a network cloned from the player's recordings (clone.py) is an opponent for
--clone-share of the league's episodes, and with --clone-kl the learner is held near it: the loss adds that times the
KL divergence from the clone's choices to the learner's (or the reverse, --clone-kl-reverse; over --clone-kl-choices,
each weighted, name:weight), on the learner's own observations.
"""
import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from duelsim import DuelEnv, _Lib
from vecenv import VecEnv

HERE = Path(__file__).parent


class Policy(nn.Module):
    """`extra` inputs follow the fair observation: with --dial, one, the aggression dial (0 to 1)."""

    def __init__(self, obs_size, hidden=256, extra=0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_size + extra, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.extra = extra
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, k) for k in (5, 3, 3)])  # turn, thrust, fire (DuelEnv.choice)
        self.value = nn.Linear(hidden, 1)
        self.hidden = hidden

    def initial(self, n):
        return torch.zeros(1, n, self.hidden)

    def forward(self, obs, h, starts=None):
        """obs: (batch, time, obs); h: (1, batch, hidden); starts: (batch, time), 1 where an episode starts (the
        hidden state is cleared there). Returns logits per head (batch, time, 3), values (batch, time), and h."""
        x = self.body(obs)
        outs = []
        for t in range(x.shape[1]):
            if starts is not None:
                h = h * (1.0 - starts[:, t]).view(1, -1, 1)
            y, h = self.gru(x[:, t:t + 1], h)
            outs.append(y)
        y = torch.cat(outs, 1)
        return [head(y) for head in self.heads], self.value(y).squeeze(-1), h


def load_policy(path, extra=None, device="cpu"):
    """A checkpoint as a Policy. Its extra inputs come from the checkpoint; with `extra`, it gets that many, the new
    ones with zero weights, so it plays as before until trained (a policy from before the dial gains one)."""
    state = torch.load(path, map_location=device, weights_only=True)
    fair_size = _Lib.get().ds_fair_size()
    had = state["body.0.weight"].shape[1] - fair_size
    extra = had if extra is None else extra
    if extra < had:
        raise ValueError(f"{path} has {had} extra inputs, more than {extra}")
    if extra > had:
        w = state["body.0.weight"]
        state["body.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra - had, dtype=w.dtype)], 1)
    policy = Policy(fair_size, extra=extra).to(device)
    policy.load_state_dict(state)
    return policy


def with_dial(obs, dial):
    """The policy's input: the fair observations (n, fair) with the dial (n,) after them; None leaves them as they are."""
    return obs if dial is None else torch.cat([obs, dial.unsqueeze(1)], 1)


class ScriptedAimer:
    """Turns toward the enemy it sees, holds about 12 tiles (braking when closing fast), and fires when lined up;
    otherwise wanders. Uses only the fair observation's own and opponent fields. It sees its own heading 1.5 steps
    late, so it allows for the turns it has made since, as a player allows for lag (a step turns about 0.45 rad, a tap
    about 0.14), and it leads its shots."""
    TURN_RATE = 0.45
    TURN_AMOUNT = (-1.0, -0.3, 0.0, 0.3, 1.0)  # each turn choice's share of a full step's turn
    BULLET_SPEED = 12.5  # tiles a second, the Warbird's in the duel profile
    DODGE_SECONDS, DODGE_TILES = 3.0, 7.0  # with dodge, bombs passing this near this soon are dodged

    # Styles: "hold" keeps about 12 tiles; "rush" closes to point-blank and fires bullets, as Veteran does when it
    # commits; "retreat" holds but, below half energy, turns and flies off until it has recharged, then comes back;
    # "punish" (ticket 51) holds, and rushes while its target's energy (target_energy, set by the caller each step
    # from the target's own view: what a player would infer from the shots it fired) is below PUNISH_ENERGY.
    PUNISH_ENERGY = 0.4

    def __init__(self, rng, dodge=False, style="hold"):
        self.rng = rng
        self.dodge = dodge
        self.style = style
        self.fleeing = False
        self.target_energy = 1.0
        self.turns = [0.0, 0.0]  # the last two steps' turns (TURN_AMOUNT), latest last

    def reset(self):
        self.turns = [0.0, 0.0]
        self.fleeing = False

    def __call__(self, obs):
        turn, thrust, fire = self.choose(obs)
        if self.dodge:
            turn, thrust = self.dodge_bombs(obs, turn, thrust)
        self.turns = [self.turns[1], self.TURN_AMOUNT[turn]]
        return turn, thrust, fire

    def dodge_bombs(self, obs, turn, thrust):
        """With dodge set: the soonest enemy bomb to pass within DODGE_TILES in the next DODGE_SECONDS is dodged.
        A ship thrusts only along its heading, so it thrusts to move its closest approach away along that axis; a bomb
        coming along the axis can't be dodged that way, so it turns side-on first."""
        worst = None
        for k in range(16):
            w = obs[16 + 8 * k:24 + 8 * k]
            if w[7] < 0.5 or w[5] < 0.5:  # not present, or not a bomb
                continue
            px, py, vx, vy = w[0] * 60.0, w[1] * 60.0, w[2] * 40.0, w[3] * 40.0
            speed2 = vx * vx + vy * vy
            if speed2 < 1e-6:
                continue
            t = -(px * vx + py * vy) / speed2
            if not 0.0 < t < self.DODGE_SECONDS:
                continue
            cx, cy = px + vx * t, py + vy * t
            miss = math.hypot(cx, cy)
            if miss < self.DODGE_TILES and (worst is None or t < worst[0]):
                worst = (t, cx, cy, vx, vy)
        if worst is None:
            return turn, thrust
        _, cx, cy, vx, vy = worst
        if abs(vx) > 2.0 * abs(vy):  # along the ship's axis: turn side-on and go
            return (0 if cy >= 0 else 4), 0
        # Moving the ship back pushes the bomb's closest approach ahead, and the other way.
        return turn, (2 if cx >= 0 else 0)

    def choose(self, obs):
        rng = self.rng
        if not (obs[8] > 0.5 or obs[9] > 0.5):
            return 4 if rng.random() < 0.5 else 2, 0 if rng.random() < 0.5 else 1, 0
        # In the ship's frame: x ahead, y to the right.
        dx, dy = obs[10] * 60.0, obs[11] * 60.0
        dvx, dvy = obs[12] * 20.0, obs[13] * 20.0
        dist = math.hypot(dx, dy) or 1.0
        # Where the enemy is now (the view is 0.15 s old), then where a bullet meets it: bullets carry the ship's own
        # velocity, so the lead is on the velocity relative to the ship.
        px, py = dx + dvx * 0.15, dy + dvy * 0.15
        a = dvx * dvx + dvy * dvy - self.BULLET_SPEED ** 2
        b = 2.0 * (px * dvx + py * dvy)
        c = px * px + py * py
        disc = b * b - 4.0 * a * c
        if a < 0 and disc >= 0:
            t = (-b - math.sqrt(disc)) / (2.0 * a)
            px, py = px + dvx * t, py + dvy * t
        angle = math.atan2(py, px)
        if self.style == "retreat":
            if obs[4] < 0.5:
                self.fleeing = True
            elif obs[4] > 0.9:
                self.fleeing = False
        if self.fleeing:  # face away from the enemy and go
            angle = math.atan2(-dy, -dx)
        angle -= self.TURN_RATE * (self.turns[1] + 0.5 * self.turns[0])
        # Hold the turn when far off, tap it when close (a tap is about 0.14 rad), and leave it within half a tap.
        if abs(angle) < 0.07:
            turn = 2
        elif abs(angle) < 0.3:
            turn = 3 if angle > 0 else 1
        else:
            turn = 4 if angle > 0 else 0
        closing = -(dx * obs[12] + dy * obs[13]) * 20.0 / dist
        if self.fleeing:
            return turn, (0 if abs(angle) < 1.0 else 1), 0
        rushing = self.style == "rush" or (self.style == "punish" and self.target_energy < self.PUNISH_ENERGY)
        if rushing:
            thrust = 2 if dist < 2 else (0 if dist > 4 and closing < 12 else 1)
        elif dist > 18:
            thrust = 0 if closing < 8 else 1
        elif dist < 8 or closing > 4:
            thrust = 2
        else:
            thrust = 1
        fire = 0
        if abs(angle) < 0.15:
            fire = 2 if dist > 10 and not rushing and rng.random() < 0.25 else 1
        return turn, thrust, fire


class Drifter:
    """A drill target that flies about without firing: it holds a random turn and thrust for 0.5 to 2 seconds."""

    def __init__(self, rng):
        self.rng = rng
        self.held, self.left = (2, 1, 0), 0

    def reset(self):
        self.left = 0

    def __call__(self, obs):
        if self.left <= 0:
            self.held = (self.rng.choice((0, 2, 2, 4)), self.rng.choice((0, 0, 1, 2)), 0)
            self.left = self.rng.randint(5, 20)
        self.left -= 1
        return self.held


# The aiming drills, in order: a ship that sits still, one that drifts, then the scripted aimer. The learner moves to
# the next when it wins --drill-pass of its last --drill-window episodes against the current one, and after the last
# to the league.
DRILLS = ("still", "drift", "scripted")
# The scripted opponents a run can include (--scripted-kinds), by their ScriptedAimer style.
SCRIPTED = {"scripted": "hold", "rusher": "rush", "retreater": "retreat", "punisher": "punish"}


def batch(arrays):
    """Observations (numpy arrays, one an environment) as one tensor on the default device."""
    return torch.from_numpy(np.stack(arrays)).to(torch.get_default_device())


def act(policy, obs, h, starts, greedy=False):
    with torch.no_grad():
        logits, value, h = policy(obs.unsqueeze(1), h, starts.unsqueeze(1))
    choices, logp = [], torch.zeros(obs.shape[0])
    for lg in logits:
        dist = torch.distributions.Categorical(logits=lg[:, 0])
        c = dist.probs.argmax(-1) if greedy else dist.sample()
        choices.append(c)
        logp = logp + dist.log_prob(c)
    return torch.stack(choices, -1), logp, value[:, 0], h


def evaluate(policy, envs, episodes, rng, dodge=False, style="hold", dial=0.0):
    """Greedy learner (at `dial`, if it has one) against the scripted aimer (one that dodges bombs, with dodge; or a
    zero bot, in bot environments): wins, losses, draws over `episodes`."""
    wins = losses = draws = 0
    n = envs.count
    fair_a, opp = envs.reset()
    obs = batch(fair_a)
    h = policy.initial(n)
    starts = torch.ones(n)
    aimers = [ScriptedAimer(rng, dodge, style) for _ in range(n)]
    dials = torch.full((n,), dial) if policy.extra else None
    done_count = 0
    while done_count < episodes:
        choice, _, _, h = act(policy, with_dial(obs, dials), h, starts, greedy=True)
        starts = torch.zeros(n)
        chosen = choice.tolist()
        actions_a = [DuelEnv.choice(*chosen[i]) for i in range(n)]
        energy = obs[:, 4].tolist()  # the learner's energy share, for a punisher
        for i in range(n):
            aimers[i].target_energy = energy[i]
        actions_b = [DuelEnv.choice(*aimers[i](opp[i])) for i in range(n)]
        steps = envs.step(actions_a, actions_b)
        for i, (_, done, info, _, _) in enumerate(steps):
            if done:
                done_count += 1
                dead = {d["side"] for d in info["deaths"]}
                wins += 1 in dead and 0 not in dead
                losses += 0 in dead and 1 not in dead
                draws += not dead or len(dead) == 2
                aimers[i].reset()
                starts[i] = 1.0
        obs, opp = batch([r[3] for r in steps]), [r[4] for r in steps]
    return wins, losses, draws


def shaping(args, o, thrust, hits, style, dial):
    """The dial's reward at full dial (ticket 50), from the learner's observation `o` it chose from, its thrust choice
    and the step's hits: a cost for range beyond --range-tiles, one for reversing while facing the enemy, and a bonus
    for bullet damage dealt. Counts the step in `style` by dial band (below a third, above two thirds)."""
    facing = False
    if o[8] > 0.5 or o[9] > 0.5:  # on screen or radar
        ahead, right = o[10] * 60.0, o[11] * 60.0
        excess = min(max(math.hypot(ahead, right) / args.range_tiles - 1.0, 0.0), 1.0)
        facing = ahead > 0 and abs(math.atan2(right, ahead)) < math.pi / 3
    else:
        excess = 1.0
    reversing = thrust == 2 and facing
    band = "low" if dial < 1 / 3 else "high" if dial > 2 / 3 else None
    if band:
        style[band] += 1
        style[band + "_beyond"] += excess > 0
        style[band + "_reverse"] += reversing
    bullets = sum(x["share"] for x in hits if x["side"] == 1 and x["class"] == "bullet" and not x["own"])
    return -args.range_cost * excess - (args.reverse_cost if reversing else 0.0) + args.bullet_bonus * bullets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--workers", type=int, default=1, help="processes stepping the environments (one a core)")
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--init", default=None, help="a checkpoint to start from")
    p.add_argument("--dodge", action="store_true", help="the scripted opponents dodge bombs")
    p.add_argument("--scripted-kinds", default="scripted", help="the scripted opponents in the league's scripted "
                   "share, by turns: scripted (holds about 12 tiles), rusher, retreater, punisher (rushes a learner low on "
                   "energy) (SCRIPTED); a kind named twice comes up twice as often")
    p.add_argument("--time-cost", type=float, default=0.0, help="taken from the learner's reward each step")
    p.add_argument("--draw-penalty", type=float, default=0.0, help="taken from the learner's reward at a timeout")
    p.add_argument("--dial", action="store_true", help="the aggression dial (ticket 50): each episode, the learner and "
                   "policy opponents get a dial between 0 and 1 as an input, and the learner's reward has, times its "
                   "dial, the costs and bonus below; a policy from before the dial (--init) gains the input")
    p.add_argument("--range-tiles", type=float, default=15.0, help="with --dial: beyond this the range cost starts")
    p.add_argument("--range-cost", type=float, default=0.004, help="with --dial: taken each step at twice --range-tiles "
                   "or more (or with the enemy out of sight), in proportion between")
    p.add_argument("--reverse-cost", type=float, default=0.003, help="with --dial: taken each step reversing with the "
                   "enemy within 60 degrees ahead")
    p.add_argument("--bullet-bonus", type=float, default=0.5, help="with --dial: bullet damage dealt is worth this much "
                   "more")
    p.add_argument("--dial-ends", type=float, default=0.0, help="with --dial: the share of episodes whose dials are 0 "
                   "or 1 (either, evenly) rather than anywhere between, for a clearer contrast")
    p.add_argument("--shot-cost", type=float, default=0.0, help="ticket 51: each shot the learner fires costs this "
                   "times its energy as a share of full energy (0.5: about 0.018 a bullet, 0.10 a bomb)")
    p.add_argument("--eval-dials", default="0,1", help="with --dial: the dials the bot evaluations play at")
    p.add_argument("--league-fixed", default="", help="earlier runs' networks kept in the league for good, comma-separated "
                   "(with or without the dial), so the learner keeps beating old styles: runs/v6-panel/final.pt,...; path:weight "
                   "plays one more often (weights 1 by default)")
    p.add_argument("--league-fixed-share", type=float, default=0.3, help="with --league-fixed: the share of the league's "
                   "random-checkpoint episodes played against one of them")
    p.add_argument("--league-fixed-greedy", action="store_true", help="the --league-fixed networks play their likeliest "
                   "choices rather than sampling (sampled, v6 is far weaker than greedy)")
    p.add_argument("--low-energy-cost", type=float, default=0.0, help="ticket 46: each step the learner is below half "
                   "energy costs this times its shortfall from half, as a share of half (1 at empty)")
    p.add_argument("--clone", default=None, help="ticket 46: a clone.py network of the player (no dial)")
    p.add_argument("--clone-share", type=float, default=0.0, help="with --clone: the share of the league's episodes "
                   "against it (sampling its choices, as a player varies)")
    p.add_argument("--clone-kl", type=float, default=0.0, help="with --clone: the loss adds this times KL(clone || "
                   "learner), summed over --clone-kl-choices, averaged over steps")
    p.add_argument("--clone-kl-choices", default="turn,thrust,fire", help="the choices the KL counts (turn is aim), "
                   "each optionally weighted, name:weight (fire:5 holds the firing five times as hard)")
    p.add_argument("--clone-kl-reverse", action="store_true", help="KL(learner || clone) instead: the learner may "
                   "settle on any of the clone's likely choices, rather than spreading over all of them as it does")
    p.add_argument("--no-drills", action="store_true", help="start in the league, without the aiming drills")
    p.add_argument("--drill-pass", type=float, default=0.8)
    p.add_argument("--drill-window", type=int, default=100)
    p.add_argument("--bot-share", type=float, default=0.0, help="the share of environments whose opponent is a zero bot "
                   "(ticket 45), always; the rest play the league")
    p.add_argument("--bot-skill", default="veteran", help="the zero bots' skill level (bots/chaos.cfg)")
    p.add_argument("--bot-specs", default="", help="several bot opponents instead, as skill:aggression pairs spread "
                   "over the bot environments in turn, each also evaluated: veteran:standard,elite:aggressive")
    p.add_argument("--lr-final", type=float, default=None, help="the learning rate falls in a straight line to this "
                   "by the last step (none: it stays --lr)")
    p.add_argument("--scripted-share", type=float, default=0.25, help="the share of episodes against the aimer")
    p.add_argument("--spawn-gap", type=float, nargs=2, default=(20.0, 40.0), help="tiles between the ships at spawn")
    p.add_argument("--spawn-center", type=float, nargs=2, default=(512.0, 512.0),
                   help="where the ships spawn around (tiles); the duel box: 186 144 with --spawn-radius 40")
    p.add_argument("--spawn-radius", type=float, default=60.0)
    p.add_argument("--run", default="local")
    p.add_argument("--device", default="cpu")
    p.add_argument("--map-dir", default=None, help="the folder holding zones/Chaos/<map>.lvl")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cpu":
        torch.set_default_device(device)  # every tensor below, and the networks, live on the device
    bot_count = int(round(args.bot_share * args.envs))
    pairs = [x.split(":") for x in args.bot_specs.split(",") if x] or [[args.bot_skill, "standard"]]
    bot_specs = [{"skill": p[0], "aggression": p[1] if len(p) > 1 else "standard"} for p in pairs]
    bot_names = [f"{b['skill']}-{b['aggression']}" if len(bot_specs) > 1 else b["skill"] for b in bot_specs]
    envs = VecEnv(args.envs, workers=args.workers, seed=args.seed * 1000, map_dir=args.map_dir, bots=bot_count,
                  bot=bot_specs,
                  spawn_gap=tuple(args.spawn_gap), spawn_center=tuple(args.spawn_center),
                  spawn_radius=args.spawn_radius, max_seconds=args.max_seconds)
    eval_envs = VecEnv(8, seed=args.seed * 1000 + 500, map_dir=args.map_dir, max_seconds=args.max_seconds)
    eval_bot_envs = [VecEnv(4, seed=args.seed * 1000 + 700 + 10 * k, map_dir=args.map_dir, max_seconds=args.max_seconds,
                            spawn_gap=tuple(args.spawn_gap), spawn_center=tuple(args.spawn_center),
                            spawn_radius=args.spawn_radius, bots=4, bot=spec)
                     for k, spec in enumerate(bot_specs)] if bot_count else []
    size = envs.fair_size
    extra = 1 if args.dial else 0
    policy = load_policy(args.init, extra, device) if args.init else Policy(size, extra=extra).to(device)
    opponent = Policy(size, extra=extra).to(device)
    opponent.load_state_dict(policy.state_dict())
    clone = load_policy(args.clone, 0, device) if args.clone else None
    clone_h = clone.initial(args.envs) if clone else None
    clone_log = collections.Counter()  # ticket 46, for the log: the KL from the clone, and how often they'd agree
    kl_weights = {name: float(w or 1) for name, _, w in (x.partition(":") for x in args.clone_kl_choices.split(","))}
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    league = []  # checkpoint paths
    fixed_weighted = [x.partition(":") for x in args.league_fixed.split(",") if x]
    fixed = [path for path, _, _ in fixed_weighted]  # earlier runs' networks, always in the league
    fixed_weights = [float(w or 1) for _, _, w in fixed_weighted]
    fixed_names = {path: "fixed-" + Path(path).parent.name for path in fixed}

    n = args.envs
    kinds_scripted = args.scripted_kinds.split(",")
    scripted = {k: [ScriptedAimer(rng, args.dodge, SCRIPTED[k]) for _ in range(n)] for k in SCRIPTED}
    aimers = scripted["scripted"]
    drifters = [Drifter(rng) for _ in range(n)]
    stage = len(DRILLS) if args.no_drills else 0
    stage_results = []
    fair_a, fair_b = envs.reset()
    obs, opp_obs = batch(fair_a), batch(fair_b)
    rows = fair_a  # the learner's observations as arrays, for the dial's costs
    opp_rows = fair_b  # the opponents' observations as arrays, for the scripted aimer
    # Each episode's dials (ticket 50), the learner's and its policy opponent's; None without --dial.
    def draw_dial():
        return float(rng.random() < 0.5) if rng.random() < args.dial_ends else rng.random()

    dial = torch.tensor([draw_dial() for _ in range(n)], dtype=torch.float32) if args.dial else None
    opp_dial = torch.tensor([draw_dial() for _ in range(n)], dtype=torch.float32) if args.dial else None
    eval_dials = [float(x) for x in args.eval_dials.split(",")] if args.dial else [0.0]
    style = collections.Counter()  # the dial's effect, for the log: steps beyond range and reversing, by dial
    energy_log = collections.Counter()  # ticket 51, for the log: the learner's steps below half energy, its shots
    h = policy.initial(n)
    opp_h = opponent.initial(n)
    starts = torch.ones(n)
    league_nets = {}

    def pick_opponent(i):
        if i < bot_count:
            return bot_names[i % len(bot_names)]  # a bot environment: its zero bot, always
        if stage < len(DRILLS):
            return DRILLS[stage]
        r = rng.random()
        if not (league or fixed) or r < args.scripted_share:
            return rng.choice(kinds_scripted)
        if clone is not None and r < args.scripted_share + args.clone_share:
            return "clone"
        # The rest: a bit over half against the latest, the others against a random checkpoint or a fixed network.
        rest = args.scripted_share + (args.clone_share if clone is not None else 0.0)
        if r < rest + 0.55 * (1 - rest):
            return "latest"
        if fixed and (not league or rng.random() < args.league_fixed_share):
            return rng.choices(fixed, fixed_weights)[0]
        return rng.choice(league)

    # Each environment's opponent for its current episode: a drill, "scripted", "latest" or a checkpoint path.
    kinds = [pick_opponent(i) for i in range(n)]

    def opponent_net(kind):
        if kind == "latest":
            return opponent
        if kind == "clone":
            return clone
        if kind in fixed and kind not in league_nets:
            league_nets[kind] = load_policy(kind, device=device)  # its own inputs: a policy from before the dial has none
        if kind not in league_nets:
            net = Policy(size, extra=extra)
            net.load_state_dict(torch.load(kind, map_location=device, weights_only=True))
            league_nets[kind] = net
        return league_nets[kind]

    log = open(out / "log.jsonl", "a")
    # The run's settings first, so a log says how it was trained (a record without "steps").
    log.write(json.dumps({"settings": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    log.flush()
    total = 0
    iteration = 0
    episode_rewards, results = [], []
    running = [0.0] * n
    started = time.time()
    while total < args.steps:
        iteration += 1
        if args.lr_final is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr + (args.lr_final - args.lr) * min(total / args.steps, 1.0)
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done, buf_start = [], [], [], [], [], [], []
        buf_clone = []  # the clone's log-probabilities for each choice, on the learner's observations
        h0 = h.clone()
        for t in range(args.rollout):
            inputs = with_dial(obs, dial)
            choice, logp, value, h = act(policy, inputs, h, starts)
            if clone is not None and args.clone_kl:
                with torch.no_grad():
                    lg, _, clone_h = clone(obs.unsqueeze(1), clone_h, starts.unsqueeze(1))
                buf_clone.append(torch.cat([torch.log_softmax(x[:, 0], -1) for x in lg], -1))
            # The opponents: group environments by opponent kind for batched play.
            opp_choice = [None] * n
            groups = {}
            for i, k in enumerate(kinds):
                groups.setdefault(k, []).append(i)
            for k, idx in groups.items():
                if k == "still" or (k in bot_names and idx[0] < bot_count):  # a bot flies itself
                    for i in idx:
                        opp_choice[i] = (2, 1, 0)
                elif k == "drift":
                    for i in idx:
                        opp_choice[i] = drifters[i](None)
                elif k in SCRIPTED:
                    for i in idx:
                        scripted[k][i].target_energy = float(rows[i][4])  # the learner's energy, for a punisher
                        opp_choice[i] = scripted[k][i](opp_rows[i])
                else:
                    net = opponent_net(k)
                    c, _, _, nh = act(net, with_dial(opp_obs[idx], opp_dial[idx] if net.extra else None),
                                      opp_h[:, idx], starts[idx], greedy=args.league_fixed_greedy and k in fixed)
                    opp_h[:, idx] = nh
                    for j, row in zip(idx, c.tolist()):
                        opp_choice[j] = tuple(row)
            rewards, dones = torch.zeros(n), torch.zeros(n)
            new_obs, new_opp = [], []
            buf_start.append(starts.clone())
            starts = torch.zeros(n)
            chosen = choice.tolist()
            steps = envs.step([DuelEnv.choice(*chosen[i]) for i in range(n)],
                              [DuelEnv.choice(*opp_choice[i]) for i in range(n)])
            dials = dial.tolist() if args.dial else None
            for i, (ra, done, info, fair_a, fair_b) in enumerate(steps):
                ra -= args.time_cost
                if args.dial:
                    ra += dials[i] * shaping(args, rows[i], chosen[i][1], info["hits"], style, dials[i])
                # Ticket 51: each shot fired costs a share of its energy, so a shot pays only when it's likely to hit.
                spent = sum(x["share"] for x in info["shots"] if x["side"] == 0)
                ra -= args.shot_cost * spent
                ra -= args.low_energy_cost * max(0.0, 0.5 - float(rows[i][4])) / 0.5
                energy_log["steps"] += 1
                energy_log["below_half"] += rows[i][4] < 0.5
                fired = sum(1 for x in info["shots"] if x["side"] == 0)
                energy_log["shots"] += fired
                energy_log["shots_below_half"] += fired if rows[i][4] < 0.5 else 0
                if done and not info["deaths"]:
                    ra -= args.draw_penalty
                rewards[i] = ra
                running[i] += ra
                if done:
                    dones[i] = 1.0
                    dead = {d["side"] for d in info["deaths"]}
                    won = 1 in dead and 0 not in dead
                    results.append((kinds[i], won, 0 in dead and 1 not in dead))
                    if stage < len(DRILLS) and kinds[i] == DRILLS[stage]:
                        stage_results.append(won)
                        recent = stage_results[-args.drill_window:]
                        if len(recent) >= args.drill_window and sum(recent) / len(recent) >= args.drill_pass:
                            print(json.dumps({"passed": DRILLS[stage], "iteration": iteration, "steps": total}),
                                  flush=True)
                            log.write(json.dumps({"passed": DRILLS[stage], "iteration": iteration, "steps": total}) + "\n")
                            stage += 1
                            stage_results = []
                    episode_rewards.append(running[i])
                    running[i] = 0.0
                    starts[i] = 1.0
                    kinds[i] = pick_opponent(i)
                    if args.dial:
                        dial[i], opp_dial[i] = draw_dial(), draw_dial()
                    opp_h[:, i] = 0.0
                    for k in SCRIPTED:
                        scripted[k][i].reset()
                    drifters[i].reset()
                new_obs.append(fair_a)
                new_opp.append(fair_b)
            buf_obs.append(inputs)
            buf_act.append(choice)
            buf_logp.append(logp)
            buf_val.append(value)
            buf_rew.append(rewards)
            buf_done.append(dones)
            obs, opp_obs = batch(new_obs), batch(new_opp)
            rows, opp_rows = new_obs, new_opp
            total += n
        # Advantages (GAE), bootstrapped from the next value.
        with torch.no_grad():
            _, last_value, _ = policy(with_dial(obs, dial).unsqueeze(1), h, starts.unsqueeze(1))
            last_value = last_value[:, 0]
        T = args.rollout
        adv = torch.zeros(T, n)
        gae = torch.zeros(n)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else buf_val[t + 1]
            nonterminal = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * next_value * nonterminal - buf_val[t]
            gae = delta + args.gamma * args.lam * nonterminal * gae
            adv[t] = gae
        values = torch.stack(buf_val)
        returns = adv + values
        # (time, env) to (env, time) sequences, cut into chunks with their starting hidden states.
        O = torch.stack(buf_obs).transpose(0, 1)
        A = torch.stack(buf_act).transpose(0, 1)
        LP = torch.stack(buf_logp).transpose(0, 1)
        ADV = adv.transpose(0, 1)
        RET = returns.transpose(0, 1)
        S = torch.stack(buf_start).transpose(0, 1)
        CL = torch.stack(buf_clone).transpose(0, 1) if buf_clone else None
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        # The hidden state at each chunk's start, from a replay of the rollout with the policy before updating.
        with torch.no_grad():
            hs = []
            hh = h0
            for c0 in range(0, T, args.chunk):
                hs.append(hh)
                _, _, hh = policy(O[:, c0:c0 + args.chunk], hh, S[:, c0:c0 + args.chunk])
        for _ in range(args.epochs):
            for ci, c0 in enumerate(range(0, T, args.chunk)):
                sl = slice(c0, c0 + args.chunk)
                logits, value, _ = policy(O[:, sl], hs[ci], S[:, sl])
                logp, entropy = 0.0, 0.0
                for k, lg in enumerate(logits):
                    dist = torch.distributions.Categorical(logits=lg)
                    logp = logp + dist.log_prob(A[:, sl, k])
                    entropy = entropy + dist.entropy()
                ratio = torch.exp(logp - LP[:, sl])
                surrogate = torch.min(ratio * ADV[:, sl], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ADV[:, sl])
                loss = -surrogate.mean() + 0.5 * ((value - RET[:, sl]) ** 2).mean() - args.entropy * entropy.mean()
                if CL is not None:
                    # Held near the player: the KL between the clone's and the learner's choices, the clone's columns
                    # in head order.
                    kl, at, agree = 0.0, 0, 0.0
                    for name, lg in zip(("turn", "thrust", "fire"), logits):
                        lp = CL[:, sl, at:at + lg.shape[-1]]
                        at += lg.shape[-1]
                        agree = agree + (lg.argmax(-1) == lp.argmax(-1)).float().mean().item()
                        if name not in kl_weights:
                            continue
                        lq = torch.log_softmax(lg, -1)
                        if args.clone_kl_reverse:
                            part = (lq.exp() * (lq - lp)).sum(-1)
                        else:
                            part = (lp.exp() * (lp - lq)).sum(-1)
                        kl = kl + kl_weights[name] * part
                        clone_log[f"kl_{name}"] += part.mean().item()
                    loss = loss + args.clone_kl * kl.mean()
                    clone_log["kl"] += kl.mean().item()
                    clone_log["agree"] += agree / len(logits)
                    clone_log["chunks"] += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        opponent.load_state_dict(policy.state_dict())
        if iteration % args.save_every == 0:
            path = out / f"ckpt_{iteration:05d}.pt"
            torch.save(policy.state_dict(), path)
            league.append(str(path))
        record = {"iteration": iteration, "steps": total, "seconds": round(time.time() - started),
                  "episode_reward": round(sum(episode_rewards[-100:]) / max(len(episode_rewards[-100:]), 1), 3)}
        record["stage"] = DRILLS[stage] if stage < len(DRILLS) else "league"
        for band in ("low", "high"):
            if style[band]:
                record[f"dial_{band}"] = {"beyond": round(style[band + "_beyond"] / style[band], 2),
                                          "reversing": round(style[band + "_reverse"] / style[band], 2)}
        style.clear()
        if energy_log["steps"]:
            record["energy"] = {"below_half": round(energy_log["below_half"] / energy_log["steps"], 3),
                                "shots_a_minute": round(600 * energy_log["shots"] / energy_log["steps"], 1),
                                "shots_below_half": round(energy_log["shots_below_half"] / max(energy_log["shots"], 1), 3)}
        energy_log.clear()
        if clone_log["chunks"]:
            record["clone"] = {"kl": round(clone_log["kl"] / clone_log["chunks"], 3),
                               "agree": round(clone_log["agree"] / clone_log["chunks"], 3),
                               **{k: round(clone_log[k] / clone_log["chunks"], 3) for k in clone_log
                                  if k.startswith("kl_")}}
        clone_log.clear()
        bot_kinds = tuple(bot_names) if bot_count else ()
        for kind in ("still", "drift") + tuple(SCRIPTED) + bot_kinds + ("clone", "latest", "league") + tuple(fixed):
            rs = [r for r in results[-300:]
                  if (r[0] == kind if kind != "league"
                      else r[0] not in DRILLS + tuple(SCRIPTED) + ("clone", "latest") + bot_kinds + tuple(fixed))]
            if rs:
                record[f"vs_{fixed_names.get(kind, kind)}"] = {
                    "episodes": len(rs), "won": round(sum(r[1] for r in rs) / len(rs), 2),
                    "lost": round(sum(r[2] for r in rs) / len(rs), 2)}
        if iteration % args.eval_every == 0:
            w, l, d = evaluate(policy, eval_envs, args.eval_episodes, rng)
            wd, ld, dd = evaluate(policy, eval_envs, args.eval_episodes, rng, dodge=True)
            record["eval_vs_dodger"] = {"won": wd, "lost": ld, "draw": dd}
            for name, bot_envs in zip(bot_names, eval_bot_envs):
                for d in eval_dials:
                    wb, lb, db = evaluate(policy, bot_envs, args.eval_episodes, rng, dial=d)
                    record[f"eval_vs_{name}" + (f"@{d:g}" if args.dial else "")] = {"won": wb, "lost": lb, "draw": db}
            for k in kinds_scripted:
                if k != "scripted":
                    wk, lk, dk = evaluate(policy, eval_envs, args.eval_episodes, rng, style=SCRIPTED[k])
                    record[f"eval_vs_{k}"] = {"won": wk, "lost": lk, "draw": dk}
            record["eval_vs_aimer"] = {"won": w, "lost": l, "draw": d}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    torch.save(policy.state_dict(), out / "final.pt")
    envs.close()
    for bot_envs in eval_bot_envs:
        bot_envs.close()


if __name__ == "__main__":
    main()
