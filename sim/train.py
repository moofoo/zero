"""Self-play PPO for duels in zero's headless simulator (bot-ai ticket 42).

    .venv/bin/python train.py --envs 16 --steps 2000000 --run local1

The learner flies side 0 of every environment. Each episode's opponent (side 1) is, by turns: a frozen copy of the
latest learner, a random earlier checkpoint (the league), or, until checkpoints exist, a scripted aimer. The policy
sees only the fair observation (duelsim.DuelEnv.fair: 150 ms old, what a player could see) and chooses turn, thrust
and fire ten times a second. The network: two layers of 256, a 256-unit GRU, a value head and three action heads.
Every --eval-every iterations it plays the scripted aimer for a score that doesn't drift with the league.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from duelsim import DuelEnv

HERE = Path(__file__).parent


class Policy(nn.Module):
    def __init__(self, obs_size, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(3)])  # turn, thrust, fire
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


class ScriptedAimer:
    """Turns toward the enemy it sees, holds about 12 tiles (braking when closing fast), and fires when lined up;
    otherwise wanders. Uses only the fair observation's own and opponent fields. It sees its own heading 1.5 steps
    late, so it allows for the turns it has made since, as a player allows for lag (a step turns about 0.45 rad), and it
    leads its shots."""
    TURN_RATE = 0.45
    BULLET_SPEED = 12.5  # tiles a second, the Warbird's in the duel profile

    def __init__(self, rng):
        self.rng = rng
        self.turns = [0, 0]  # the last two steps' turns, latest last: -1 left, 0 none, 1 right

    def reset(self):
        self.turns = [0, 0]

    def __call__(self, obs):
        turn, thrust, fire = self.choose(obs)
        self.turns = [self.turns[1], turn - 1]
        return turn, thrust, fire

    def choose(self, obs):
        rng = self.rng
        hx, hy = obs[2], obs[3]
        if not (obs[8] > 0.5 or obs[9] > 0.5):
            return 2 if rng.random() < 0.5 else 1, 0 if rng.random() < 0.5 else 1, 0
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
        angle = math.atan2(hx * py - hy * px, hx * px + hy * py)
        angle -= self.TURN_RATE * (self.turns[1] + 0.5 * self.turns[0])
        turn = 1 if abs(angle) < 0.25 else (2 if angle > 0 else 0)
        closing = -(dx * obs[12] + dy * obs[13]) * 20.0 / dist
        if dist > 18:
            thrust = 0 if closing < 8 else 1
        elif dist < 8 or closing > 4:
            thrust = 2
        else:
            thrust = 1
        fire = 0
        if abs(angle) < 0.3:
            fire = 2 if dist > 10 and rng.random() < 0.25 else 1
        return turn, thrust, fire


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


def evaluate(policy, envs, episodes, rng):
    """Greedy learner against the scripted aimer: wins, losses, draws over `episodes`."""
    wins = losses = draws = 0
    n = len(envs)
    for e in envs:
        e.reset()
    obs = torch.tensor([e.fair(0) for e in envs])
    opp = [e.fair(1) for e in envs]
    h = policy.initial(n)
    starts = torch.ones(n)
    aimers = [ScriptedAimer(rng) for _ in envs]
    done_count = 0
    while done_count < episodes:
        choice, _, _, h = act(policy, obs, h, starts, greedy=True)
        starts = torch.zeros(n)
        new_obs, new_opp = [], []
        for i, e in enumerate(envs):
            a = DuelEnv.choice(*[int(x) for x in choice[i]])
            b = DuelEnv.choice(*aimers[i](opp[i]))
            _, _, _, done, info = e.step(a, b)
            if done:
                done_count += 1
                dead = {d["side"] for d in info["deaths"]}
                wins += 1 in dead and 0 not in dead
                losses += 0 in dead and 1 not in dead
                draws += not dead or len(dead) == 2
                e.reset()
                aimers[i].reset()
                starts[i] = 1.0
            new_obs.append(e.fair(0))
            new_opp.append(e.fair(1))
        obs, opp = torch.tensor(new_obs), new_opp
    return wins, losses, draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
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
    torch.set_default_device(device)  # every tensor below, and the networks, live on the device
    envs = [DuelEnv(map_dir=args.map_dir, seed=args.seed * 1000 + i, max_seconds=args.max_seconds) for i in range(args.envs)]
    eval_envs = [DuelEnv(map_dir=args.map_dir, seed=args.seed * 1000 + 500 + i, max_seconds=args.max_seconds) for i in range(8)]
    size = envs[0].fair_size
    policy = Policy(size).to(device)
    opponent = Policy(size).to(device)
    opponent.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    league = []  # checkpoint paths

    n = args.envs
    aimers = [ScriptedAimer(rng) for _ in range(n)]
    for e in envs:
        e.reset()
    obs = torch.tensor([e.fair(0) for e in envs])
    opp_obs = torch.tensor([e.fair(1) for e in envs])
    h = policy.initial(n)
    opp_h = opponent.initial(n)
    starts = torch.ones(n)
    # Each environment's opponent for its current episode: "latest", a checkpoint path, or "scripted".
    kinds = ["scripted"] * n
    league_nets = {}

    def pick_opponent():
        if not league:
            return "scripted"
        return "latest" if rng.random() < 0.5 else rng.choice(league)

    def opponent_net(kind):
        if kind == "latest":
            return opponent
        if kind not in league_nets:
            net = Policy(size)
            net.load_state_dict(torch.load(kind, map_location=device, weights_only=True))
            league_nets[kind] = net
        return league_nets[kind]

    log = open(out / "log.jsonl", "a")
    total = 0
    iteration = 0
    episode_rewards, results = [], []
    running = [0.0] * n
    started = time.time()
    while total < args.steps:
        iteration += 1
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done, buf_start = [], [], [], [], [], [], []
        h0 = h.clone()
        for t in range(args.rollout):
            choice, logp, value, h = act(policy, obs, h, starts)
            # The opponents: group environments by opponent kind for batched play.
            opp_choice = [None] * n
            groups = {}
            for i, k in enumerate(kinds):
                groups.setdefault(k, []).append(i)
            for k, idx in groups.items():
                if k == "scripted":
                    for i in idx:
                        opp_choice[i] = aimers[i](opp_obs[i].tolist())
                else:
                    net = opponent_net(k)
                    c, _, _, nh = act(net, opp_obs[idx], opp_h[:, idx], starts[idx])
                    opp_h[:, idx] = nh
                    for j, i in enumerate(idx):
                        opp_choice[i] = tuple(int(x) for x in c[j])
            rewards, dones = torch.zeros(n), torch.zeros(n)
            new_obs, new_opp = [], []
            buf_start.append(starts.clone())
            starts = torch.zeros(n)
            for i, e in enumerate(envs):
                a = DuelEnv.choice(*[int(x) for x in choice[i]])
                b = DuelEnv.choice(*opp_choice[i])
                _, _, (ra, rb), done, info = e.step(a, b)
                rewards[i] = ra
                running[i] += ra
                if done:
                    dones[i] = 1.0
                    dead = {d["side"] for d in info["deaths"]}
                    results.append((kinds[i], 1 in dead and 0 not in dead, 0 in dead and 1 not in dead))
                    episode_rewards.append(running[i])
                    running[i] = 0.0
                    e.reset()
                    starts[i] = 1.0
                    kinds[i] = pick_opponent()
                    opp_h[:, i] = 0.0
                    aimers[i].reset()
                new_obs.append(e.fair(0))
                new_opp.append(e.fair(1))
            buf_obs.append(obs)
            buf_act.append(choice)
            buf_logp.append(logp)
            buf_val.append(value)
            buf_rew.append(rewards)
            buf_done.append(dones)
            obs, opp_obs = torch.tensor(new_obs), torch.tensor(new_opp)
            total += n
        # Advantages (GAE), bootstrapped from the next value.
        with torch.no_grad():
            _, last_value, _ = policy(obs.unsqueeze(1), h, starts.unsqueeze(1))
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
        for kind in ("scripted", "latest", "league"):
            rs = [r for r in results[-300:] if (r[0] == kind if kind != "league" else r[0] not in ("scripted", "latest"))]
            if rs:
                record[f"vs_{kind}"] = {"episodes": len(rs), "won": round(sum(r[1] for r in rs) / len(rs), 2),
                                        "lost": round(sum(r[2] for r in rs) / len(rs), 2)}
        if iteration % args.eval_every == 0:
            w, l, d = evaluate(policy, eval_envs, args.eval_episodes, rng)
            record["eval_vs_aimer"] = {"won": w, "lost": l, "draw": d}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    torch.save(policy.state_dict(), out / "final.pt")


if __name__ == "__main__":
    main()
