"""Behaviour cloning from a player's recordings (bot-ai ticket 46): train.Policy taught to choose what the player chose.

    .venv/bin/python clone.py data/player.npz --test 1 --init runs/v6-panel/final.pt --out runs/clone1.pt

The examples come from player_data.py (obs, choices, ticks, session). Sessions named by --test are held out; the rest
train. The network sees each session in order, in windows of --window decisions with its memory cleared at each
window's start, and learns all three choices (turn, thrust, fire) by cross-entropy. --fraction trains on the first
part of each training session only, for a learning curve. --mirror adds each window mirrored left to right (the
observation's right-hand numbers negated, the rays reversed, left and right turns swapped).

Prints held-out accuracy per choice (and the share of the most common choice, for scale) as training goes, and the
starting network's (with --init) before any.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from train import Policy, load_policy
from duelsim import _Lib

HEADS = ("turn", "thrust", "fire")


def mirror(obs, choices):
    """Left and right swapped: the ship-frame "right" numbers negated, the rays reversed, turns swapped. The map
    heading (2, 3) is left as it is: the map isn't mirrored."""
    o = obs.copy()
    for i in (1, 11, 13, 15):  # own velocity, opponent offset, relative velocity, its heading: right
        o[..., i] = -o[..., i]
    for k in range(16):
        base = 16 + 8 * k
        o[..., base + 1] = -o[..., base + 1]
        o[..., base + 3] = -o[..., base + 3]
    rays = o[..., 144:160].copy()
    o[..., 144:160] = rays[..., [(-r) % 16 for r in range(16)]]
    c = choices.copy()
    c[..., 0] = 4 - c[..., 0]
    return o, c


def windows(obs, choices, session, ticks, sessions, window, fraction=1.0):
    """(obs, choices, valid) windows of `window` decisions from the given sessions, cut at gaps in time."""
    xs, ys, ms = [], [], []
    for s in sessions:
        idx = np.where(session == s)[0]
        idx = idx[:max(1, int(len(idx) * fraction))]
        # Runs of consecutive decisions (10 ticks apart).
        breaks = np.where(np.diff(ticks[idx]) != 10)[0] + 1
        for run in np.split(idx, breaks):
            for start in range(0, len(run), window):
                part = run[start:start + window]
                x = np.zeros((window, obs.shape[1]), np.float32)
                y = np.zeros((window, 3), np.int64)
                m = np.zeros(window, np.float32)
                x[:len(part)], y[:len(part)], m[:len(part)] = obs[part], choices[part], 1.0
                xs.append(x), ys.append(y), ms.append(m)
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys)), torch.tensor(np.array(ms))


def run(policy, x, h=None):
    logits, _, _ = policy(x, policy.initial(x.shape[0]) if h is None else h)
    return logits


def evaluate(policy, x, y, m):
    with torch.no_grad():
        logits = run(policy, x)
    out = {}
    for k, lg in enumerate(logits):
        right = ((lg.argmax(-1) == y[..., k]).float() * m).sum() / m.sum()
        loss = (nn.functional.cross_entropy(lg.transpose(1, 2), y[..., k], reduction="none") * m).sum() / m.sum()
        out[HEADS[k]] = (round(right.item(), 3), round(loss.item(), 3))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+", help="player_data.py .npz files")
    p.add_argument("--test", default="1", help="held-out sessions, by index (comma-separated)")
    p.add_argument("--init", default=None, help="a train.py checkpoint to start from (default: a new network)")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    parts = [np.load(f) for f in args.data]
    offset, obs, choices, session, ticks = 0, [], [], [], []
    for d in parts:
        obs.append(d["obs"]), choices.append(d["choices"]), ticks.append(d["ticks"])
        session.append(d["session"] + offset)
        offset += int(d["session"].max()) + 1
    obs, choices, session, ticks = map(np.concatenate, (obs, choices, session, ticks))
    test = [int(s) for s in args.test.split(",")]
    train = [s for s in range(offset) if s not in test]
    x, y, m = windows(obs, choices, session, ticks, train, args.window, args.fraction)
    if args.mirror:
        mx, my = mirror(x.numpy(), y.numpy())
        x, y, m = torch.cat([x, torch.tensor(mx)]), torch.cat([y, torch.tensor(my)]), torch.cat([m, m])
    tx, ty, tm = windows(obs, choices, session, ticks, test, args.window)
    common = {HEADS[k]: round(float(np.bincount(choices[np.isin(session, test), k]).max()
                                    / np.isin(session, test).sum()), 3) for k in range(3)}
    print(json.dumps({"train_decisions": int(m.sum().item()), "test_decisions": int(tm.sum().item()),
                      "most_common_share": common}), flush=True)

    fair_size = _Lib.get().ds_fair_size()
    policy = load_policy(args.init) if args.init else Policy(fair_size)
    if policy.extra:
        raise SystemExit("cloning a dial policy isn't supported; start from one without")
    if args.init:
        print(json.dumps({"epoch": 0, "held_out": evaluate(policy, tx, ty, tm)}), flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(x.shape[0])
        total = 0.0
        for b in range(0, len(order), args.batch):
            i = order[b:b + args.batch]
            logits = run(policy, x[i])
            loss = sum((nn.functional.cross_entropy(lg.transpose(1, 2), y[i][..., k], reduction="none") * m[i]).sum()
                       for k, lg in enumerate(logits)) / m[i].sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(i)
        held = evaluate(policy, tx, ty, tm)
        score = sum(v[1] for v in held.values())
        print(json.dumps({"epoch": epoch, "train_loss": round(total / len(order), 3), "held_out": held}), flush=True)
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.clone() for k, v in policy.state_dict().items()})
    print(json.dumps({"best_epoch": best[1], "held_out_loss": round(best[0], 3)}))
    if args.out:
        torch.save(best[2], args.out)


if __name__ == "__main__":
    sys.exit(main())
