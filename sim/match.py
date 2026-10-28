"""Two policies duel each other in the duel box (bot-ai ticket 46): which is stronger head to head.

    .venv/bin/python match.py runs/s2r-all01/final.pt runs/v8c-bombs/final.pt --games 200

Greedy play, one game a life, sides swapped every game (spawns 20 to 56 tiles apart). Each policy plays at --dial if it
has one. Prints wins, losses and draws for the first, its kill share with a 95% interval, and each side's bombs a
minute.
"""
import argparse
import collections
import math
import sys

import torch

from duelsim import DuelEnv
from train import act, load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--dial", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--map-dir", default=None)
    args = p.parse_args()

    nets = [load_policy(args.a), load_policy(args.b)]
    env = DuelEnv(seed=args.seed, max_seconds=90, map_dir=args.map_dir,
                  spawn_center=(186, 144), spawn_radius=40, spawn_gap=(20, 56))
    s = collections.Counter()
    for game in range(args.games):
        env.reset()
        sides = [0, 1] if game % 2 == 0 else [1, 0]  # sides[k]: the env side policy k flies
        hs = [net.initial(1) for net in nets]
        starts = torch.ones(1)
        done = False
        while not done:
            chosen = [None, None]
            for k, net in enumerate(nets):
                x = torch.from_numpy(env.fair_array(sides[k])).unsqueeze(0)
                if net.extra:
                    x = torch.cat([x, torch.full((1, 1), args.dial)], 1)
                c, _, _, hs[k] = act(net, x, hs[k], starts, greedy=True)
                chosen[sides[k]] = DuelEnv.choice(*c[0].tolist())
            starts = torch.zeros(1)
            _, _, _, done, info = env.step(chosen[0], chosen[1])
            s["steps"] += 1
            for shot in info["shots"]:
                if shot["class"] == "bomb":
                    s[f"bombs_{sides.index(shot['side'])}"] += 1
        dead = {sides.index(d["side"]) for d in info["deaths"]}
        s["won" if dead == {1} else "lost" if dead == {0} else "draw"] += 1
    kills = s["won"] + s["lost"]
    share = s["won"] / max(kills, 1)
    half = 1.96 * math.sqrt(share * (1 - share) / max(kills, 1))
    minutes = s["steps"] / 600.0
    print(f"{args.a} against {args.b}: won {s['won']} lost {s['lost']} draw {s['draw']}, kill share {share:.2f} "
          f"[{max(share - half, 0):.2f}-{min(share + half, 1):.2f}]; bombs a minute {s['bombs_0'] / minutes:.1f} "
          f"against {s['bombs_1'] / minutes:.1f}; {minutes:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
