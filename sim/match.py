"""Two policies duel each other in the duel box (bot-ai ticket 46): which is stronger head to head.

    .venv/bin/python match.py runs/s2r-all01/final.pt runs/v8c-bombs/final.pt --games 200

Greedy play, one game a life, sides swapped every game (spawns 20 to 56 tiles apart). Each policy plays at --dial if it
has one. Prints wins, losses and draws for the first, its kill share with a 95% interval, and each side's bombs a
minute; then for each, as style.py measures it: bullets and bombs fired a minute and the share of each that hit, the
time spent below half energy and the damage taken a minute; and its drains below 30% energy a minute, with the share it
got back to 60% from and the share it died in (ticket 46: energy management).
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
        drained = [False, False]
        done = False
        while not done:
            chosen = [None, None]
            for k, net in enumerate(nets):
                o = env.fair_array(sides[k])
                s[f"below_half_{k}"] += o[7] > 0.5 and o[4] < 0.5
                if o[7] > 0.5 and o[4] < 0.3 and not drained[k]:
                    drained[k] = True
                    s[f"drains_{k}"] += 1
                elif o[4] >= 0.6 and drained[k]:
                    drained[k] = False
                    s[f"recovered_{k}"] += 1
                x = torch.from_numpy(o).unsqueeze(0)
                if net.extra:
                    x = torch.cat([x, torch.full((1, 1), args.dial)], 1)
                c, _, _, hs[k] = act(net, x, hs[k], starts, greedy=True)
                chosen[sides[k]] = DuelEnv.choice(*c[0].tolist())
            starts = torch.zeros(1)
            _, _, _, done, info = env.step(chosen[0], chosen[1])
            s["steps"] += 1
            for shot in info["shots"]:
                s[f"{shot['class']}s_{sides.index(shot['side'])}"] += 1
            for hit in info["hits"]:
                k = sides.index(hit["side"])  # the side hit
                s[f"taken_{k}"] += hit["damage"]
                if not hit["own"]:
                    s[f"landed_{hit['class']}_{1 - k}"] += 1
        dead = {sides.index(d["side"]) for d in info["deaths"]}
        for k in dead:
            s[f"died_drained_{k}"] += drained[k]
        s["won" if dead == {1} else "lost" if dead == {0} else "draw"] += 1
    kills = s["won"] + s["lost"]
    share = s["won"] / max(kills, 1)
    half = 1.96 * math.sqrt(share * (1 - share) / max(kills, 1))
    minutes = s["steps"] / 600.0
    print(f"{args.a} against {args.b}: won {s['won']} lost {s['lost']} draw {s['draw']}, kill share {share:.2f} "
          f"[{max(share - half, 0):.2f}-{min(share + half, 1):.2f}]; bombs a minute {s['bombs_0'] / minutes:.1f} "
          f"against {s['bombs_1'] / minutes:.1f}; {minutes:.1f} min")
    for k, name in enumerate((args.a, args.b)):
        print(f"  {name}: bullets {s[f'bullets_{k}'] / minutes:.1f} a minute hitting "
              f"{s[f'landed_bullet_{k}'] / max(s[f'bullets_{k}'], 1):.2f}, bombs {s[f'bombs_{k}'] / minutes:.1f} hitting "
              f"{s[f'landed_bomb_{k}'] / max(s[f'bombs_{k}'], 1):.2f}; below half energy "
              f"{s[f'below_half_{k}'] / s['steps']:.2f} of the time; damage taken {s[f'taken_{k}'] / minutes:.0f} a minute; drains below 30% "
              f"{s[f'drains_{k}'] / minutes:.1f} a minute, recovered from {s[f'recovered_{k}'] / max(s[f'drains_{k}'], 1):.2f}, "
              f"died in {s[f'died_drained_{k}'] / max(s[f'drains_{k}'], 1):.2f}")


if __name__ == "__main__":
    sys.exit(main())
