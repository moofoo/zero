"""How a policy duels a zero bot in the duel box: its kill share and how it fights (bot-ai ticket 50).

    .venv/bin/python style.py runs/v6-panel/final.pt --games 100 --dials 0,0.5,1 --bot veteran:standard

Greedy play, one game a life, in the duel box (spawns 20 to 56 tiles apart). Per dial setting (policies trained with
--dial; one without plays once): kill share, damage taken a minute and its share from the policy's own bombs, the
distance to the enemy while on screen (median and quartiles), the share of steps reversing while facing the enemy
(within 60 degrees ahead), the share of time beyond --range tiles, and the bullet and bomb hits it lands a minute; and
for ticket 51, shots fired a minute, hits a bullet, its energy when firing and the time it spends below half energy
(the energy it saw, 150 ms old).
"""
import argparse
import collections
import math
import sys
import time

import torch

from duelsim import DuelEnv
from train import Policy, act, load_policy


def play(policy, env, games, dial, range_tiles):
    s = collections.Counter()
    dists, fire_energy = [], []
    for _ in range(games):
        env.reset()
        h = policy.initial(1)
        starts = torch.ones(1)
        done = False
        while not done:
            o = env.fair_array(0)
            x = torch.from_numpy(o).unsqueeze(0)
            if policy.extra:
                x = torch.cat([x, torch.full((1, 1), dial)], 1)
            c, _, _, h = act(policy, x, h, starts, greedy=True)
            starts = torch.zeros(1)
            turn, thrust, fire = c[0].tolist()
            s["steps"] += 1
            seen = o[8] > 0.5 or o[9] > 0.5
            if seen:
                ahead, right = o[10] * 60.0, o[11] * 60.0
                d = math.hypot(ahead, right)
                if o[8] > 0.5:
                    dists.append(d)
                s["beyond"] += d > range_tiles
                if thrust == 2 and ahead > 0 and abs(math.atan2(right, ahead)) < math.pi / 3:
                    s["reverse_facing"] += 1
            else:
                s["beyond"] += 1
            s["thrust_reverse"] += thrust == 2
            s["below_half"] += o[7] > 0.5 and o[4] < 0.5
            s["fire_bullet"] += fire == 1
            s["fire_bomb"] += fire == 2
            _, _, _, done, info = env.step(DuelEnv.choice(turn, thrust, fire), 0)
            for shot in info["shots"]:
                if shot["side"] == 0:
                    s["shots_" + shot["class"]] += 1
                    s["shots_below_half"] += o[4] < 0.5
                    fire_energy.append(float(o[4]))
            for hit in info["hits"]:
                if hit["side"] == 0:
                    s["taken"] += hit["damage"]
                    s["taken_own"] += hit["damage"] if hit["own"] else 0
                elif not hit["own"]:
                    s["landed_" + hit["class"]] += 1
        dead = {d["side"] for d in info["deaths"]}
        s["won" if dead == {1} else "lost" if dead == {0} else "draw"] += 1
    return s, sorted(dists), sorted(fire_energy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--dials", default="0,0.5,1")
    p.add_argument("--bot", default="veteran:standard", help="skill:aggression")
    p.add_argument("--range", type=float, default=15.0, help="tiles; the share of time beyond it is reported")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--map-dir", default=None)
    args = p.parse_args()

    policy = load_policy(args.checkpoint)
    skill, _, aggression = args.bot.partition(":")
    dials = [float(x) for x in args.dials.split(",")] if policy.extra else [0.0]
    for dial in dials:
        env = DuelEnv(seed=args.seed, max_seconds=90, map_dir=args.map_dir,
                      bot={"skill": skill, "aggression": aggression or "standard"},
                      spawn_center=(186, 144), spawn_radius=40, spawn_gap=(20, 56))
        started = time.time()
        s, ds, fe = play(policy, env, args.games, dial, args.range)
        minutes = s["steps"] / 600.0
        kills = s["won"] + s["lost"]
        q = (lambda f: ds[int(f * (len(ds) - 1))]) if ds else (lambda f: float("nan"))
        print(f"dial {dial:.2f} vs {args.bot}: won {s['won']} lost {s['lost']} draw {s['draw']}, "
              f"kill share {s['won'] / max(kills, 1):.2f}; {minutes:.1f} min")
        print(f"  damage taken {s['taken'] / minutes:.0f} a minute, {s['taken_own'] / max(s['taken'], 1):.0%} own bombs; "
              f"distance on screen median {q(0.5):.1f} (quartiles {q(0.25):.1f}-{q(0.75):.1f}); "
              f"beyond {args.range:.0f} tiles {s['beyond'] / s['steps']:.0%}")
        print(f"  reversing {s['thrust_reverse'] / s['steps']:.0%} of steps, facing the enemy {s['reverse_facing'] / s['steps']:.0%}; "
              f"fires bullets {s['fire_bullet'] / s['steps']:.0%}, bombs {s['fire_bomb'] / s['steps']:.0%}; "
              f"lands a minute: bullets {s['landed_bullet'] / minutes:.1f}, bombs {s['landed_bomb'] / minutes:.1f}, "
              f"other {s['landed_other'] / minutes:.1f}  ({time.time() - started:.0f} s)")
        shots = s["shots_bullet"] + s["shots_bomb"]
        print(f"  energy (ticket 51): fires a minute bullets {s['shots_bullet'] / minutes:.1f}, bombs "
              f"{s['shots_bomb'] / minutes:.1f}; hits a bullet {s['landed_bullet'] / max(s['shots_bullet'], 1):.2f}; "
              f"energy when firing median {fe[len(fe) // 2] if fe else float('nan'):.0%}, below half "
              f"{s['shots_below_half'] / max(shots, 1):.0%} of shots; time below half {s['below_half'] / s['steps']:.0%}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
