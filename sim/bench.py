"""The piloting test bench (bot-ai ticket 45): zero's Chaos bot flying from point to point on the Chaos map.

    .venv/bin/python bench.py --trials 500 --baseline data/bench-baseline.json      # a build against the baseline
    .venv/bin/python bench.py --trials 500 --out runs/bench/new.json                # a new baseline

Each trial puts the bot (--behavior bench: the GoToNode it roams with) at an open start and gives it a goal 30 to 300
tiles off by a route (start and goal pairs fixed by --seed, so builds fly the same trials). It ends when the bot is
within 3 tiles of the goal, or at 3 times the shortest route's time at top speed plus 10 s. Scores:
- arrived: the share of trials that got there; ratio: time taken over the shortest route's time at top speed
- stuck: seconds with under 2 tiles of progress over the last 3 s
- bounces: wall hits a minute (the speed falling by a third and turning more than 60 degrees in 0.1 s)
- flips: turn reversals a minute (the heading turning one way then the other), a sign of oscillating
The shortest route is an 8-way grid path keeping a tile clear of walls (a Warbird's radius is 0.875 tiles); trials
need a route without doors, which open and shut on a timer.
"""
import argparse
import heapq
import json
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np

from duelsim import DuelEnv

HERE = Path(__file__).parent
TOP_SPEED = 20.3  # tiles a second, a Warbird at its maximum (MaximumSpeed 3250)


def load_walls(path):
    data = path.read_bytes()
    off = struct.unpack_from("<I", data, 2)[0] if data[:2] == b"BM" else 0
    solid = np.zeros((1024, 1024), dtype=bool)
    for k in range(off, len(data) - 3, 4):
        v = struct.unpack_from("<I", data, k)[0]
        x, y, t = v & 0x3FF, (v >> 12) & 0x3FF, v >> 24
        if 0 < t < 170:  # walls and doors (162-169: a route through a door depends on its timer); the rest are open
            solid[y, x] = True
    # A tile's clearance: blocked if any wall is within one tile.
    blocked = solid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            blocked |= np.roll(np.roll(solid, dy, 0), dx, 1)
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
    return blocked


def route_length(blocked, start, goal, limit=400000):
    """The shortest 8-way path's length in tiles, or None."""
    sx, sy = start
    gx, gy = goal
    best = {(sx, sy): 0.0}
    heap = [(math.hypot(gx - sx, gy - sy), 0.0, sx, sy)]
    seen = 0
    while heap and seen < limit:
        _, g, x, y = heapq.heappop(heap)
        if (x, y) == (gx, gy):
            return g
        if g > best.get((x, y), 1e18):
            continue
        seen += 1
        for dx, dy, c in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                          (1, 1, 1.41421), (1, -1, 1.41421), (-1, 1, 1.41421), (-1, -1, 1.41421)):
            nx, ny = x + dx, y + dy
            if blocked[ny, nx] or (dx and dy and (blocked[y, nx] or blocked[ny, x])):
                continue
            ng = g + c
            if ng < best.get((nx, ny), 1e18):
                best[(nx, ny)] = ng
                heapq.heappush(heap, (ng + math.hypot(gx - nx, gy - ny), ng, nx, ny))
    return None


def make_trials(blocked, count, seed):
    rng = random.Random(seed)
    open_tiles = np.argwhere(~blocked)
    trials = []
    while len(trials) < count:
        sy, sx = open_tiles[rng.randrange(len(open_tiles))]
        gy, gx = open_tiles[rng.randrange(len(open_tiles))]
        if not 30 <= math.hypot(gx - sx, gy - sy) <= 300:
            continue
        length = route_length(blocked, (int(sx), int(sy)), (int(gx), int(gy)))
        if length is None or length > 450:
            continue
        trials.append({"start": [int(sx) + 0.5, int(sy) + 0.5], "goal": [int(gx) + 0.5, int(gy) + 0.5],
                       "orientation": rng.random(), "route": round(length, 1)})
    return trials


def fly(env, trial, park):
    sx, sy = trial["start"]
    gx, gy = trial["goal"]
    env.bot_goal(gx, gy)
    env.reset(spawns=(park, (sx, sy, trial["orientation"])))
    limit = 3 * trial["route"] / TOP_SPEED + 10
    steps, positions = 0, []
    arrived, stuck, bounces, flips = False, 0, 0, 0
    last_turn, prev = 0, None
    while steps * 0.1 < limit:
        env.step(0, 0)
        steps += 1
        s = env.observe(1)["self"]
        x, y, vx, vy, o = s["x"], s["y"], s["vx"], s["vy"], s["orientation"]
        positions.append((x, y))
        if prev:
            speed0, speed1 = math.hypot(prev[2], prev[3]), math.hypot(vx, vy)
            if speed0 > 3 and speed1 < speed0 * 0.67:
                cos = (prev[2] * vx + prev[3] * vy) / (speed0 * max(speed1, 1e-6))
                if cos < 0.5:
                    bounces += 1
            turn = ((o - prev[4] + 0.5) % 1.0) - 0.5
            if abs(turn) > 0.005:
                direction = 1 if turn > 0 else -1
                if last_turn and direction != last_turn:
                    flips += 1
                last_turn = direction
        prev = (x, y, vx, vy, o)
        if len(positions) > 30 and math.hypot(x - positions[-31][0], y - positions[-31][1]) < 2:
            stuck += 1
        if math.hypot(gx - x, gy - y) <= 3:
            arrived = True
            break
    seconds = steps * 0.1
    return {"arrived": arrived, "seconds": round(seconds, 1), "ratio": round(seconds / (trial["route"] / TOP_SPEED), 2),
            "stuck": round(stuck * 0.1, 1), "bounces": bounces, "flips": flips}


def summarize(results):
    n = len(results)
    minutes = sum(r["seconds"] for r in results) / 60
    arrived = [r for r in results if r["arrived"]]
    ratios = sorted(r["ratio"] for r in arrived)
    def se(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1) / len(xs))
    return {"trials": n, "arrived": round(len(arrived) / n, 3),
            "ratio_median": ratios[len(ratios) // 2] if ratios else None,
            "ratio_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "ratio_se": round(se(ratios), 3) if len(ratios) > 1 else None,
            "stuck_per_trial": round(sum(r["stuck"] for r in results) / n, 2),
            "bounces_per_minute": round(sum(r["bounces"] for r in results) / minutes, 2),
            "flips_per_minute": round(sum(r["flips"] for r in results) / minutes, 2)}


def compare(summary, baseline):
    """Lines for the scores worse than the baseline beyond noise."""
    worse = []
    n = summary["trials"]
    se_arrived = math.sqrt(max(baseline["arrived"] * (1 - baseline["arrived"]), 0.01) / n)
    if summary["arrived"] < baseline["arrived"] - 2 * math.sqrt(2) * se_arrived:
        worse.append(f"arrived {summary['arrived']} against {baseline['arrived']}")
    if summary["ratio_mean"] and baseline["ratio_mean"]:
        se = math.hypot(summary["ratio_se"] or 0, baseline["ratio_se"] or 0)
        if summary["ratio_mean"] > baseline["ratio_mean"] + 2 * se:
            worse.append(f"ratio {summary['ratio_mean']} against {baseline['ratio_mean']}")
    for key in ("stuck_per_trial", "bounces_per_minute", "flips_per_minute"):
        if summary[key] > baseline[key] * 1.25 + 0.2:
            worse.append(f"{key} {summary[key]} against {baseline[key]}")
    return worse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--skill", default="veteran")
    p.add_argument("--bot-args", default="", help="more of zero's command line for the bot")
    p.add_argument("--map-dir", default=None)
    p.add_argument("--out", help="write the summary and the trials here (a baseline)")
    p.add_argument("--baseline", help="compare with this saved run; exits 1 if anything is worse")
    args = p.parse_args()

    map_dir = Path(args.map_dir or Path.home() / "newcode/chaos-offline/work/bots")
    blocked = load_walls(map_dir / "zones" / "Chaos" / "bdhadar.lvl")
    trials = make_trials(blocked, args.trials, args.seed)
    env = DuelEnv(map_dir=map_dir, seed=args.seed, max_seconds=1e9,
                  bot={"skill": args.skill, "args": f"--behavior bench {args.bot_args}"})
    # The other ship waits out of the way, still, in the duel box.
    park = (186.0, 144.0, 0.0)
    results = []
    for k, trial in enumerate(trials):
        results.append(dict(trial, **fly(env, trial, park)))
        if (k + 1) % 50 == 0:
            print(f"{k + 1} trials: {summarize(results)}", file=sys.stderr, flush=True)
    summary = summarize(results)
    print(json.dumps(summary))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "args": vars(args), "trials": results}, indent=1))
    if args.baseline:
        worse = compare(summary, json.loads(Path(args.baseline).read_text())["summary"])
        for line in worse:
            print("worse:", line)
        sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
