"""A player's recordings as examples for behaviour cloning (bot-ai ticket 46): the fair observation the policy would
have had, and the choices it would have made to press the keys the player pressed.

    python3 player_data.py <recording> [<recording> ...] --out data/player.npz

A recording is the path without its extension: <name>.cap and <name>.inp from nullspace (chaos.py play
--record-player), and optionally <name>.player.events.jsonl, the player's combat log. Each is replayed through zero's
simulator (`duelsim player`, run from --map-dir) into fair observations, then paired with the player's inputs: a
decision every 10 ticks, seeing the observation from at least kObsDelayTicks (15) before, as the policy does.

Keys to choices, over the decision's 10 ticks (the frames' own times):
- turn: none if neither turn key was held for a tick in all; else the key held longer, a tap (3 ticks) if held under
  6.5 ticks, a full turn (10) otherwise;
- thrust: forward or reverse if held for 5 ticks or more (forward first), else none;
- fire: bomb if the bomb key was down at all, else bullet if the bullet key was, else none.
Prints how far the choices are from the keys (turn ticks, by the step) and the choices' shares.
"""
import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from recording import read_inputs

HERE = Path(__file__).parent
STEP, DELAY = 10, 15
LEFT, RIGHT, FORWARD, BACKWARD, BOMB, BULLET = 0, 1, 2, 3, 5, 6
TURN_TICKS = (-10, -3, 0, 3, 10)  # the turn choices, as signed ticks of turning (right positive)


def replay(base, map_dir, simulator):
    """Runs the simulator over a recording; returns (local ticks, observations) and prints its report."""
    base = Path(base)
    cap, inp = base.with_suffix(".cap"), base.with_suffix(".inp")
    if not cap.exists() or not inp.exists():
        raise FileNotFoundError(f"{base}: needs {cap.name} and its .inp")
    out = base.with_suffix(".obs")
    cmd = [str(simulator), "player", str(cap.resolve()), str(inp.resolve()), str(out.resolve())]
    events = base.parent / (base.name + ".player.events.jsonl")
    if events.exists():
        cmd.append(str(events.resolve()))
    result = subprocess.run(cmd, cwd=map_dir, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(("replayed", "  hits")):  # the replay's report, not the client's log
            print(f"  {line}")
    if result.returncode:
        raise RuntimeError(f"{base}: the replay failed: {result.stderr[-500:]}")
    data = out.read_bytes()
    magic, size, _ = struct.unpack_from("<4sIi", data)
    if magic != b"ZRO1":
        raise ValueError(f"{out}: not an observation file")
    record = 4 + 4 * size
    count = (len(data) - 12) // record
    rows = np.frombuffer(data, dtype=np.uint8, count=count * record, offset=12).reshape(count, record)
    ticks = rows[:, :4].copy().view("<u4")[:, 0].astype(np.int64)
    obs = rows[:, 4:].copy().view("<f4")
    return ticks, obs, inp


def held_ticks(frames, start, bit):
    """Ticks the action was held from `start` for STEP ticks, from the frames' times."""
    total = 0.0
    for tick, actions, dt in frames:
        if start <= tick < start + STEP and actions >> bit & 1:
            total += dt * 100.0
    return total


def examples(ticks, obs, inp):
    """(observations, choices, turn errors) for every decision with the player alive in a ship."""
    updates, _ = read_inputs(inp)
    frames = [(u["tick"], u["actions"], u["dt"] if u["dt"] is not None else 0.01) for u in updates]
    alive = {u["tick"] for u in updates if u["ship"] < 8 and u["enter_delay"] <= 0}
    first, last = updates[0]["tick"], updates[-1]["tick"]
    xs, ys, errors = [], [], []
    k = 0
    # Frames by tick, for the windows.
    by_tick = {}
    for f in frames:
        by_tick.setdefault(f[0], []).append(f)
    for start in range(first - first % STEP + STEP, last - STEP, STEP):
        if start not in alive and start + 1 not in alive:
            continue
        while k + 1 < len(ticks) and ticks[k + 1] <= start - DELAY:
            k += 1
        if ticks[k] > start - DELAY:
            continue
        window = [f for t in range(start, start + STEP) for f in by_tick.get(t, [])]
        left, right = held_ticks(window, start, LEFT), held_ticks(window, start, RIGHT)
        if max(left, right) < 1.0:
            turn = 2
        elif right >= left:
            turn = 3 if right < 6.5 else 4
        else:
            turn = 1 if left < 6.5 else 0
        errors.append(abs((right - left) - TURN_TICKS[turn]))
        forward, back = held_ticks(window, start, FORWARD), held_ticks(window, start, BACKWARD)
        thrust = 0 if forward >= 5 else 2 if back >= 5 else 1
        fire = 2 if held_ticks(window, start, BOMB) > 0 else 1 if held_ticks(window, start, BULLET) > 0 else 0
        xs.append(obs[k])
        ys.append((turn, thrust, fire))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64), np.array(errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recordings", nargs="+")
    p.add_argument("--out", default=None, help="an .npz of obs (n, fair size) and choices (n, 3: turn, thrust, fire)")
    p.add_argument("--map-dir", default=str(Path.home() / "newcode/chaos-offline/work/bots"))
    p.add_argument("--simulator", default=str(HERE.parent / "build" / "duelsim"))
    args = p.parse_args()
    all_x, all_y, all_e = [], [], []
    for r in args.recordings:
        base = Path(r)
        base = base.with_suffix("") if base.suffix in (".cap", ".inp", ".obs") else base
        print(base.name)
        ticks, obs, inp = replay(base, args.map_dir, args.simulator)
        x, y, e = examples(ticks, obs, inp)
        seen = (x[:, 8] > 0.5).mean() if len(x) else 0.0
        print(f"  {len(x)} decisions ({len(x) / 600:.1f} min alive), the opponent on screen in {seen:.0%}")
        all_x.append(x), all_y.append(y), all_e.append(e)
    x, y, e = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_e)
    names = (("left", "tap left", "none", "tap right", "right"), ("forward", "none", "reverse"),
             ("none", "bullet", "bomb"))
    for h, head in enumerate(("turn", "thrust", "fire")):
        counts = np.bincount(y[:, h], minlength=len(names[h])) / max(len(y), 1)
        print(f"{head}: " + ", ".join(f"{n} {c:.0%}" for n, c in zip(names[h], counts)))
    turning = e[(y[:, 0] != 2)] if len(e) else e
    print(f"turn rounding: mean {e.mean():.2f} ticks a decision off the keys (while turning {turning.mean():.2f}); "
          f"within a tick {np.mean(e <= 1.0):.0%}")
    if args.out:
        np.savez_compressed(args.out, obs=x, choices=y)
        print(f"wrote {len(x)} examples to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
