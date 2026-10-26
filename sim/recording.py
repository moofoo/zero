"""Reads a player's recording (bot-ai ticket 46) and checks it: a packet capture (.cap, zero's ZERO_CAPTURE or
nullspace's NULLSPACE_CAPTURE) and an input log (.inp, ZERO_INPUTLOG or NULLSPACE_INPUTLOG).

    python3 recording.py ~/newcode/chaos-offline/work/player-recordings/20261026-201500-12345

(the path without its extension; either file alone works). Prints the recording's length, packets by type, the
frame rate, the time in a ship, and how long the turn keys were held (the policy's tap is 3 ticks, a full turn 10),
for checking a pilot session before long ones.
"""
import collections
import struct
import sys
from pathlib import Path

ACTIONS = ("left", "right", "forward", "backward", "afterburner", "bomb", "bullet", "mine")
SHIP_FIELDS = 17
UPDATE = struct.Struct("<II6fBf%dI" % SHIP_FIELDS)  # after the type byte: zero's record 1
FRAME_DT = struct.Struct("<f")  # nullspace's record 3 adds the frame's time


def read_capture(path):
    """(direction, tick, bytes) for each packet: direction 0 received, 1 sent."""
    data = Path(path).read_bytes()
    out, at = [], 0
    while at + 9 <= len(data):
        direction, tick, size = struct.unpack_from("<BII", data, at)
        at += 9
        if at + size > len(data):
            break  # cut short by the program stopping
        out.append((direction, tick, data[at:at + size]))
        at += size
    return out


def read_inputs(path):
    """Updates as dicts (tick, actions, x, y, vx, vy, orientation, energy, ship, enter_delay, fields, dt: None for
    zero's tick updates) and one-shot actions as (tick, action)."""
    data = Path(path).read_bytes()
    updates, actions, at = [], [], 0
    while at < len(data):
        kind = data[at]
        at += 1
        if kind in (1, 3):
            size = UPDATE.size + (FRAME_DT.size if kind == 3 else 0)
            if at + size > len(data):
                break
            v = UPDATE.unpack_from(data, at)
            dt = FRAME_DT.unpack_from(data, at + UPDATE.size)[0] if kind == 3 else None
            at += size
            updates.append({"tick": v[0], "actions": v[1], "x": v[2], "y": v[3], "vx": v[4], "vy": v[5],
                            "orientation": v[6], "energy": v[7], "ship": v[8], "enter_delay": v[9],
                            "fields": v[10:10 + SHIP_FIELDS], "dt": dt})
        elif kind == 2:
            if at + 5 > len(data):
                break
            actions.append(struct.unpack_from("<IB", data, at))
            at += 5
        else:
            raise ValueError(f"{path}: unknown record type {kind} at byte {at - 1}")
    return updates, actions


def packet_type(payload):
    """A game packet's type as text: core packets as 'core 0x..'."""
    if not payload:
        return "empty"
    if payload[0] == 0 and len(payload) > 1:
        return f"core 0x{payload[1]:02X}"
    return f"0x{payload[0]:02X}"


def hold_lengths(updates, bit):
    """How long (ticks, from the frames' times) each press of an action was held."""
    lengths, held = [], 0.0
    for u in updates:
        dt = u["dt"] if u["dt"] is not None else 0.01
        if u["actions"] >> bit & 1:
            held += dt * 100.0
        elif held:
            lengths.append(held)
            held = 0.0
    return lengths


def summarize(base):
    base = Path(base)
    cap, inp = base.with_suffix(".cap"), base.with_suffix(".inp")
    if cap.exists():
        packets = read_capture(cap)
        if packets:
            ticks = packets[-1][1] - packets[0][1]
            print(f"{cap.name}: {len(packets)} packets over {ticks / 100 / 60:.1f} min")
            for direction, name in ((0, "received"), (1, "sent")):
                types = collections.Counter(packet_type(p) for d, _, p in packets if d == direction)
                top = ", ".join(f"{t} {n}" for t, n in types.most_common(8))
                print(f"  {name}: {sum(types.values())} ({top})")
        else:
            print(f"{cap.name}: no packets")
    if inp.exists():
        updates, actions = read_inputs(inp)
        if not updates:
            print(f"{inp.name}: no updates")
            return
        ticks = updates[-1]["tick"] - updates[0]["tick"]
        frames = [u["dt"] for u in updates if u["dt"] is not None]
        print(f"{inp.name}: {len(updates)} updates over {ticks / 100 / 60:.1f} min, {len(actions)} one-shot actions")
        if frames:
            fs = sorted(frames)
            print(f"  frames: {len(frames) / max(sum(frames), 1e-9):.0f} a second, dt median {fs[len(fs) // 2] * 1000:.1f} ms, "
                  f"longest {fs[-1] * 1000:.0f} ms")
        alive = [u for u in updates if u["ship"] < 8 and u["enter_delay"] <= 0]
        print(f"  in a ship and alive: {len(alive) / len(updates):.0%} of updates; ships flown: "
              f"{sorted(collections.Counter(u['ship'] for u in updates if u['ship'] < 8).items())}")
        held = {name: sum(1 for u in alive if u["actions"] >> bit & 1) / max(len(alive), 1)
                for bit, name in enumerate(ACTIONS)}
        print("  held while alive: " + ", ".join(f"{k} {v:.0%}" for k, v in held.items() if v > 0))
        for bit, name in ((0, "left"), (1, "right")):
            lengths = sorted(hold_lengths(alive, bit))
            if lengths:
                short = sum(1 for x in lengths if x < 3) / len(lengths)
                print(f"  {name} presses: {len(lengths)}, median {lengths[len(lengths) // 2]:.1f} ticks, quartiles "
                      f"{lengths[len(lengths) // 4]:.1f}-{lengths[3 * len(lengths) // 4]:.1f}; shorter than the "
                      f"policy's 3-tick tap: {short:.0%}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg).with_suffix(""))
