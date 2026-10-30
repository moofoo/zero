"""Python bindings for zero's headless duel simulator (bot-ai ticket 41), through ctypes: no other dependencies.

    env = DuelEnv()                  # two Warbirds, Chaos's arena settings, no items
    obs_a, obs_b = env.reset()
    obs_a, obs_b, (reward_a, reward_b), done, info = env.step(action_a, action_b)

An action is a set of held inputs: a bitmask of ACTIONS (env.action(forward=True, bullet=True) builds one). Each step
runs `frame_skip` ticks of 10 ms. An observation is a dict of the side's own ship, the opponent as that side's client
sees it, and the nearest enemy weapons; ticket 42 decides what a policy may see of it (fair observations).

The library is libduelsim.so (or duelsim.dll), built by zero's CMake target duelsim_lib. A new environment replays a
captured login (sim/data/duelbox_a.cap, duelbox_b.cap: chaos-offline's duelbox profile, two Warbirds on freqs 0 and 1,
each with a human duel's ship: full stats, 3 guns, 2 bombs, bouncing bullets, proximity, multifire, full shrapnel; the
older duel_a.cap and duel_b.cap, from the duel profile, happened to give the same ship) and loads the
map from map_dir/zones/Chaos/<map>.lvl.
"""
import ctypes
import math
import os
import random
from pathlib import Path

HERE = Path(__file__).parent

# zero's InputAction bits.
ACTIONS = {"left": 0, "right": 1, "forward": 2, "backward": 3, "afterburner": 4, "bomb": 5, "bullet": 6, "mine": 7}

SELF_FIELDS = ("x", "y", "vx", "vy", "orientation", "energy", "max_energy", "enter_delay", "bullet_wait", "bomb_wait")
OTHER_FIELDS = ("x", "y", "vx", "vy", "orientation", "enter_delay", "energy", "present")
WEAPON_CLASSES = ("bullet", "bomb", "other")


def default_library():
    names = ["libduelsim.so", "duelsim.dll", "libduelsim.dylib"]
    for folder in (HERE.parent / "build", HERE.parent / "build" / "Release", HERE.parent / "build" / "Debug"):
        for name in names:
            if (folder / name).exists():
                return folder / name
    raise FileNotFoundError("libduelsim not found; build zero's duelsim_lib target")


class _Lib:
    _instance = None

    @classmethod
    def get(cls, path=None):
        if cls._instance is None:
            lib = ctypes.CDLL(str(path or default_library()))
            lib.ds_create.restype = ctypes.c_void_p
            lib.ds_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.ds_create_bot.restype = ctypes.c_void_p
            lib.ds_create_bot.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            lib.ds_reset.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 6
            lib.ds_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
            lib.ds_observe.restype = ctypes.c_int
            lib.ds_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.ds_happenings.restype = ctypes.c_int
            lib.ds_happenings.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.ds_solid.restype = ctypes.c_int
            lib.ds_solid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.ds_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            lib.ds_ship_fields.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
            lib.ds_estimate_observe.restype = ctypes.c_int
            lib.ds_estimate_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            lib.ds_set_energy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
            lib.ds_bot_goal.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
            lib.ds_fair_size.restype = ctypes.c_int
            lib.ds_fair_observe.restype = ctypes.c_int
            lib.ds_fair_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            cls._instance = lib
        return cls._instance


class DuelEnv:
    """One duel: two ships, each driven by the caller.

    Rewards per step, for each side: damage dealt less damage taken, as shares of max energy, plus `kill_reward` for a
    kill and minus it for a death. An episode ends at the first death or after `max_seconds`.

    With `bot`, side 1 is a zero Chaos bot (ticket 45) and step()'s second action is ignored: bot={"skill": "veteran"},
    optionally "aggression" (default standard), "config" (a bot config; default chaos-offline's bots/chaos.cfg) and
    "args" (more of zero's command line).

    With `low_start` (share, low, high), that share of episodes start side 0 between low and high of its maximum energy,
    side 1 full (ticket 46: in trouble, where it has to get away and recharge).

    With `estimate`, fair_array() has two more floats after the fair observation: the opponent's energy as this side
    estimates it from the shots and hits it saw (zero's EnergyEstimate), mean and standard deviation as shares of its
    average maximum, as of the same moment (ticket 46). fair_size stays the fair observation's.
    """

    def __init__(self, map_dir=None, captures=None, frame_skip=10, max_weapons=16, max_seconds=120.0,
                 kill_reward=1.0, spawn_center=(512.0, 512.0), spawn_radius=60.0, spawn_gap=(20.0, 40.0),
                 seed=None, library=None, bot=None, low_start=None, estimate=False):
        self.lib = _Lib.get(library)
        self.frame_skip = frame_skip
        self.max_weapons = max_weapons
        self.max_ticks = int(max_seconds * 100)
        self.kill_reward = kill_reward
        self.spawn_center, self.spawn_radius, self.spawn_gap = spawn_center, spawn_radius, spawn_gap
        self.random = random.Random(seed)
        self.low_start = low_start
        self.estimate = estimate
        self._estimate = (ctypes.c_float * 2)()
        map_dir = Path(map_dir or Path.home() / "newcode/chaos-offline/work/bots")
        a, b = captures or (HERE / "data" / "duelbox_a.cap", HERE / "data" / "duelbox_b.cap")
        here = os.getcwd()
        os.chdir(map_dir)  # the client loads zones/<server>/<map>.lvl from where it runs
        try:
            if bot:
                config = Path(bot.get("config") or Path.home() / "newcode/chaos-offline-clean/bots/chaos.cfg")
                args = f"--skill {bot['skill']} --aggression {bot.get('aggression', 'standard')} --ship 1 " + bot.get("args", "")
                self.handle = self.lib.ds_create_bot(str(Path(a).resolve()).encode(), str(Path(b).resolve()).encode(),
                                                     str(config.resolve()).encode(), args.encode(),
                                                     (seed or 0) & 0xFFFFFFFF)
            else:
                self.handle = self.lib.ds_create(str(Path(a).resolve()).encode(), str(Path(b).resolve()).encode())
        finally:
            os.chdir(here)
        if not self.handle:
            raise RuntimeError("the simulator couldn't set up the duel (captures or map missing?)")
        size = len(SELF_FIELDS) + len(OTHER_FIELDS) + 5 * max_weapons
        self._obs = (ctypes.c_float * size)()
        self._events = (ctypes.c_float * (4 * 256))()
        self.lib.ds_seed(self.handle, (seed if seed is not None else random.getrandbits(32)) & 0xFFFFFFFF)
        self.fair_size = self.lib.ds_fair_size()
        self._fair = (ctypes.c_float * self.fair_size)()
        self.ticks = 0

    @staticmethod
    def action(**held):
        """A bitmask of held inputs, e.g. action(forward=True, left=True, bullet=True)."""
        mask = 0
        for name, on in held.items():
            if on:
                mask |= 1 << ACTIONS[name]
        return mask

    # The policy's actions (ticket 42): three choices each step, no afterburner. A turn is held for the whole step
    # (about 2.9 of the ship's 40 headings) or tapped for TAP_TICKS of it (about one heading), as a player taps the
    # key to line up a shot.
    TURNS = ({"left": True}, {"left": True}, {}, {"right": True}, {"right": True})
    TAPPED = (False, True, False, True, False)
    TAP_TICKS = 3
    THRUSTS = ({"forward": True}, {}, {"backward": True})
    FIRES = ({}, {"bullet": True}, {"bomb": True})

    @classmethod
    def choice(cls, turn, thrust, fire):
        """A step's inputs from the three choices: turn 0 left / 1 tap left / 2 none / 3 tap right / 4 right, thrust
        0 forward / 1 none / 2 reverse, fire 0 none / 1 bullet / 2 bomb. A tuple for step(): the held inputs, the
        inputs after the tap, and the tap's ticks (None for no tap)."""
        rest = {}
        rest.update(cls.THRUSTS[thrust])
        rest.update(cls.FIRES[fire])
        first = dict(rest, **cls.TURNS[turn])
        if cls.TAPPED[turn]:
            return cls.action(**first), cls.action(**rest), cls.TAP_TICKS
        return cls.action(**first), cls.action(**first), None

    def fair(self, side):
        """The side's fair observation, 150 ms old (ds_fair_observe): a list of fair_size floats."""
        if not self.lib.ds_fair_observe(self.handle, side, self._fair):
            return [0.0] * self.fair_size
        return list(self._fair)

    def fair_array(self, side):
        """fair() as a numpy float32 array (numpy is imported only for this)."""
        import numpy as np
        size = self.fair_size + (2 if self.estimate else 0)
        if not self.lib.ds_fair_observe(self.handle, side, self._fair):
            return np.zeros(size, dtype=np.float32)
        fair = np.ctypeslib.as_array(self._fair).copy()
        if not self.estimate:
            return fair
        if not self.lib.ds_estimate_observe(self.handle, side, self._estimate):
            return np.concatenate([fair, np.zeros(2, dtype=np.float32)])
        return np.concatenate([fair, np.ctypeslib.as_array(self._estimate)])

    SHIP_FIELDS = ("energy", "recharge", "rotation", "guns", "bombs", "thrust", "speed", "shrapnel", "repels", "bursts",
                   "decoys", "thors", "bricks", "rockets", "portals", "multifire", "capability")

    def bot_goal(self, x, y):
        """Where side 1's bot flies with --behavior bench (the piloting test bench)."""
        self.lib.ds_bot_goal(self.handle, x, y)

    def ship_fields(self, side):
        """A side's ship upgrades as a dict (zero's InputLog.h fields)."""
        out = (ctypes.c_uint * len(self.SHIP_FIELDS))()
        self.lib.ds_ship_fields(self.handle, side, out)
        return dict(zip(self.SHIP_FIELDS, out))

    def solid(self, x, y):
        return bool(self.lib.ds_solid(self.handle, 0, int(x), int(y)))

    def _open(self, x, y, margin=2):
        return all(not self.solid(x + dx, y + dy) for dx in range(-margin, margin + 1) for dy in range(-margin, margin + 1))

    def _spawns(self):
        cx, cy = self.spawn_center
        for _ in range(1000):
            ax = cx + self.random.uniform(-self.spawn_radius, self.spawn_radius)
            ay = cy + self.random.uniform(-self.spawn_radius, self.spawn_radius)
            angle = self.random.uniform(0, 2 * math.pi)
            gap = self.random.uniform(*self.spawn_gap)
            bx, by = ax + gap * math.cos(angle), ay + gap * math.sin(angle)
            # Both inside the spawn square, so a walled area (the duel box) keeps both ships in.
            inside = abs(bx - cx) <= self.spawn_radius and abs(by - cy) <= self.spawn_radius
            if inside and self._open(ax, ay) and self._open(bx, by):
                return (ax, ay, self.random.random()), (bx, by, self.random.random())
        raise RuntimeError("no open spawn spots found")

    def reset(self, spawns=None):
        (ax, ay, ao), (bx, by, bo) = spawns or self._spawns()
        self.lib.ds_reset(self.handle, ax, ay, ao, bx, by, bo)
        if self.low_start and self.random.random() < self.low_start[0]:
            self.lib.ds_set_energy(self.handle, 0, self.random.uniform(*self.low_start[1:]))
        self.lib.ds_step(self.handle, 0, 0, 20)  # let each client hear where the other is, and fill the delay
        self.lib.ds_happenings(self.handle, self._events, 256)  # drop what the respawn caused
        self.ticks = 0
        return self.observe(0), self.observe(1)

    def observe(self, side):
        count = self.lib.ds_observe(self.handle, side, self._obs, self.max_weapons)
        v = list(self._obs)
        n_self, n_other = len(SELF_FIELDS), len(OTHER_FIELDS)
        weapons = []
        for k in range(count):
            w = v[n_self + n_other + 5 * k:n_self + n_other + 5 * k + 5]
            weapons.append({"x": w[0], "y": w[1], "vx": w[2], "vy": w[3], "class": WEAPON_CLASSES[int(w[4])]})
        return {"self": dict(zip(SELF_FIELDS, v[:n_self])),
                "other": dict(zip(OTHER_FIELDS, v[n_self:n_self + n_other])),
                "weapons": weapons}

    def step(self, action_a, action_b):
        """Each action is a bitmask held for the step, or a tuple from choice() (inputs, inputs after a tap, ticks)."""
        a = action_a if isinstance(action_a, tuple) else (action_a, action_a, None)
        b = action_b if isinstance(action_b, tuple) else (action_b, action_b, None)
        cuts = sorted({0, self.frame_skip} | {x[2] for x in (a, b) if x[2] is not None and x[2] < self.frame_skip})
        for start, end in zip(cuts, cuts[1:]):
            ma = a[0] if a[2] is None or start < a[2] else a[1]
            mb = b[0] if b[2] is None or start < b[2] else b[1]
            self.lib.ds_step(self.handle, ma, mb, end - start)
        self.ticks += self.frame_skip
        n = self.lib.ds_happenings(self.handle, self._events, 256)
        rewards = [0.0, 0.0]
        hits, deaths, shots = [], [], []
        max_energy = [self.observe(0)["self"]["max_energy"] or 1.0, self.observe(1)["self"]["max_energy"] or 1.0]
        for k in range(n):
            kind, side, value, weapon = self._events[4 * k:4 * k + 4]
            side = int(side)
            if kind in (0, 2):  # a hit taken by `side`, from its own weapon for kind 2
                hits.append({"side": side, "damage": value, "share": value / max_energy[side],
                             "class": WEAPON_CLASSES[int(weapon)], "own": kind == 2})
                rewards[side] -= value / max_energy[side]
                rewards[1 - side] += value / max_energy[side]
            elif kind == 3:  # `side` fired a shot costing `value` energy (ticket 51)
                shots.append({"side": side, "cost": value, "share": value / max_energy[side],
                              "class": WEAPON_CLASSES[int(weapon)]})
            else:  # `side` died, to `value`
                deaths.append({"side": side, "killer": int(value)})
                rewards[side] -= self.kill_reward
                if int(value) != side:
                    rewards[int(value)] += self.kill_reward
        done = bool(deaths) or self.ticks >= self.max_ticks
        return self.observe(0), self.observe(1), tuple(rewards), done, {"hits": hits, "deaths": deaths, "shots": shots,
                                                                         "ticks": self.ticks}


if __name__ == "__main__":
    # A smoke test: two ships that thrust, turn and fire, for a few episodes; prints the speed.
    import time
    env = DuelEnv(seed=1)
    start, steps = time.time(), 0
    for episode in range(5):
        obs_a, obs_b = env.reset()
        done = False
        total = [0.0, 0.0]
        while not done:
            t = env.ticks
            a = env.action(forward=(t // 150) % 2 == 0, left=(t // 70) % 3 == 0, bullet=True)
            b = env.action(forward=True, right=(t // 90) % 2 == 0, bullet=(t // 50) % 2 == 0, bomb=(t % 300) < 20)
            obs_a, obs_b, (ra, rb), done, info = env.step(a, b)
            total[0] += ra
            total[1] += rb
            steps += 1
            for d in info["deaths"]:
                print(f"  episode {episode}: side {d['side']} killed by side {d['killer']} at {info['ticks'] / 100:.1f} s")
        print(f"episode {episode}: {env.ticks / 100:.1f} s, rewards {total[0]:+.2f} / {total[1]:+.2f}, "
              f"A at ({obs_a['self']['x']:.0f}, {obs_a['self']['y']:.0f}) energy {obs_a['self']['energy']:.0f}, "
              f"sees {len(obs_a['weapons'])} enemy weapons")
    wall = time.time() - start
    simulated = steps * env.frame_skip / 100
    print(f"{steps} steps, {simulated:.0f} simulated s in {wall:.2f} s: {simulated / wall:.0f} times real time "
          f"(with Python per step of {env.frame_skip} ticks)")
