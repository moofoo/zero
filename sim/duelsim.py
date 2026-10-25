"""Python bindings for zero's headless duel simulator (bot-ai ticket 41), through ctypes: no other dependencies.

    env = DuelEnv()                  # two Warbirds, Chaos's arena settings, no items
    obs_a, obs_b = env.reset()
    obs_a, obs_b, (reward_a, reward_b), done, info = env.step(action_a, action_b)

An action is a set of held inputs: a bitmask of ACTIONS (env.action(forward=True, bullet=True) builds one). Each step
runs `frame_skip` ticks of 10 ms. An observation is a dict of the side's own ship, the opponent as that side's client
sees it, and the nearest enemy weapons; ticket 42 decides what a policy may see of it (fair observations).

The library is libduelsim.so (or duelsim.dll), built by zero's CMake target duelsim_lib. A new environment replays a
captured login (sim/data/duel_a.cap, duel_b.cap: Chaos's duel profile, two Warbirds on freqs 0 and 1) and loads the
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
            lib.ds_reset.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 6
            lib.ds_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
            lib.ds_observe.restype = ctypes.c_int
            lib.ds_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.ds_happenings.restype = ctypes.c_int
            lib.ds_happenings.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.ds_solid.restype = ctypes.c_int
            lib.ds_solid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.ds_fair_size.restype = ctypes.c_int
            lib.ds_fair_observe.restype = ctypes.c_int
            lib.ds_fair_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            cls._instance = lib
        return cls._instance


class DuelEnv:
    """One duel: two ships, each driven by the caller.

    Rewards per step, for each side: damage dealt less damage taken, as shares of max energy, plus `kill_reward` for a
    kill and minus it for a death. An episode ends at the first death or after `max_seconds`.
    """

    def __init__(self, map_dir=None, captures=None, frame_skip=10, max_weapons=16, max_seconds=120.0,
                 kill_reward=1.0, spawn_center=(512.0, 512.0), spawn_radius=60.0, spawn_gap=(20.0, 40.0),
                 seed=None, library=None):
        self.lib = _Lib.get(library)
        self.frame_skip = frame_skip
        self.max_weapons = max_weapons
        self.max_ticks = int(max_seconds * 100)
        self.kill_reward = kill_reward
        self.spawn_center, self.spawn_radius, self.spawn_gap = spawn_center, spawn_radius, spawn_gap
        self.random = random.Random(seed)
        map_dir = Path(map_dir or Path.home() / "newcode/chaos-offline/work/bots")
        a, b = captures or (HERE / "data" / "duel_a.cap", HERE / "data" / "duel_b.cap")
        here = os.getcwd()
        os.chdir(map_dir)  # the client loads zones/<server>/<map>.lvl from where it runs
        try:
            self.handle = self.lib.ds_create(str(Path(a).resolve()).encode(), str(Path(b).resolve()).encode())
        finally:
            os.chdir(here)
        if not self.handle:
            raise RuntimeError("the simulator couldn't set up the duel (captures or map missing?)")
        size = len(SELF_FIELDS) + len(OTHER_FIELDS) + 5 * max_weapons
        self._obs = (ctypes.c_float * size)()
        self._events = (ctypes.c_float * (4 * 256))()
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

    # The policy's actions (ticket 42): three choices each step, no afterburner.
    TURNS = ({"left": True}, {}, {"right": True})
    THRUSTS = ({"forward": True}, {}, {"backward": True})
    FIRES = ({}, {"bullet": True}, {"bomb": True})

    @classmethod
    def choice(cls, turn, thrust, fire):
        """Held inputs from the three choices: turn 0 left / 1 none / 2 right, thrust 0 forward / 1 none / 2 reverse,
        fire 0 none / 1 bullet / 2 bomb."""
        held = {}
        held.update(cls.TURNS[turn])
        held.update(cls.THRUSTS[thrust])
        held.update(cls.FIRES[fire])
        return cls.action(**held)

    def fair(self, side):
        """The side's fair observation, 150 ms old (ds_fair_observe): a list of fair_size floats."""
        if not self.lib.ds_fair_observe(self.handle, side, self._fair):
            return [0.0] * self.fair_size
        return list(self._fair)

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
            if self._open(ax, ay) and self._open(bx, by):
                return (ax, ay, self.random.random()), (bx, by, self.random.random())
        raise RuntimeError("no open spawn spots found")

    def reset(self, spawns=None):
        (ax, ay, ao), (bx, by, bo) = spawns or self._spawns()
        self.lib.ds_reset(self.handle, ax, ay, ao, bx, by, bo)
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
        self.lib.ds_step(self.handle, action_a, action_b, self.frame_skip)
        self.ticks += self.frame_skip
        n = self.lib.ds_happenings(self.handle, self._events, 256)
        rewards = [0.0, 0.0]
        hits, deaths = [], []
        max_energy = [self.observe(0)["self"]["max_energy"] or 1.0, self.observe(1)["self"]["max_energy"] or 1.0]
        for k in range(n):
            kind, side, value, weapon = self._events[4 * k:4 * k + 4]
            side = int(side)
            if kind == 0:  # a hit taken by `side`
                hits.append({"side": side, "damage": value, "class": WEAPON_CLASSES[int(weapon)]})
                rewards[side] -= value / max_energy[side]
                rewards[1 - side] += value / max_energy[side]
            else:  # `side` died, to `value`
                deaths.append({"side": side, "killer": int(value)})
                rewards[side] -= self.kill_reward
                if int(value) != side:
                    rewards[int(value)] += self.kill_reward
        done = bool(deaths) or self.ticks >= self.max_ticks
        return self.observe(0), self.observe(1), tuple(rewards), done, {"hits": hits, "deaths": deaths, "ticks": self.ticks}


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
