"""Many DuelEnvs stepped together, in worker processes (one a core) or in this process.

    venv = VecEnv(64, workers=8, map_dir=..., seed=1000)
    fair_a, fair_b = venv.reset()
    for ra, done, info, fair_a, fair_b in venv.step(actions_a, actions_b): ...

Each environment's own seed is seed + its index. An environment that finishes an episode is reset at once, and step()
returns the new episode's first observations with the finished one's reward, done and info. Observations are numpy
float32 arrays (DuelEnv.fair_array).
"""
import multiprocessing as mp

from duelsim import DuelEnv


def _make(index, seed, kwargs):
    return DuelEnv(seed=seed + index, **kwargs)


def _step(env, a, b):
    _, _, (ra, _), done, info = env.step(a, b)
    if done:
        env.reset()
    return ra, done, info, env.fair_array(0), env.fair_array(1)


def _worker(conn, indices, seed, kwargs):
    envs = [_make(i, seed, kwargs) for i in indices]
    conn.send(envs[0].fair_size)
    while True:
        message = conn.recv()
        if message[0] == "reset":
            for e in envs:
                e.reset()
            conn.send([(e.fair_array(0), e.fair_array(1)) for e in envs])
        elif message[0] == "step":
            conn.send([_step(e, a, b) for e, a, b in zip(envs, message[1], message[2])])
        else:
            conn.close()
            return


class VecEnv:
    def __init__(self, count, workers=1, seed=0, **kwargs):
        self.count = count
        workers = max(1, min(workers, count))
        if workers == 1:
            self.envs = [_make(i, seed, kwargs) for i in range(count)]
            self.fair_size = self.envs[0].fair_size
            return
        self.envs = None
        # Spawned, not forked: each worker loads its own copy of the simulator library.
        context = mp.get_context("spawn")
        self.slices, self.conns, self.procs = [], [], []
        for w in range(workers):
            indices = list(range(w, count, workers))
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, indices, seed, kwargs), daemon=True)
            proc.start()
            self.slices.append(indices)
            self.conns.append(parent)
            self.procs.append(proc)
        self.fair_size = [c.recv() for c in self.conns][0]

    def reset(self):
        if self.envs is not None:
            for e in self.envs:
                e.reset()
            return [e.fair_array(0) for e in self.envs], [e.fair_array(1) for e in self.envs]
        for c in self.conns:
            c.send(("reset",))
        pairs = [None] * self.count
        for indices, c in zip(self.slices, self.conns):
            for i, pair in zip(indices, c.recv()):
                pairs[i] = pair
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def step(self, actions_a, actions_b):
        if self.envs is not None:
            return [_step(e, a, b) for e, a, b in zip(self.envs, actions_a, actions_b)]
        for indices, c in zip(self.slices, self.conns):
            c.send(("step", [actions_a[i] for i in indices], [actions_b[i] for i in indices]))
        out = [None] * self.count
        for indices, c in zip(self.slices, self.conns):
            for i, r in zip(indices, c.recv()):
                out[i] = r
        return out

    def close(self):
        if self.envs is None:
            for c in self.conns:
                c.send(("close",))
            for p in self.procs:
                p.join(timeout=5)
