"""Writes a train.py checkpoint as a policy file for zero (zero/policy/DuelPolicy.h).

    .venv/bin/python export_policy.py runs/p15b/final.pt runs/p15b/policy.zdp
"""
import struct
import sys

import torch


def export(checkpoint, path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensors = {k: v for k, v in state.items() if not k.startswith("value.")}
    with open(path, "wb") as f:
        f.write(b"ZDP1")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            data = t.detach().to(torch.float32).contiguous()
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<I", data.dim()))
            f.write(struct.pack(f"<{data.dim()}I", *data.shape))
            f.write(data.numpy().tobytes())


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
