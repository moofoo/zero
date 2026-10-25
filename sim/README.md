# The duel simulator and self-play training

- `DuelSim.cpp`: the headless simulator (bot-ai ticket 41), built as `duelsim` (replay and scripted CLI) and
  `libduelsim` (the C API used by Python).
- `duelsim.py`: `DuelEnv`, the ctypes environment. `python duelsim.py` runs a smoke test and prints the speed.
- `train.py`: PPO self-play on fair observations (ticket 42). Runs go to `runs/<name>/` (`log.jsonl`, checkpoints).

## On this laptop

    cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release && cmake --build ../build -j --target duelsim_lib
    python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
    nice -n 19 .venv/bin/python train.py --envs 16 --steps 1500000 --run check1

## On the P15 (Windows 11, RTX A5000): one-time setup

1. In an administrator PowerShell, install WSL with Ubuntu (`wsl --install -d Ubuntu`), then restart and create the
   Ubuntu user. The NVIDIA Windows driver already gives WSL the GPU; don't install a Linux driver inside it.
   Check from Ubuntu with `nvidia-smi`.
2. In Ubuntu:

       sudo apt update && sudo apt install -y build-essential cmake git python3-venv openssh-server
       git clone -b duel-sim https://github.com/moofoo/zero.git ~/zero
       cmake -S ~/zero -B ~/zero/build -DCMAKE_BUILD_TYPE=Release && cmake --build ~/zero/build -j --target duelsim_lib
       cd ~/zero/sim && python3 -m venv .venv && .venv/bin/pip install torch numpy   # the CUDA build
       .venv/bin/python -c "import torch; print(torch.cuda.is_available())"

3. The map: copy `chaos-offline/work/bots/zones/Chaos/*.lvl` from this laptop to `~/chaos/zones/Chaos/` on the P15,
   and pass `--map-dir ~/chaos` to `train.py`.
4. SSH from this laptop: in Ubuntu run `sudo service ssh start`. On Windows, forward a port to WSL
   (`netsh interface portproxy add v4tov4 listenport=2222 connectport=22 connectaddress=<WSL ip>`) and allow it
   through the firewall, or set WSL's networking to mirrored (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`)
   so that `ssh -p 22 <user>@<P15>` reaches Ubuntu directly. Put this laptop's public key in `~/.ssh/authorized_keys`.
5. Training: `cd ~/zero/sim && nohup .venv/bin/python train.py --device cuda --envs 64 --steps 30000000 --run p15a --map-dir ~/chaos &`.
   The environments step one after another on one CPU core (about 900 steps a second here), so 30 million steps
   take most of a night; stepping them in worker processes is the next speed-up if that's too slow.
