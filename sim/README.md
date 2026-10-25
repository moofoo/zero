# The duel simulator and self-play training

- `DuelSim.cpp`: the headless simulator (bot-ai ticket 41), built as `duelsim` (replay and scripted CLI) and
  `libduelsim` (the C API used by Python).
- `duelsim.py`: `DuelEnv`, the ctypes environment. `python duelsim.py` runs a smoke test and prints the speed.
- `train.py`: PPO self-play on fair observations (ticket 42). Runs go to `runs/<name>/` (`log.jsonl`, checkpoints).

## On this laptop

    cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release && cmake --build ../build -j --target duelsim_lib
    python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
    nice -n 19 .venv/bin/python train.py --envs 16 --steps 1500000 --run check1

## On the P15 (Windows 11, RTX A5000)

### What has to be done at the P15 itself

1. In an administrator PowerShell: `wsl --install -d Ubuntu`, restart, and create the Ubuntu user when it asks. The
   NVIDIA Windows driver already gives WSL the GPU; don't install a Linux driver inside Ubuntu.
2. Mirrored networking, so the laptop can reach Ubuntu at the P15's own address: create `%UserProfile%\.wslconfig`
   with

       [wsl2]
       networkingMode=mirrored

   then `wsl --shutdown` in PowerShell and open Ubuntu again.
3. In Ubuntu, an SSH server that the laptop's key can log in to:

       sudo apt update && sudo apt install -y openssh-server
       sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config && sudo service ssh start
       mkdir -p ~/.ssh && nano ~/.ssh/authorized_keys     # paste the laptop's ~/.ssh/id_ed25519.pub

   and in an administrator PowerShell, open the port:
   `New-NetFirewallRule -DisplayName "WSL ssh" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow`.
   (Port 2222 keeps clear of any SSH server on Windows itself.) `sudo service ssh start` is needed again after the
   P15 restarts.
4. Check from the laptop: `ssh -p 2222 <ubuntu user>@<P15 name or address> nvidia-smi`.

### The rest, over SSH from the laptop

    sudo apt install -y build-essential cmake git python3-venv
    git clone -b duel-sim https://github.com/moofoo/zero.git ~/zero
    cmake -S ~/zero -B ~/zero/build -DCMAKE_BUILD_TYPE=Release && cmake --build ~/zero/build -j --target duelsim_lib
    cd ~/zero/sim && python3 -m venv .venv && .venv/bin/pip install torch numpy   # the CUDA build
    .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
    mkdir -p ~/chaos/zones/Chaos    # then from the laptop:
    # scp -P 2222 ~/newcode/chaos-offline/work/bots/zones/Chaos/*.lvl <user>@<P15>:chaos/zones/Chaos/

Training, detached so it survives the SSH session:

    cd ~/zero/sim && nohup .venv/bin/python train.py --device cuda --envs 64 --steps 30000000 --run p15a \
        --map-dir ~/chaos > runs/p15a.out 2>&1 &

The environments step one after another on one CPU core (about 1,000 steps a second on the laptop), so 30 million
steps take about 8 hours, and the network is small enough that the GPU may not beat the CPU (`--device cpu`; worth a
short timing of both). Stepping the environments in worker processes, one a core, is the speed-up if that's too slow.
