#!/bin/bash -l
## Name of the run
#SBATCH -J histnet-2
## Number of allocated nodes (Definitions differ between HPCs. Here it is close to CPU definition)
#SBATCH --nodes=1
## Tasks _per_ node (Here it is number of cpu cores by default)
#SBATCH --ntasks-per-node=10
## Memory _per_ node/task (Here it is memory per task)
#SBATCH --mem-per-cpu=5GB
## GPU allocation
#SBATCH --gres=gpu
## Max time of runtime (format HH:MM:SS)
#SBATCH --time=23:30:00
## Grant name for allocation
#SBATCH -A plgmiadl
## Partition specification of the HPC
#SBATCH -p plgrid-gpu-v100
## File for logging information
#SBATCH --output="logs/HPC/histnet-2-%J.log"
## File for logging errors
#SBATCH --error="logs/HPC/histnet-2-%J.err"

## Go into folder from which the script was called
cd $SLURM_SUBMIT_DIR

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
user=$USER

## Change this domain to you HPC provider
domain="@pro.cyfronet.pl"

## print tunneling instructions to --output
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -o ServerAliveInterval=300 -N -L $ipnport:$ipnip:$ipnport ${user}$domain
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport
    ------------------------------------------------------------------
    "

## GPU check
nvidia-smi

## Module loading
wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p data/micromamba
export MAMBA_ROOT_PREFIX=./data/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
micromamba create --name hpc_run --file Histology-Style-Transfer-Research/environment-linux.yml -y
micromamba activate hpc_run
micromamba install -c fastai opencv-python-headless

## start an ipcluster instance and launch jupyter server
tensorboard --logdir=runs --host=$ipnip --port=$ipnport &
python3 Histology-Style-Transfer-Research/HistNet/train.py