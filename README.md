# Sharpening DAggers

Code, Experimental Configurations, and Visualization Tools for "Sharpening DAggers." Spins up both toy environments and
simulated robosuite environments, and provides utilities for collecting demonstrations using various "algorithmic
experts," from human user inputs, and from reinforcement learned policies (code for training these also included).

**Core Question**: How does the *type, or quality* of interventions (corrections, annotation) affect interactive
imitation learning algorithms like DAgger, LazyDAgger, or ThriftyDAgger?

*Side Questions*:
- Can we identify a clear way of scoring/ranking various demonstrations given an existing policy, task, and environment?
- Can we *elicit* better demonstrations from human users using this score? 

---

## Quickstart

Clones `thriftydagger` (TODO: rename repo) to the working directory, then walks through dependency setup, mostly
using the `environment-<arch>.yaml` files.

### Shared Environments (ILIAD Cluster)

Conda environment `sharp-daggers` already exists on the ILIAD cluster, and should work for any GPU-capable nodes. The 
only necessary steps to take are cloning this repository, and activating the environment.

### Local CPU Development (Mac OS)

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path. Use the `-cpu`
environment file.

```bash
git clone https://github.com/madelineliao/thriftydagger
cd thriftydagger
conda env create -f environments/environment-cpu.yaml
conda activate sharp-daggers
```

--- 

## Experiments

### Reach2D experiments with BC

1. Generate data: ```./scripts/generate_reach2d_data/{desired experiment name here, see folder for options} ```
2. Train BC model: ```./scripts/reach2d_exps/bc_oracle.sh```
3. Training metrics and model checkpoints will be saved to `./out/{exp name here}`

### Reach2D experiments with HGDagger: 
In-progress

--- 

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary, but is documented in the
case that building from conda `.yaml` breaks in the future).

Generally, if you're just trying to run/use this code, look at the Quickstart section above.

### Local CPU Environment (Mac OS)

```bash
conda create --name sharp-daggers python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
conda install ipython

pip install black gym isort matplotlib pyyaml

# Follow instructions here to install mujoco-py: https://github.com/openai/mujoco-py#install-mujoco
# < Assumes mujoco binaries are at ~/.mujoco/mujoco210 > (can copy from `/sailhome/siddk/.mujoco`)
pip install 'mujoco-py<2.2,>=2.1'

# Install Robosuite
pip install robosuite
```

### GPU & Cluster Environments (ILIAD Cluster - CUDA 11.1, PyTorch 1.8)

```bash
conda create --name sharp-daggers python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install ipython

pip install black gym isort matplotlib pyyaml

# Follow instructions here to install mujoco-py: https://github.com/openai/mujoco-py#install-mujoco
# < Assumes mujoco binaries are at ~/.mujoco/mujoco210 >
pip install 'mujoco-py<2.2,>=2.1'

# Install Robosuite
pip install robosuite
```
