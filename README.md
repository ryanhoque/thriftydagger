## Setup
0. Create a conda environment from `thriftydagger.yaml`.
## Reach2D experiments with BC
1. Generate data: ```./scripts/generate_reach2d_data/{desired experiment name here, see folder for options} ```
2. Train BC model: ```./scripts/reach2d_exps/bc_oracle.sh```
3. Training metrics and model checkpoints will be saved to `./out/{exp name here}`

## Reach2D experiments with HGDagger: 
In-progress
