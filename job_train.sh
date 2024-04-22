#!/bin/bash
#SBATCH --chdir .
#SBATCH --account digital_humans
#SBATCH --time=10:00:00
#SBATCH -o /cluster/courses/digital_humans/datasets/team_1/motion-diffusion-model/output_files/slurm_output_%j.out  # Change this line
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=14G
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
source /cluster/courses/digital_humans/datasets/team_1/conda_envs/bin/activate
conda activate mdm

# Run your experiment
python -m train.train_mdm --save_dir save/with_smplh --dataset GRAB --cond_mask_prob 0 --unconstrained

echo "Done."
echo FINISHED at $(date)