#!/bin/bash
#PBS -P RDS-FEI-Bayes_BMC-RW
#PBS -q defaultQ
#PBS -l select=1:ncpus=4:mem=16GB,walltime=01:00:00
#PBS -m e
#PBS -M david.kohn@sydney.edu.au
ulimit -c unlimited
module load gcc/4.9.3
cd /project/RDS-FSC-obsidian-RW/obsidian-dk/experiments/08_08_2018/01
/project/RDS-FSC-obsidian-RW/obsidian-dk/builds/build2/mason -l-4 -x 32 -y 32 -z 32 -p prior.npz -o prior_voxels.npz
