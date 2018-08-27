#!/bin/bash
project=RDS-FEI-Bayes_BMC-RW
queue=dtq
select="select=1:ncpus=2:mem=16GB,walltime=24:00:00"
voxels_str_project=voxels
voxels_str_rds=voxels_thin100
experiment_date=07_18_2018
dir_project=/project/RDS-FSC-obsidian-RW/obsidian-dk/experiments
dir_rds=/rds/PRJ-SIH4HPC/obsidian/experiments
dir_project_exp=$dir_project/$experiment_date
dir_rds_exp=$dir_rds/$experiment_date
declare -a num_arr=(02 03 04 05 06 07 08 09 10 11 12 13)
for num in "${num_arr[@]}"
do
	echo $num
	dir_project_num=$dir_project_exp/$num/$voxels_str_project/*
	echo $dir_project_num
	dir_rds_num=$dir_rds_exp/$num/$voxels_str_rds
	echo $dir_rds_num
	qsub -P $project -q $queue -l $select -v dir_project=$dir_project_num,dir_rds=$dir_rds_num data_transfer_args.pbs
done
