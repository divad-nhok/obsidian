#!/bin/bash
dir_rds_parent=/rds/PRJ-SIH4HPC/obsidian/experiments/08_08_2018
script_copy_files=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/standard_experiment/move_files.pbs
pbs_script_get_mean_posterior=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/mean_posterior_layers.pbs
python_script_get_mean_posterior=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/get_mean_posterior.py
copy_data=false

queue_default=defaultQ
project_default=RDS-FEI-Bayes_BMC-RW
queue_bgh=alloc-dm
project_bgh=RDS-FSC-BGH-RW
project=$project_bgh
queue=$queue_bgh

dir_project_parent=/project/RDS-FSC-obsidian-RW/obsidian-dk/experiments/07_18_2018
mem=64gb

all_job_nums=""
#declare -a num_arr=(01 02 03 04 05 06 07 08 09 10 11 12 13)
#declare -a num_arr=(02 03 04 05 06 07 08 09 10 11 12 13)
declare -a num_arr=(01)
for num in "${num_arr[@]}"
do
	echo $num
	dir_rds=$dir_rds_parent/$num/voxels
	dir_project=$dir_project_parent/$num
	dir_project_vox=$dir_project/voxels

	if [ "$copy_data" = true ]
	then
		mkdir -p $dir_project_vox
		copy_files_pbs=`qsub -q dtq -P $project -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$dir_rds,path_out=$dir_project_vox $script_copy_files`
		copy_files_job_id=`echo $copy_files_pbs | grep -oP "\K([0-9]+)"`
	fi

	mean_posterior_pbs=`qsub -q $queue -P $project -l select=1:ncpus=1:mem=$mem,walltime=24:00:00 -v dir_project_vox=$dir_project_vox,dir_project=$dir_project,python_script_get_mean_posterior=$python_script_get_mean_posterior  $pbs_script_get_mean_posterior`
	mean_posterior_job_id=`echo $mean_posterior_pbs | grep -oP "\K([0-9]+)"`

done
