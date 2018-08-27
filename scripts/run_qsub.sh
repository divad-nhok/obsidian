#!/bin/bash
app=mason
dir_exp=/project/RDS-FSC-obsidian-RW/obsidian-dk/experiments/07_18_2018
fpath_script=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/run_$app.sh
fpath_script_pbs=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/standard_experiment/$app.pbs
fpath_current_jobs=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/current_jobs
queue_default=defaultQ
project_default=RDS-FEI-Bayes_BMC-RW
queue_bgh=alloc-dm
project_bgh=RDS-FSC-BGH-RW

project=$project_default
queue=$queue_default

all_job_nums=""
#declare -a num_arr=(01 02 03 04 05 06 07 08 09 10 11 12 13)
declare -a num_arr=(02 03 04 05 06 07 08 09 10 11 12 13)
for num in "${num_arr[@]}"
do
	echo $num
	dir_exp_num=$dir_exp/$num
	$fpath_script $dir_exp_num $fpath_script_pbs $project $queue
done

echo $all_jobs_nums >> $fpath_current_jobs
