#!/bin/bash
queue_bgh=alloc-dm
project_bgh=RDS-FSC-BGH-RW

queue_default=defaultQ
project_default=RDS-FEI-Bayes_BMC-RW

project=$project_bgh
queue=$queue_bgh
#project=$project_default
#queue=$queue_default

dir_obsidian=/project/RDS-FSC-obsidian-RW/obsidian-dk
dir_project=`pwd`
dir_pbs_scripts=$dir_obsidian/scripts/standard_experiment
dir_data=$dir_obsidian/datasets/gascoyne_rs_v4_2018_08_16
dir_datasets=$dir_obsidian/datasets
specific_build=build-adaptive-multivar
dir_builds=$dir_obsidian/builds
dir_rds=/rds/PRJ-SIH4HPC/obsidian/experiments

port=7025
threads=5 #essentially CPUs per shard job divided by number of sensors, then rounded down to integer
clear_old_job_files=false

obsidian_walltime_hours=24
obsidian_finish_buffer_hours=10
obsidian_input_walltime_seconds=$(($obsidian_walltime_hours * 60 * 60))
shard_pbs_walltime=$obsidian_walltime_hours:00:00
obsidian_pbs_walltime=$(($obsidian_walltime_hours + $obsidian_finish_buffer_hours)):00:00
obsidian_anneal_length=1
#obsidian_input_walltime_seconds=300
#obsidian_pbs_walltime=00:30:00
obsidian_input_stacks=4
obsidian_input_chains=16
obsidian_input_proposal=AdaptiveMulti
pickaxe_walltime=12:00:00
mason_walltime=12:00:00
#pickaxe_walltime=00:30:00
#mason_walltime=00:30:00
email=david.kohn@sydney.edu.au 

no_shards=$(($obsidian_input_stacks * $obsidian_input_chains / $threads + 1))
#no_shards=1

pickaxe_burnin=1000
pickaxe_nthin=100
pickaxe_output_file=$dir_project/output.npz

log_file=$dir_project/details.log

# run individual jobs of your choice, specify below
run_custom=true
# run all the subscripts and jobs
run_all=false
# run none of the subscripts and jobs
# if both run_all and run_none are true, nothing will be run
run_none=false
# copy data files; set to false if you don't want custom files overwritten
copy_data_files=false

if [ "$run_custom" = true ]; then
	run_obsidian=false
	run_obsidian_diagnostics=true
	run_move_obsidian_error_log=false
	run_mason=false
	run_pickaxe=false
	run_get_mean_layer_voxels=false
	run_cleanup_mason=false
fi

if [ "$run_all" = true ]; then
	run_obsidian=true
	run_obsidian_diagnostics=true
	run_move_obsidian_error_log=true
	run_mason=true
	run_pickaxe=true
	run_get_mean_layer_voxels=true
	run_cleanup_mason=true
fi

if [ "$run_none" = true ]; then
	run_obsidian=false
	run_obsidian_diagnostics=false
	run_move_obsidian_error_log=false
	run_mason=false
	run_pickaxe=false
	run_get_mean_layer_voxels=false
	run_cleanup_mason=false
fi

input_obsidian_file=input.obsidian
dir_build=$dir_builds/$specific_build
dir_num=`basename "$dir_project"`
dir_date=$(basename $(dirname $dir_project))
dir_target=$dir_date/$dir_num
dir_rds_experiment=$dir_rds/$dir_date/$dir_num
all_jobs=""

cd $dir_project

# write build and directory to log file
echo "experiment started at: `date`" > $log_file
echo "directory: `pwd`" >> $log_file
echo "build: ${build}" >> $log_file
echo "data: ${dir_data}" >> $log_file

if [ "$clear_old_job_files" = true ]
then
	rm $dir_project/*.pbs.*
	rm $dir_project/core.*
fi

# copy data files in
if [ "$copy_data_files" = true ]
then
	cp -n ${dir_data}/*.csv $dir_project
	cp -n ${dir_data}/input.obsidian $dir_project
	cp -n ${dir_datasets}/obsidian_config $dir_project
fi

# make rds dir
mkdir -p $dir_rds_experiment

# make project dir
mkdir -p $dir_project

if [ "$run_obsidian" = true ]
then

	obsidian_script=$dir_pbs_scripts/obsidian.pbs
	obsidian_input_walltime_line_no=`grep -oPn "wallTime = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_walltime_line_no}s/.*/wallTime = ${obsidian_input_walltime_seconds}/" $input_obsidian_file
	obsidian_input_stacks_line_no=`grep -oPn "stacks = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_stacks_line_no}s/.*/stacks = ${obsidian_input_stacks}/" $input_obsidian_file
	obsidian_input_chains_line_no=`grep -oPn "chains = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_chains_line_no}s/.*/chains = ${obsidian_input_chains}/" $input_obsidian_file
	obsidian_input_proposal_line_no=`grep -oPn "distribution = \K([a-zA-Z]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_proposal_line_no}s/.*/distribution = ${obsidian_input_proposal}/" $input_obsidian_file

	# queue obsidian
	obsidian_pbs=`qsub -P $project -q $queue -l select=1:ncpus=2:mem=16GB,walltime=$obsidian_pbs_walltime -m e -M $email -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,port=$port,anneal_length=$obsidian_anneal_length $obsidian_script`
	obsidian_job_id=`echo $obsidian_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${obsidian_job_id}"
	echo "obsidian: ${obsidian_job_id}" >> $log_file

	# queue shards
	shard_script=$dir_pbs_scripts/shard.pbs
	for value in $(seq 1 $no_shards)
	do
		shard_pbs=`qsub -P $project -q $queue -l select=1:ncpus=16:mem=16GB,walltime=$shard_pbs_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,port=$port,threads=$threads -W depend=after:$obsidian_job_id $shard_script`
		shard_job_id=`echo $shard_pbs | grep -oP "\K([0-9]+)"`
		echo "${shard_job_id}" >> $log_file
		all_jobs+=" ${shard_job_id}"
	done
fi

#when the obsidian job is finished, run diagnostics on obsidian.pbs.e*
if [ "$run_obsidian_diagnostics" = true ] 
then
	script=$dir_pbs_scripts/parse_error_log.pbs
	if [ "$run_obsidian" = true ]
	then
		fname_obsidian_err=obsidian.pbs.e$obsidian_job_id
		path_obsidian_err=$dir_project/$fname_obsidian_err
		parse_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=04:00:00 -v dir_project=$dir_project,fname_obsidian_err=$fname_obsidian_err -W depend=afterok:$obsidian_job_id $script` 
	else
		fname_obsidian_err="obsidian.pbs.e*"
		path_obsidian_err=$dir_project/$fname_obsidian_err
		parse_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=04:00:00 -v dir_project=$dir_project,fname_obsidian_err=$fname_obsidian_err $script` 
	fi
	parse_job_id=`echo $parse_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${parse_job_id}"
fi

#when the diagnostics are finished, move the obsidian.pbs.e* to rds dir
if [ "$run_move_obsidian_error_log" = true ] 
then
	script=$dir_pbs_scripts/move_files.pbs
	if [ "$run_obsidian_diagnostics" = true ] 
	then
		cleanup_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$path_obsidian_err,path_out=$dir_rds_experiment -W depend=afterok:$parse_job_id $script`
	else
		fname_obsidian_err=obsidian.pbs.e$obsidian_job_id
		path_obsidian_err=$dir_project/$fname_obsidian_err
		cleanup_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$path_obsidian_err,path_out=$dir_rds_experiment $script`
	fi
	cleanup_job_id=`echo $cleanup_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${cleanup_job_id}"
fi

if [ "$run_pickaxe" = true ]
then
	script=$dir_pbs_scripts/pickaxe.pbs
	if [ "$run_obsidian" = true ]
	then
		pickaxe_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=32gb,walltime=$pickaxe_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,burnin=$pickaxe_burnin,nthin=$pickaxe_burnin,output_file=$pickaxe_output_file -W depend=afterok:$obsidian_job_id $script`
	else
		pickaxe_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=32gb,walltime=$pickaxe_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,burnin=$pickaxe_burnin,nthin=$pickaxe_burnin,output_file=$pickaxe_output_file $script`
	fi
	pickaxe_job_id=`echo $pickaxe_pbs | grep -oP "\K([0-9]+)"`
	echo "pickaxe: ${pickaxe_job_id}" >> $log_file
	all_jobs+=" ${pickaxe_job_id}"
fi

#when pickaxe is finished, run diagnostics on output.npz
#qsub run_diagnostics.pbs

if [ "$run_mason" = true ]
then
	script=$dir_pbs_scripts/mason.pbs
	if [ "$run_pickaxe" = true ]
	then
		mason_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=16GB,walltime=12:00:00 -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,pickaxe_file=$pickaxe_output_file -W depend=afterok:$pickaxe_job_id $script`
	else
		mason_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=16GB,walltime=12:00:00 -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,pickaxe_file=$pickaxe_output_file $script`
	fi
	mason_job_id=`echo $mason_pbs | grep -oP "\K([0-9]+)"`
	echo "mason: ${mason_job_id}" >> $log_file
	all_jobs+=" ${mason_job_id}"
fi

#when mason is finished, save mean layer voxels
if [ "$run_get_mean_layer_voxels" = true ]
then
	script=$dir_pbs_scripts/get_mean_posterior_layers.pbs
	dir_voxels=$dir_project/voxels
	if [ "$run_mason" = true ]
	then
		mean_posterior_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=06:00:00 -v dir_voxels=$dir_voxels -W depend=afterok:$mason_job_id $script`
	else
		mean_posterior_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=06:00:00 -v dir_voxels=$dir_voxels $script`
	fi
	cleanup_job_id=`echo $cleanup_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${cleanup_job_id}"
fi

#when mason is finished, move voxels to rds dir/voxels and move output.npz to rds dir
if [ "$run_cleanup_mason" = true ]
then
	script=$dir_pbs_scripts/cleanup_mason.pbs
	dir_voxels=$dir_project/voxels
	if [ "$run_get_mean_layer_voxels" = true ]
	then
		cleanup_mason_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$dir_voxels,path_out=$dir_rds_experiment -W depend=afterok:$cleanup_job_id $script`
	else
		cleanup_mason_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$dr_voxels,path_out=$dir_rds_experiment $script`
	fi
	cleanup_mason_job_id=`echo $cleanup_mason_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${cleanup_mason_pbs}"
fi

echo $all_jobs > $dir_project/current_jobs
