#!/bin/bash

queue_bgh=alloc-dm
project_bgh=RDS-FSC-BGH-RW

queue_default=defaultQ
project_default=RDS-FEI-Bayes_BMC-RW

project=$project_bgh
queue=$queue_bgh

dir_data=/project/RDS-FSC-obsidian-RW/obsidian-dk/datasets/gascoyne_rs

port=7050
threads=5 #essentially CPUs per shard job divided by number of sensors, then rounded down to integer
clear_old_job_files=true

obsidian_walltime_hours=72
obsidian_finish_buffer_hours=10
obsidian_input_walltime_seconds=$(($obsidian_walltime_hours * 60 * 60))
shard_pbs_walltime=$obsidian_walltime_hours:00:00
obsidian_pbs_walltime=$(($obsidian_walltime_hours + $obsidian_finish_buffer_hours)):00:00
obsidian_anneal_length=1
#obsidian_input_walltime_seconds=300
#obsidian_pbs_walltime=00:30:00
obsidian_input_stacks=4
obsidian_input_chains=8
obsidian_input_proposal=Normal
pickaxe_walltime=12:00:00
mason_walltime=12:00:00
#pickaxe_walltime=00:30:00
#mason_walltime=00:30:00
email=david.kohn@sydney.edu.au 

pickaxe_burnin=1000
pickaxe_nthin=100

dir_obsidian=/project/RDS-FSC-obsidian-RW/obsidian-dk
dir_builds=$dir_obsidian/builds
specific_build=build_test_ports
log_file=details.log
dir_rds=/rds/PRJ-SIH4HPC/obsidian/experiments

run_obsidian=true
run_mason=true
run_pickaxe=true

dir_project=`pwd`
input_obsidian_file=input.obsidian
dir_build=$dir_builds/$specific_build
dir_num=`basename "$dir_project"`
dir_date=$(basename $(dirname $dir_project))
dir_target=$dir_date/$dir_num
dir_rds_experiment=$dir_rds/$dir_date/$dir_num
all_jobs=""

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
cp -n ${dir_data}/*.csv .

# make rds dir
mkdir -p $dir_rds_experiment

if [ "$run_obsidian" = true ]
then

	obsidian_input_walltime_line_no=`grep -oPn "wallTime = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_walltime_line_no}s/.*/wallTime = ${obsidian_input_walltime_seconds}/" $input_obsidian_file
	obsidian_input_stacks_line_no=`grep -oPn "stacks = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_stacks_line_no}s/.*/stacks = ${obsidian_input_stacks}/" $input_obsidian_file
	obsidian_input_chains_line_no=`grep -oPn "chains = \K([0-9]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_chains_line_no}s/.*/chains = ${obsidian_input_chains}/" $input_obsidian_file
	obsidian_input_proposal_line_no=`grep -oPn "distribution = \K([a-zA-Z]+)" $input_obsidian_file | cut -f1 -d:`
	sed -i "${obsidian_input_proposal_line_no}s/.*/distribution = ${obsidian_input_proposal}/" $input_obsidian_file

	# queue obsidian
	obsidian_pbs=`qsub -P $project_bgh -q $queue_bgh -l select=1:ncpus=2:mem=16GB,walltime=$obsidian_pbs_walltime -m e -M $email -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,port=$port,anneal_length=$obsidian_anneal_length obsidian.pbs`
	obsidian_job_id=`echo $obsidian_pbs | grep -oP "\K([0-9]+)"`
	all_jobs+=" ${obsidian_job_id}"
	echo "obsidian: ${obsidian_job_id}" >> $log_file

	# queue shards
	no_shards=$(($obsidian_input_stacks * $obsidian_input_chains / $obsidian_input_threads + 1))
	#no_shards=1
	for value in $(seq 1 $no_shards)
	do
		shard_pbs=`qsub -P $project_bgh -q $queue_bgh -l select=1:ncpus=16:mem=16GB,walltime=$shard_pbs_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,port=$port,threads=$threads -W depend=after:$obsidian_job_id shard.pbs`
		shard_job_id=`echo $shard_pbs | grep -oP "\K([0-9]+)"`
		echo "${shard_job_id}" >> $log_file
		all_jobs+=" ${shard_job_id}"
	done
fi

#when the obsidian job is finished, run diagnostics on obsidian.pbs.e*
fname_obsidian_err=obsidian.pbs.e$obsidian_job_id
path_obsidian_err=$dir_project/$fname_obsidian_err

parse_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=04:00:00 -v dir_project=$dir_project,fname_obsidian_err=$fname_obsidian_err -W depend=afterok:$obsidian_job_id parse_error_log.pbs` 
parse_job_id=`echo $parse_pbs | grep -oP "\K([0-9]+)"`
all_jobs+=" ${parse_job_id}"

#when the diagnostics are finished, move the obsidian.pbs.e* to rds dir
cleanup_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$path_obsidian_err,path_out=$dir_rds_experiment -W depend=afterok:$parse_job_id move_files.pbs`
cleanup_job_id=`echo $cleanup_pbs | grep -oP "\K([0-9]+)"`
all_jobs+=" ${cleanup_job_id}"

if [ "$run_pickaxe" = true ]
then
	pickaxe_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=32gb,walltime=$pickaxe_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,burnin=$pickaxe_burnin,nthin=$pickaxe_burnin -W depend=afterok:$obsidian_job_id pickaxe.pbs`
	pickaxe_job_id=`echo $pickaxe_pbs | grep -oP "\K([0-9]+)"`
	echo "pickaxe: ${pickaxe_job_id}" >> $log_file
	all_jobs+=" ${pickaxe_job_id}"
fi

#when pickaxe is finished, run diagnostics on output.npz
#qsub run_diagnostics.pbs

if [ "$run_mason" = true ]
then
	mason_pbs=`qsub -q $queue_default -P $project_default -l select=1:ncpus=4:mem=16GB,walltime=12:00:00 -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file -W depend=afterok:$pickaxe_job_id mason.pbs`
	mason_job_id=`echo $mason_pbs | grep -oP "\K([0-9]+)"`
	echo "mason: ${mason_job_id}" >> $log_file
	all_jobs+=" ${mason_job_id}"
fi

#when mason is finished, save mean layer voxels
mean_posterior_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=06:00:00 -v dir_voxels=$dir_project -W depend=afterok:$mason_job_id get_mean_posterior_layers.pbs`
cleanup_job_id=`echo $cleanup_pbs | grep -oP "\K([0-9]+)"`
all_jobs+=" ${cleanup_job_id}"

#when mason is finished, move voxels to rds dir/voxels and move output.npz to rds dir
cleanup_mason_pbs=`qsub -q dtq -P $project_default -l select=1:ncpus=1:mem=16gb,walltime=24:00:00 -v path_in=$path_voxels,path_out=$dir_rds_experiment -W depend=afterok:$parse_job_id cleanup_mason.pbs`
cleanup_mason_job_id=`echo $cleanup_mason_pbs | grep -oP "\K([0-9]+)"`
all_jobs+=" ${cleanup_mason_pbs}"

echo $all_jobs > current_jobs
