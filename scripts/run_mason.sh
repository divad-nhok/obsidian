#!/bin/bash

dir_project=$1
mason_file=$2
project=$3
queue=$4
cd $dir_project

dir_obsidian=/project/RDS-FSC-obsidian-RW/obsidian-dk
dir_builds=$dir_obsidian/builds
specific_build=build_test_ports
dir_build=$dir_builds/$specific_build

log_file=$dir_project/details.log

mason_walltime=20:00:00
mason_memory=32gb
pickaxe_file=$dir_project/output2.npz

all_jobs=""

mason_pbs=`qsub -q $queue -P $project -l select=1:ncpus=4:mem=$mason_memory,walltime=$mason_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,pickaxe_file=$pickaxe_file $mason_file`
mason_job_id=`echo $mason_pbs | grep -oP "\K([0-9]+)"`
echo "mason: ${mason_job_id}" >> $log_file
echo $mason_job_id
