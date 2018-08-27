#!/bin/bash

dir_project=$1
pickaxe_file=$2
project=$3
queue=$4
cd $dir_project

dir_obsidian=/project/RDS-FSC-obsidian-RW/obsidian-dk
dir_builds=$dir_obsidian/builds
specific_build=build_test_ports
dir_build=$dir_builds/$specific_build

log_file=$dir_project/details.log

pickaxe_walltime=12:00:00
pickaxe_burnin=1000
pickaxe_nthin=100
pickaxe_output_fname=$dir_project/output2.npz

all_jobs=""

echo $dir_build

pickaxe_pbs=`qsub -q $queue -P $project -l select=1:ncpus=4:mem=32gb,walltime=$pickaxe_walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,burnin=$pickaxe_burnin,nthin=$pickaxe_burnin,output_file=$pickaxe_output_fname $pickaxe_file`
pickaxe_job_id=`echo $pickaxe_pbs | grep -oP "\K([0-9]+)"`
echo "pickaxe: ${pickaxe_job_id}" >> $log_file
echo $pickaxe_job_id
