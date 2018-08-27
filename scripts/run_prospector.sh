#!/bin/bash

dir_obsidian=/project/RDS-FSC-obsidian-RW/obsidian-dk
dir_project=$dir_obsidian/experiments/08_08_2018/01
pbs_script=$dir_obsidian/scripts/standard_experiment/prospector.pbs
project=RDS-FSC-BGH-RW
queue=alloc-dm
dir_data=$dir_obsidian/datasets/gascoyne_config_two_layers_v2
dir_builds=$dir_obsidian/builds
specific_build=build_test_ports
dir_build=$dir_builds/$specific_build
log_file=$dir_project/details.log
pbs_script_name=prospector.pbs
pbs_script_path=/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/standard_experiment
pbs_script=$pbs_script_path/$pbs_script_name
walltime=01:00:00
compute=1:ncpus=4:mem=32gb
outputfile=prior.npz
nsamples=1000
fwdmodel=true

mkdir -p $dir_project
cd $dir_project

pbs_out=`qsub -q $queue -P $project -l select=$compute,walltime=$walltime -v dir_project=$dir_project,dir_build=$dir_build,log_file=$log_file,outputfile=$outputfile,nsamples=$nsamples,fwdmodel=$fwdmodel $pbs_script`
job_id=`echo $pbs_out | grep -oP "\K([0-9]+)"`
