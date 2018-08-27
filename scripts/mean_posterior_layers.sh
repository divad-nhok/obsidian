#!/bin/bash
ulimit -c unlimited
module load python/3.6.5
dir_project_vox=$1
dir_project=$2
python_script_get_mean_posterior=$3

echo $dir_project_vox
echo $dir_project
echo $python_script_get_mean_posterior

cd $dir_project
python $python_script_get_mean_posterior $dir_project_vox $dir_project
