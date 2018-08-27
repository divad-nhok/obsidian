#!/bin/bash
mkdir -p $2
cp $1/*.csv $2/
cp $1/*.pbs $2/
cp $1/input.obsidian $2/
cp $1/obsidian_config $2/
cp $1/run_experiment.sh $2/
