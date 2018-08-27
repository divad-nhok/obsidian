#!/usr/bin/python3

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(prog = 'rename_pbs_scripts', description = 'Rename PBS scripts in a target directory')
parser.add_argument('target_dir', metavar = 'T', type=str, help = 'Path of target directory')

args = parser.parse_args()

obsidian_dir = "/project/RDS-FSC-obsidian-RW/obsidian-dk/"
search_str = "cd " + obsidian_dir
new_dir = search_str + args.target_dir + "\n"
target_dir = os.path.join(obsidian_dir, args.target_dir)

for f1 in os.listdir(target_dir):
	if f1.endswith(".pbs"):
		fpath = os.path.join(target_dir, f1)
		print(fpath)
		lines = []
		with open(fpath) as infile:
			for line in infile:
				if line.startswith(search_str):
					line = new_dir
				lines.append(line)
			with open(fpath, 'w') as outfile:
				for line in lines:
					outfile.write(line)
