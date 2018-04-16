#!/usr/bin/env python3

# script to generate sensor readings CSVs from prior forward model samples

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(prog = 'gen_readings', description = 'Convert forward model observations to readings CSVs')
parser.add_argument('keys', metavar = 'K', type=str, nargs = '+', help = 'Keys for sensor readings')
parser.add_argument('--parent_dir', dest='parent_dir', default = '.', help = 'Path of parent directory')
parser.add_argument('--samples_fname', dest='samples_fname', default = 'output.npz', help = 'Name of samples file')
parser.add_argument('--idx', dest = 'idx', default = 0, help = 'Index of row to take sample from')
parser.add_argument('--mu', dest = 'mu', default = 0, help = 'Mean of noise')
parser.add_argument('--sigma', dest = 'sigma', default = 1, help = 'Variance of noise')
parser.add_argument('--seed', dest = 'seed', default = 99, help = 'Random seed')

args = parser.parse_args()

fpath = os.path.join(args.parent_dir, args.samples_fname)
samples = np.load(fpath)
file_extension = '.csv'

np.random.seed(args.seed)
for key in args.keys:
    sensor_samples = samples[key]
    fwd_model_obs = sensor_samples[args.idx, :]
    e = np.random.normal(args.mu, args.sigma, fwd_model_obs.size)
    new_obs = fwd_model_obs + e
    fname = key + file_extension
    fpath = os.path.join(args.parent_dir, fname)
    np.savetxt(fpath, new_obs)
