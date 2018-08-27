import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import visvis as vv
import geovis_notebook_version

def get_view(
	dir_voxels,
	voxel_number_list = None
):
	if not voxel_number_list:
		fname_voxels_list = [
			os.path.join(dir_voxels, f)
			for f in os.listdir(dir_voxels) 
			if os.path.isfile(os.path.join(dir_voxels, f))
			and 'voxel' in f
		]
	else:
		fname_voxels_list = [
			os.path.join(dir_voxels, 'voxels{}.npz'.format(num))
			for num in voxel_number_list
		]
	for idx, fn in enumerate(fname_voxels_list):
		if idx == 0:
			view = geovis_notebook_version.MasonView(fn)
		else:
			view.add_samples(geovis_notebook_version.MasonView(fn))
	return(view)

def get_mean_layer(view, layer_idx):
	layer_mean = view.meanlayer(layer_idx)
	ml = get_mean_layer()

len_argv = len(sys.argv)
voxel_number_list = None
if len_argv == 1:
	print("No voxel directory specified")
if len_argv >= 2:
	dir_voxels = sys.argv[1]
	print(dir_voxels)
if len_argv == 3:
	dir_out = sys.argv[2]
	print(dir_out)
if len_argv == 4:
	voxel_number_list = eval(sys.argv[3])
	print(voxel_number_list)

view = get_view(dir_voxels, voxel_number_list)

no_layers = np.shape(view.layers)[0]
for layer_idx in range(no_layers):
	layer_mean = view.meanlayer(layer_idx)
	fname = 'mean-posterior-layer{}'.format(layer_idx)
	out_path = os.path.join(dir_out, fname)
	np.save(out_path, layer_mean)
