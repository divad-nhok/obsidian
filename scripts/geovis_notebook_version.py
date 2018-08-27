#!/usr/bin/env python

"""
RS 2018/04/24:  Visualization of samples from geological prior/posterior

This rips off many visualization elements from bin/python/visWorld.
Looked short enough, and important enough, to be worth rewriting.
"""

import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import visvis as vv

plt.ioff()

vv.settings.preferredBackEnd = 'pyside'
vv.settings.figureSize=(560,420)

rockprop_names = ['Density', 'LogSusceptibility', 'ThermalConductivity',
                  'ThermalProductivity', 'LogResistivityX', 'LogResistivityY',
                  'LogResistivityZ', 'ResistivityPhase', 'PWaveVelocity']

class MasonView(object):
    """
    Holds geological data.
    """

    def __init__(self, npzfname):
        """
        :param inputFile:  numpy NPZ file from mason
        """
        # Load all the base properties from the file
        raw = np.load(npzfname)
        xyzres = np.array(raw['resolution'].flatten(), dtype=int)
        self.xres = xyzres[0]
        self.yres = xyzres[1]
        self.zres = xyzres[2]
        self.xbounds = raw['x_bounds'].flatten()
        self.ybounds = raw['y_bounds'].flatten()
        self.zbounds = raw['z_bounds'].flatten()
        self.layers = [raw[key] for key in raw if key.startswith('layer')]
        self.fbounds = [raw[key] for key in raw if key.startswith('boundary')]
        self.rockprops = { key: raw[key] for key in rockprop_names }
        self.samples = self.fbounds[0].shape[0]

        # Reshape to associate samples with 3-D voxel grids
        newshapeVox = np.concatenate([[-1], xyzres[::-1]])
        newshapeSurf = np.concatenate([[-1], xyzres[1::-1]])
        self.layers = [k.reshape(newshapeVox, order='f') for k in self.layers]
        self.fbounds = [k.reshape(newshapeSurf, order='f') for k in self.fbounds]
        self.rockprops = { k: v.reshape(newshapeVox, order='f')
                           for k, v in self.rockprops.items() }

    def add_samples(self, other):
        """
        :param other:  MasonView object with same shape as this one
        """
        # Status checks all good, so proceed
        npc = np.concatenate
        for i in range(len(self.layers)):
            self.layers[i] = npc([self.layers[i], other.layers[i]])
        for i in range(len(self.fbounds)):
            self.fbounds[i] = npc([self.fbounds[i], other.fbounds[i]])
        for k in self.rockprops:
            self.rockprops[k] = npc([self.rockprops[k], other.rockprops[k]])

    def show_layer_boundaries(self, sample):
        """
        Displays a 3-D rendering of boundary surfaces.
        :param sample: index of sample for which to plot boundaries
        """
        app = vv.use()
        vv.figure(1)
        X = np.linspace(self.xbounds[0], self.xbounds[1], self.xres)
        Y = np.linspace(self.ybounds[0], self.xbounds[1], self.yres)
        Z = np.linspace(self.zbounds[0], self.xbounds[1], self.zres)
        vv.xlabel('Eastings (m)')
        vv.ylabel('Northings (m)')
        vv.zlabel('Depth (m)')
        a = vv.gca()
        a.camera.fov = 70
        a.daspect = 1, 1, -1
        for i in range(len(self.layers)):
            C = plt.cm.jet(i/float(len(self.layers)))
            C = np.array([[[C[0], C[1], C[2]]]])
            m = vv.surf(X, Y, self.fbounds[i][sample], C)

        vv.ColormapEditor(a)
        app.Run()

    def rockprop(self, prop, sample):
        """
        Returns a voxelization of a rock property for a single sample.
        """
        return self.rockprops[prop][sample]

    def layerprop(self, layer, sample):
        """
        Returns a voxelization of layer membership for a single sample.
        """
        return self.layers[layer][sample]

    def meanrockprop(self, prop):
        """
        Returns a mean voxelized rock property over all samples.
        """
        return np.mean(self.rockprops[prop], axis=0)

    def meanlayer(self, layer):
        """
        Returns a mean voxelized layer membership over all samples.
        """
        return np.mean(self.layers[layer], axis=0)

    def meanlayer_all(self):
        """
        Returns a mean voxelized layer membership over all samples.
        """
        layidx = np.arange(len(self.layers))
        laywtd = [i*np.mean(self.layers[i], axis=0) for i in layidx]
        return np.sum(laywtd, axis=0)

    def show_vox(self, vfunc):
        """
        Displays a 3-D rendering of a voxelized property.
        :param vfunc: function accepting this MasonView instance and
            returning a 3-D np.array of some voxelized property
        """
        app = vv.use()
        vv.figure(1)
        vv.xlabel('Eastings (units)')
        vv.ylabel('Northings (units)')
        vv.zlabel('Depth (units)')
        a = vv.gca()
        a.camera.fov = 70
        a.daspect = 1, 1, -1
        vox = vfunc(self)
        t = vv.volshow(vox, cm=vv.CM_JET, renderStyle='ray')

        vv.ColormapEditor(a)
        app.Run()


if __name__ == "__main__":
    print("Initializing from", args.npzfname[0])
    view = MasonView(args.npzfname[0])
    for fn in args.npzfname[1:]:
        print("Adding samples from", fn)
        view.add_samples(MasonView(fn))
    for i in range(len(view.layers)):
        view.show_vox(lambda v: MasonView.meanlayer(v, i))
