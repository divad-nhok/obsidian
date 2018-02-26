#!/usr/bin/env python

"""
RS 2018/02/23:  Digesting Obsidian Output

Plots we want to see:
    Trace plots of key parameters like layer depths, rock properties
    Histograms of residuals between forward model values and data
    Contour maps of residuals between forward models and data
"""

import numpy as np
import matplotlib.pyplot as plt
import GPy


def autoshape(sensors):
    """
    Senses the shape of a flattened coordinate array by looking for the
    index at which coordinates start to repeat.  matplotlib contour maps
    have to have (x, y) in regular grid inputs, which is really annoying
    because the sensor grids are provided in terms of just (x, y) lists.
    :param sensors:  np.array of shape (N, 3) with the physical
        coordinates (x,y,z) of the sensors in meters; assumed to have a
        regular raster-like structure where N factors into (Nx, Ny)
    :return:  inferred shape (x, y) of coordinate array
    """
    # Take differences between successive oordinates in the list
    x, y, z = sensors.T
    mask_x = (np.abs(x[1:] - x[:-1]) > 1e-3)
    mask_y = (np.abs(y[1:] - y[:-1]) > 1e-3)
    # Find the minimum index at which a difference is found
    nx = np.min(np.arange(len(mask_x))[mask_x]) + 1
    ny = np.min(np.arange(len(mask_y))[mask_y]) + 1
    # Use whichever of these is nontrivial to reshape the list
    if nx == 1:
        return (-1, ny)
    elif ny == 1:
        return (nx, -1)

def plot_sensor(sensors, readings, chain, sample=None, units='unknown units'):
    """
    Plots the value of a sensor against the forward model:
        Contour plots of real data and forward models
        Contour plots of mean model residuals across the chain
        Histograms of model residuals across the chain
    :param sensors:  np.array of shape (N, 3) with the physical
        coordinates (x,y,z) of the sensors in meters; assumed to have a
        regular raster-like structure where N factors into (Nx, Ny)
    :param readings:  np.array of shape (N, ) with the observed
        sensor readings for the real dataset
    :param chain:  np.array of shape (M, N) with the synthetic
        sensor readings for each forward model
    :param sample:  int index of sample to grab from chain
        (defaults to None, which averages over the chain)
    :param units:  str describing the units of sensor readings
    """
    x, y, z = sensors.T
    d = readings - readings.mean()
    if sample is None:
        f = chain.mean(axis=0) - chain.mean()
    elif not isinstance(sample, int):
        print( "ERROR:  sample = {} is of type {}, not int".format(
            sample, sample.__class__)
        )
        return
    elif abs(sample) > len(chain):
        print("ERROR:  sample = {} not in range ({}, {})".format(
                sample, -len(chain), len(chain))
        )
        return
    else:
        f = chain[sample] - chain[sample].mean()

    # Reshape sensors and readings to an automatically detected grid shape
    gridshape = autoshape(sensors)
    xgrid = x.reshape(*gridshape)
    ygrid = y.reshape(*gridshape)
    dgrid = d.reshape(*gridshape)
    fgrid = f.reshape(*gridshape)

    # Contour map of residuals in f
    plt.subplot(2, 1, 1)
    plt.contourf(xgrid, ygrid, dgrid, alpha=0.5)
    plt.colorbar()
    plt.contour(xgrid, ygrid, fgrid, colors='k')
    plt.xlabel("Eastings (m)")
    plt.ylabel("Northings (m)")

    # Histogram of residuals in f
    plt.subplot(2, 1, 2)
    plt.hist(d-f, bins=20)
    plt.xlabel("Data $-$ Model ({})".format(units))
    # Show
    plt.show()

def gp_predict(sensors, layer_pars, bounds):
    """
    Bare-bones GP predictor for layer depths at a grid of locations.
    ...ugh I'm not going to finish that today.  :/
    :param sensors:  np.array of shape (N, 3) with the physical
        coordinates (x,y,z) of the sensors in meters
    :param layer_pars:  np.array of shape (Nl, Nx, Ny) containing the
        layer heights at the control points
    :param bounds:  np.array of layer bounds
    """
    # Boundaries of the region
    xmin, Lx = bounds[0][0], bounds[0][1] - bounds[0][0]
    ymin, Ly = bounds[1][0], bounds[1][1] - bounds[1][0]
    Xpred = sensors[:,:-1]

    gp_pred_list = [ ]
    for l, pars in enumerate(layer_pars):
        # Figure out all these auto-magical lengthscales
        nx, ny = pars.shape
        xLS = 0.5 * Lx/(nx - 0.99999)
        yLS = 0.5 * Ly/(ny - 0.99999)
        xvals = xmin + Lx*np.arange(nx)/(nx - 0.99999)
        yvals = ymin + Ly*np.arange(ny)/(ny - 0.99999)
        xg, yg = np.meshgrid(xvals, yvals)
        # Set up and fit a shitty GP, and add it to the stack
        k = GPy.kern.RBF(input_dim=2,
                         lengthscale=(xLS, yLS), ARD=True)
        X = np.vstack([xg.ravel(), yg.ravel()]).T
        Y = pars.reshape(-1,1)
        gp = GPy.models.GPRegression(X[:], Y, kernel=k)
        gp.Gaussian_noise.variance = 0.001
        gpmu, gpsig = gp.predict(Xpred)
        # gp.plot()
        # plt.show()
        gp_pred_list.append(gpmu)

    # Read out the formation at the surface
    gpz = np.array(gp_pred_list).reshape(len(layer_pars), len(Xpred))
    print("gpz.shape =", gpz.shape)
    print("gpz =", gpz)
    synthform = np.zeros(len(Xpred))
    for i, xy in enumerate(Xpred):
        for l in range(len(layer_pars)):
            if gpz[l,i] > 0:
                synthform[i] = l
                break
    print("synthform =", synthform)
    
    # Make a contour plot, because why not
    gridshape = autoshape(sensors)
    x, y = Xpred.T
    xg, yg = Xpred.reshape(2, *gridshape)
    zg = synthform.reshape(*gridshape)
    # plt.contourf(xg, yg, zg, alpha=0.5)
    # plt.colorbar()
    print("x =", x)
    print("y =", y)
    for l in range(len(layer_pars)):
        idx = (synthform == l)
        plt.plot(x[idx], y[idx], label='layer {}'.format(l), ls='None', marker='o')
    plt.legend()


def main_contours():
    """
    The main routine to run a suite of diagnostic plots
    """

    # Load everything
    magSensors = np.loadtxt("magSensors.csv", delimiter=',')
    magReadings = np.loadtxt("magReadings.csv", delimiter=',')
    gravSensors = np.loadtxt("gravSensors.csv", delimiter=',')
    gravReadings = np.loadtxt("gravReadings.csv", delimiter=',')
    samples = np.load("output.npz")

    # Make a few plots of sensors
    plot_sensor(magSensors, magReadings, samples['magReadings'], units='nT')
    plot_sensor(gravSensors, gravReadings, samples['gravReadings'], units='mgal')

def main_boundarymovie():
    """
    Makes a movie of how the boundaries change as the chain samples
    """
    # Load everything
    magSensors = np.loadtxt("magSensors.csv", delimiter=',')
    magReadings = np.loadtxt("magReadings.csv", delimiter=',')
    gravSensors = np.loadtxt("gravSensors.csv", delimiter=',')
    gravReadings = np.loadtxt("gravReadings.csv", delimiter=',')
    samples = np.load("output.npz")

    # Try fitting a few GP layers
    layer_labels = ['layer{}ctrlPoints'.format(i) for i in range(4)]
    for i in np.arange(0, 2500, 25):
        layer_pars = np.array([samples[ll][i] for ll in layer_labels]).reshape(4,5,5)
        fig = plt.figure()
        gp_predict(gravSensors, layer_pars, ((0.0, 2e+4), (0.0, 2e+4)))
        plt.savefig('boundary_movie_frame{:04d}.png'.format(i))
        plt.close()


if __name__ == "__main__":
    main_contours()
