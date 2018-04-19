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
import os

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

def plot_sensor(sensors, readings, chain, sample=None, units='unknown units', plot_fpath = None):
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
    print('gridshape: {}'.format(gridshape))
    xgrid = x.reshape(*gridshape)
    ygrid = y.reshape(*gridshape)
    dgrid = d.reshape(*gridshape)
    fgrid = f.reshape(*gridshape)

    # Contour map of residuals in f
    plt.subplot(2, 1, 1)
    # filled contours with real sensor data
    plt.contourf(xgrid, ygrid, dgrid, alpha=0.5)
    plt.colorbar()
    # line contours with forward model sensor data
    plt.contour(xgrid, ygrid, fgrid, colors='k')
    plt.xlabel("Eastings (m)")
    plt.ylabel("Northings (m)")

    # Histogram of residuals in f
    plt.subplot(2, 1, 2)
    plt.hist(d-f, bins=20)
    plt.xlabel("Data $-$ Model ({})".format(units))
    # Show
    if plot_fpath:
        plt.savefig('{}.png'.format(plot_fpath))
    else:
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

def load_data(parent_dir):
    fname_list = [
        ("magSensors", ".csv"),
        ("magReadings", ".csv"),
        ("gravSensors", ".csv"),
        ("gravReadings", ".csv"),
        ("output", ".npz")
    ]
    data_dict = {}
    for fname, extension in fname_list:
        data = None
        fpath = os.path.join(parent_dir, fname + extension)
        if extension == '.csv':
            data = np.loadtxt(fpath, delimiter = ',')
        elif extension == '.npz':
            data = np.load(fpath)
        data_dict[fname] = data
    return(data_dict)

def main_contours(
    data_dir = '',
    save_dir = ''
):
    """
    The main routine to run a suite of diagnostic plots
    """
    # load everything
    data_dict = load_data(data_dir)

    # tuple format = (sensor key, reading key, samples key, unit, plot file name)
    plot_key_tuple_list = [
        ('magSensors', 'magReadings', 'output', 'nT', 'mag_contours'),
        ('gravSensors', 'gravReadings', 'output', 'mgal', 'grav_contours')
    ]
    # Make a few plots of sensors
    for sensor_key, reading_key, output_key, unit, plot_name in plot_key_tuple_list:
        samples = data_dict.get(output_key)[reading_key]
        if samples.shape[1] > 0:
            plot_sensor(
                data_dict.get(sensor_key), 
                data_dict.get(reading_key), 
                samples, 
                units=unit,
                plot_fpath=os.path.join(save_dir, plot_name)
            )
            plt.clf()

def main_boundarymovie(parent_dir = ''):
    """
    Makes a movie of how the boundaries change as the chain samples
    """

    # load everything
    data_dict = load_data(parent_dir)
    samples = data_dict.get('output')
    gravSensors = data_dict.get('gravSensors')

    # Try fitting a few GP layers
    layer_labels = ['layer{}ctrlPoints'.format(i) for i in range(4)]
    for i in np.arange(0, 2500, 25):
        layer_pars = np.array([samples[ll][i] for ll in layer_labels]).reshape(4,5,5)
        fig = plt.figure()
        gp_predict(gravSensors, layer_pars, ((0.0, 2e+4), (0.0, 2e+4)))
        plt.savefig('boundary_movie_frame{:04d}.png'.format(i))
        plt.close()

def parameters_plot(
    samples, 
    key,
    rows, cols,
    fig_width, fig_height,
    fontsize = 20,
    plot_type = 'hist', # 'hist' or 'trace'
    plot_extension = 'png',
    save_dir = ''
):
    vals = samples[key]
    #print('vals.shape: {}'.format(vals.shape))
    n = vals.shape[1]
    iterations = range(vals.shape[0])
    total = rows * cols

    fig, axes = plt.subplots(
        rows, cols, 
        figsize = (fig_width, fig_height)
    )

    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top='off', bottom='off', left='off', right='off'
    )
    plt.grid(False)

    if plot_type == 'hist': 
        xstr = 'Parameter value bin'
        ystr = 'Frequency'
        title_str = 'Histogram for {}'
        save_str = 'hist-{}' + '.{}'.format(plot_extension)
    elif plot_type == 'trace':
        xstr = 'MCMC iteration'
        ystr = 'Parameter value'
        title_str = 'Trace plots for {}'
        save_str = 'trace-plots-{}' + '.{}'.format(plot_extension)

    plt.xlabel(
        xstr,
        fontsize = fontsize
    )
    plt.ylabel(
        ystr,
        fontsize = fontsize
    )

    plot_i = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if plot_i < n:
                #print(plot_i, row_idx, col_idx)
                x = vals[:, plot_i]
                if plot_type == 'hist':
                    axes[row_idx][col_idx].hist(x)
                elif plot_type == 'trace':
                    axes[row_idx][col_idx].plot(iterations, x)
            else:
                axes[row_idx][col_idx].set_visible(False)
            plot_i += 1

    #plt.tight_layout()
    # see https://github.com/pandas-dev/pandas/issues/9351 for tight_layout issues
    plt.title(
        title_str.format(key),
        fontsize = fontsize
    )
    save_path = os.path.join(
        save_dir, save_str.format(key)
    )
    plt.savefig(save_path)
    plt.clf()

if __name__ == "__main__":
    main_contours()
