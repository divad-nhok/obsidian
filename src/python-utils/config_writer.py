#!/usr/bin/env python

"""
RS 2018/01/24:  Script to write configuration files for Obsidian

An Obsidian run has a *lot* of config files.  It's now our job to try to
wrangle all the priors, etc., into the proper form.  Unfortunately this
requires some input of geology; our prior expectations on the number and
depth of layers at a particular point may depend strongly on location.
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from webmercator import mercator


def form_csv_fnames(layers, prop):
    """
    Returns a list of CSV filenames that describe, for each geological
    layer, the model property named by prop.
    :param layers: pd.DataFrame describing rock layers (see above)
    :param prop: text attribute name
    """
    return ["{}{}.csv".format(n.replace(' ',''), prop) for n in layers.name]

def select_volume(data, lng, lat, L):
    """
    Selects a volume to be rendered from a data suite.
    :param data:  pd.DataFrame with three columns 'lng', 'lat', 'val',
        where lng, lat are in decimal degrees and val contains readings
    :param lng:  longitude of modeled area centre in decimal degrees E
    :param lat:  latitude of modeled area centre in decimal degrees E
    :param L:  length of side of (square) modeled area in metres
    :return x:  x-coordinates of extracted data points in metres
    :return y:  y-coordinates of extracted data points in metres
    :return val:  labels for (x, y) point, in any format
    """
    # Project all latitudes and longitudes to Web Mercator coordinates,
    # selecting only those measurements lying in the volume to render.
    x0, y0 = mercator(lng, lat)
    x, y = mercator(np.array(data.lng, dtype=float),
                    np.array(data.lat, dtype=float)) - np.array([[x0], [y0]])
    idx = np.all([np.abs(x) < 0.5*L, np.abs(y) < 0.5*L], axis=0)
    x = x[idx] + 0.5*L
    y = y[idx] + 0.5*L
    v = data.val.values[idx]
    return x, y, v

def write_sensor_data(
   data, lng, lat, L, stag,
   rhdr="", shdr="", zval=-0.5, axthin=1,
   zkey=False
):
    """
    Given a dataframe with sensor readings at (lng, lat) locations,
    write output to Obsidian-readable "Readings" and "Sensors" files.
    "[stag]Sensors.csv" has three columns (x, y, z) of x-values;
    "[stag]Readings.csv" has a single column of y-values.
    :param data:  pd.DataFrame with three columns 'lng', 'lat', 'val',
        where lng, lat are in decimal degrees and val contains readings
    :param lng:  longitude of modeled area centre in decimal degrees E
    :param lat:  latitude of modeled area centre in decimal degrees E
    :param L:  length of side of (square) modeled area in metres
    :param stag:  short string describing the kind of data
    :param rhdr:  string header to write to top of Readings file
    :param shdr:  string header to write to top of Sensors file
    :param zval:  depth of sensors (assumed constant, +z is down)
    :param axthin:  int, take every (axthin)th reading along each axis
    :param zkey: Bool, whether there is a z-key for the dataframe `data`
    """
    # Project all latitudes and longitudes to Web Mercator coordinates,
    # selecting only those measurements lying in the volume to render.
    x0, y0 = mercator(lng, lat)
    x, y = mercator(data.lng, data.lat) - np.array([[x0], [y0]])
    idx = np.all([np.abs(x) < 0.5*L, np.abs(y) < 0.5*L], axis=0)
    x = x[idx] + 0.5*L
    y = y[idx] + 0.5*L
    v = data.val.values[idx]
    if zkey: z = data.z.values[idx]

    # Subsampling on a grid
    if axthin > 1:
        xkeep = sorted(np.unique(x))[::axthin]
        ykeep = sorted(np.unique(y))[::axthin]
        idx = np.array([xi in xkeep and yi in ykeep for xi, yi in zip(x, y)])
        x, y, v = x[idx], y[idx], v[idx]
        if zkey: z = z[idx]

    # Come up with transformed DataFrames so we can just use to_csv.
    # If we're doing FieldObs sensors those are understood to be at z = 0.
    if zval is None:
        sdata = pd.DataFrame(np.array([x, y])).T
    else:
	if not zkey:
		z = np.ones(x.shape) * zval
        sdata = pd.DataFrame(np.array([x, y, z])).T
    rdata = pd.DataFrame(np.array([v])).T

    # Write Readings file
    readings_fname = "{}Readings.csv".format(stag)
    with open(readings_fname, 'w') as outfile:
        outfile.write(rhdr)
        outfile.write(rdata.to_csv(index=False, header=False))

    # Write Sensors file
    sensors_fname = "{}Sensors.csv".format(stag)
    with open(sensors_fname, 'w') as outfile:
        outfile.write(shdr)
        outfile.write(sdata.to_csv(index=False, header=False))

def write_config(lng, lat, L, maxdepth, layers, H_IGRF,
                 grav_data, mag_data, field_data, rockpriormu, rockpriorcov):
    """
    Write a suite of configuration files for Obsidian describing the
    priors and settings for a likely configuration.
    :param lng:  longitude of modeled area centre in decimal degrees E
    :param lat:  latitude of modeled area centre in decimal degrees E
    :param L:  length of side of (square) modeled area in metres
    :param maxdepth:   maximum depth below surface to render in m
    :param layers:  pd.DataFrame w/columns (name, zmin, zmax) describing
        names of geological layers and bounds on where they occur
    :param H_IGRF:  3-tuple with (x,y,z) components of the ambient
        magnetic field, evaluated via the IGRF model
    :param grav_data:  pd.DataFrame with columns (id, val, lat, lng)
    :param mag_data:  pd.DataFrame with columns (id, val, lat, lng)
    :param field_data:  pd.DataFrame with columns (id, val, lat, lng)
    :param rockpriormu:  hash, indexed by layer name, of pd.Series
        instances with mean of multivariate Gaussian rock property prior
        -- layer names must match "layers" parameter or code will crash
    :param rockpriorcov:  hash, indexed by layer name, of pd.DataFrame
        instances with covariance of multivariate Gaussian rock prior
        -- layer names must match "layers" parameter or code will crash
    """
    # RS 2018/01/29:  I apologize in advance about the readability of the
    # following code.  If I want to generate config files that are totally
    # self-explanatory (via embedded comments) that means formatting huge
    # multi-line strings.  The least bad choice I could find was to exploit
    # the ability of multi-line strings to self-continue, to format all the
    # fields explicitly by keywords, and to not worry about > 80-char lines.

    # ========================================================================
    #                    Part 1:  The Master Config File
    # ========================================================================

    # ------------------------------------------------------------------------
    # General config file header

    config_output = (
        "################################################################################\n"
        "# Obsidian File                                                                #\n"
        "#                                                                              #\n"
        "# This file contains all the configuration information for an Obsidian         #\n"
        "# inversion. Ensure that any files referenced in the sections below are in the #\n"
        "#                                                                              #\n"
        "# same directory as this file, which must also be the directory from which     #\n"
        "# obsidian is run.                                                             #\n"
        "################################################################################\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # World parameters -- compute from (lng, lat) + depth bounds

    config_output += (
        "####################\n"
        "# World Parameters #\n"
        "####################\n"
        "# This section contains the parameters that define the boundaries and behaviour\n"
        "# of the world model.\n"
        "[world]\n"
        "\n"
        "# The x components of the world boundary in metres\n"
        "# min, max \n"
        "xRange = 0.0 {L}\n"
        "\n"
        "# The y components of the world boundary in metres\n"
        "# min, max\n"
        "yRange = 0.0 {L}\n"
        "\n"
        "# The z components of the world boundary in metres\n"
        "# min, max\n"
        "depthRange = 0.0 {maxdepth}\n"
        "\n"
        "\n".format(L=L, maxdepth=maxdepth))

    # ------------------------------------------------------------------------
    # World boundaries -- specify the properties of the GP layer boundaries

    offset_fnames = form_csv_fnames(layers,"Offset")
    ctrlmask_fnames = form_csv_fnames(layers,"CtrlMask")
    ctrlmin_fnames = form_csv_fnames(layers,"CtrlMin")
    ctrlmax_fnames = form_csv_fnames(layers,"CtrlMax")
    gpsig = 0.25*(layers.zmax - layers.zmin)
    gpcov = 0.25*(layers.zmax - layers.zmin)
    config_output += (
        "####################\n"
        "# World Boundaries #\n"
        "####################\n"
        "# This section contains the variables that relate to the boundary surfaces that\n"
        "# delineate different rock layers in the world model\n"
        "[boundaries]\n"
        "\n"
        "# Boundary offset CSV files. 1 file per boundary and space separated\n"
        "# (RS: these files contain bird's-eye maps of the GP mean function.)\n"
        "offsets = {offsetline}\n"
        "\n"
        "# Boundary time CSV files. If useTimes is enabled, for layer 2 onwards, these\n"
        "# files are interpreted as two way times rather than depths. First boundary is\n"
        "# always depths. Leave blank if useTimes is off.\n"
        "# (RS: relevant only if using seismic.)\n"
        "times = \n"
        "\n"
        "# Control point masks. These files determine the number of control points,\n"
        "# their shape, and which control points vary and are fixed.\n"
        "ctrlPointMasks = {ctrlmaskline}\n"
        "\n"
        "# Control point minimum. For each boundary, the minimun value a control point\n"
        "# can take from the offset surface\n"
        "# (RS: these provide hard reflection boundaries for the Metropolis proposal.)\n"
        "ctrlPointMins = {ctrlminline}\n"
        "\n"
        "# Control point maximum. For each boundary, the maximum value a control point\n"
        "# can take from the offset surface.\n"
        "# (RS: these provide hard reflection boundaries for the Metropolis proposal.)\n"
        "ctrlPointMaxs = {ctrlmaxline}\n"
        "\n"
        "# Standard deviations for for how the individual control points should vary.\n"
        "# These numbers (one per boundary) are the square roots of the diagonal of the\n"
        "# covariance matrix of a joint Gaussian over each layer's control points\n"
        "# (RS: amplitude of variation of the GP boundary for each layer.)\n"
        "uncoupledSDs = {gpsigline}\n"
        "\n"
        "# Standard deviations for for how the individual control points should co-vary.\n"
        "# These numbers (one per boundary) are the square roots of the off diagonal\n"
        "# terms of the covariance matrix of a joint Gaussian over each layer's control\n"
        "# points\n"
        "# (RS: this is like adding a totally correlated component to the GPs that\n"
        "#  causes them to vary together.  it's turned off here but think of these as\n"
        "#  systematic uncertainties on a constant offset to the GP mean function in\n"
        "#  each layer; see distrib::coupledGaussianBlock to see how it's implemented.)\n"
        "coupledSDs = {gpcovline}\n"
        "\n"
        "# boundary \"type\", either normal or warped. warped boundaries are intended to\n"
        "# be used as intrusion layers -- they are clipped differently and exaggerated\n"
        "# in the z direction to produce structures consistent with granite intrusions\n"
        "types = {typeline}\n"
        "\n"
        "# Flag to determine whether the offset files (exepct the top boundary) should\n"
        "# be interpreted as two-way times rather than distances. This allows the use of\n"
        "# processed seismic horizons without the need for depth conversion. Depth\n"
        "# conversions happens implicitly within the algorithm based on each layers'\n"
        "# p-wave velocity.\n"
        "useTimes = false\n"
        "\n"
        "\n".format(offsetline=" ".join(offset_fnames),
                    ctrlmaskline=" ".join(ctrlmask_fnames),
                    ctrlminline=" ".join(ctrlmin_fnames),
                    ctrlmaxline=" ".join(ctrlmax_fnames),
                    gpsigline=" ".join(["{:.1f}".format(si) for si in gpsig]),
                    gpcovline=" ".join(["{:.1f}".format(ci) for ci in gpcov]),
                    typeline=" ".join(layers.type)))

    # ------------------------------------------------------------------------
    # Rock properties

    rockmean_fnames = form_csv_fnames(layers,"RockMeans")
    rockmin_fnames = form_csv_fnames(layers,"RockMins")
    rockmax_fnames = form_csv_fnames(layers,"RockMaxs")
    rockcov_fnames = form_csv_fnames(layers,"RockCov")
    rockmask_fnames = form_csv_fnames(layers,"RockMask")
    config_output += (
        "#########\n"
        "# Rocks #\n"
        "#########\n"
        "# This section contains the variables that relate to the various rock\n"
        "# properties in each layer of the world model\n"
        "[rocks]\n"
        "\n"
        "# Rock property means. The rock propetries for each layer are a multivariate\n"
        "# (truncated) gaussian, and this is the mean vector for that gaussian.\n"
        "# One file per layer\n"
        "means = {rockmeanline}\n"
        "\n"
        "# Rock property minimum values. The PDF for rock properties is truncated beyond\n"
        "# this maximum value. The MCMC will also never explore outside this boundary.\n"
        "mins = {rockminline}\n"
        "\n"
        "# Rock property maximum values. The PDF for rock properties is truncated beyond\n"
        "# this maximum value. The MCMC will also never explore outside this boundary.\n"
        "maxs = {rockmaxline}\n"
        "\n"
        "# Rock property covariances. Each layer has rock properties which are jointly\n"
        "# Gaussian, and this propetry gives the covariance matrix for each.\n"
        "covariances = {rockcovline}\n"
        "\n"
        "# Rock property masks. Allows the user to turn on or off varying of rock\n"
        "# properties on a per-layer basis. If a rock property is masked out, it will be\n"
        "# fixed at it's mean value and will be ignored by the MCMC\n"
        "masks = {rockmaskline}\n"
        "\n"
        "\n".format(rockmeanline=" ".join(rockmean_fnames),
                    rockminline=" ".join(rockmin_fnames),
                    rockmaxline=" ".join(rockmax_fnames),
                    rockcovline=" ".join(rockcov_fnames),
                    rockmaskline=" ".join(rockmask_fnames)))

    # ------------------------------------------------------------------------
    # Simulation -- we're not using this but Obsidian will expect it

    config_output += (
        "##############\n"
        "# Simulation #\n"
        "##############\n"
        "# Rather than using real sensor inputs, it is possible to use simulated inputs\n"
        "# from a \"true\" model than has been input in this section.\n"
        "[simulation]\n"
        "\n"
        "#Control points: These files give the true z position of the control points\n"
        "#with respect to the offset files. Leave blank if using real data.\n"
        "ctrlPoints = \n"
        "\n"
        "# The true rock properties for each simulated layer. Leave blank if using real\n"
        "# data \n"
        "layerProperties = \n"
        "\n"
        "\n"
        "[initialisation]\n"
        "# optional initialisation, a file for each layer of the t=1 chain of each\n"
        "# stack. Ie stack1layer1.csv, stack1layer2.csv, stack2layer1.csv, stack2layer2.csv etc.\n"
        "ctrlPoints =\n"
        "layerProperties = \n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Gravity sensors

    config_output += (
        "###########\n"
        "# Gravity #\n"
        "###########\n"
        "# Section containing the variables for the gravity sensor and forward model\n"
        "[gravity] \n"
        "\n"
        "# Enable gravity as a sensor, if disabled, all the other gravity inputs are\n"
        "# ignored.\n"
        "enabled = true\n"
        "\n"
        "# Locations for the gravity sensors as a csv file\n"
        "sensorLocations= gravSensors.csv\n"
        "\n"
        "# The z component of the G field as taken by the sensors defined in locations.\n"
        "sensorReadings = gravReadings.csv\n"
        "\n"
        "# The resolution of the forward model voxelisation of the world\n"
        "# x y z\n"
        "gridResolution = 30 30 40\n"
        "\n"
        "# voxelisation supersample; horizontal antialiasing; depricated; do not use\n"
        "supersample = 0\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for gravity. Alpha parameter\n"
        "noiseAlpha = 1\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for gravity. Beta parameter.\n"
        "noiseBeta = 1\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Magnetic sensors
    # Ambient magnetic field from the IGRF model, which can be queried at
    #     https://www.ngdc.noaa.gov/geomag-web/#igrfwmm

    config_output += (
        "#############\n"
        "# Magnetism #\n"
        "#############\n"
        "# Section containing the variables for the magnetics sensor and forward model\n"
        "[magnetism]\n"
        "\n"
        "\n"
        "# Enable magnetics as a sensor, if disabled, all the other magnetic inputs are\n"
        "# ignored.\n"
        "enabled = true\n"
        "\n"
        "# Locations of the magnetics sensors as a csv file.\n"
        "sensorLocations = magSensors.csv\n"
        "\n"
        "# Readings of the total magnetic anomaly for each of the sensors define in\n"
        "# sensorLocations.\n"
        "sensorReadings = magReadings.csv\n"
        "\n"
        "# The resolution of the forward model voxelisation of the world\n"
        "# x y z\n"
        "gridResolution = 30 30 40\n"
        "\n"
        "# voxelisation supersample; horizontal antialiasing; depricated; do not use\n"
        "supersample = 0\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for gravity. Alpha parameter\n"
        "noiseAlpha = 1\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for gravity. Beta parameter\n"
        "noiseBeta = 1\n"
        "\n"
        "# The ambient magnetic field in nanotesla. Assumed constant over the world.\n"
        "# Components: east, north, down\n"
        "magneticField = {magline}\n"
        "\n"
        "\n".format(magline="{:.5g} {:.5g} {:.5g}".format(*H_IGRF)))

    # ------------------------------------------------------------------------
    # Magnetotelluric sensors -- disabled at present

    config_output += (
        "###################################\n"
        "# 1D Anisotropic Magnetotellurics #\n"
        "###################################\n"
        "# Section containing the variables for the 1D MT model and sensor input.\n"
        "[mtaniso]\n"
        "\n"
        "\n"
        "# Enable MT as a sensor, if disabled, all the other MT inputs are\n"
        "# ignored.\n"
        "enabled = false\n"
        "\n"
        "# File containing the locations *and Frequencies* of the magnetotelluric\n"
        "# sensors.\n"
        "sensorLocations = mtSensors.csv\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Alpha parameter\n"
        "noiseAlpha = 5\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Beta parameter\n"
        "noiseBeta = 0.5\n"
        "\n"
        "# if true, consider only the x component of resistivity and assume an isotropic\n"
        "# model. Resistivity Y, Resistivity Z and phase should be automatically masked.\n"
        "ignoreAniso = true \n"
        "\n"
        "# The sensor readings associated with each location and frequency defined\n"
        "# above. For details see the example CSV.\n"
        "sensorReadings = mtReadings.csv\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Seismic sensors -- disabled at present

    config_output += (
        "#########################\n"
        "# 1D Seismic Reflection #\n"
        "#########################\n"
        "# Section containing the variables for the 1D Seismic model and sensor data.\n"
        "[seismic1d]\n"
        "\n"
        "\n"
        "# Enable seismic as a sensor, if disabled, all the other seismic inputs are\n"
        "# ignored.\n"
        "enabled = false\n"
        "\n"
        "# List of locations for the seismic 2-way time sensors\n"
        "sensorLocations = seismic1dLocations.csv\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Alpha parameter\n"
        "noiseAlpha = 5\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Beta parameter\n"
        "noiseBeta = 0.5\n"
        "\n"
        "# Sensor reading containing the two way times for each layer and for each\n"
        "# sensor as specified in sensorLocations. For details see\n"
        "# the example file.\n"
        "sensorReadings =\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Contact points, i.e. drill holes -- disabled at present

    config_output += (
        "##################\n"
        "# Contact Points #\n"
        "##################\n"
        "# Section containing the variables for the drill hole contact point data\n"
        "[contactpoint]\n"
        "\n"
        "# Enable contact points as a sensor. If disabled, all the other contact point\n"
        "# inputs will be ignored.\n"
        "enabled = false\n"
        "\n"
        "#  The x,y,z locations and layers intersected by the drill holes\n"
        "sensorLocations = contactpointLocations.csv\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Alpha parameter\n"
        "noiseAlpha = 5\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Beta parameter\n"
        "noiseBeta = 0.5\n"
        "\n"
        "# Sensor reading containing the contact point depths for the layers and\n"
        "# locations defined in sensorLocations. For details see the example file.\n"
        "sensorReadings =\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Thermal sensors -- disabled at present

    config_output += (
        "###########\n"
        "# Thermal #\n"
        "###########\n"
        "# Section containing the variables for the thermal (temperature) sensor and\n"
        "# model\n"
        "[thermal]\n"
        "\n"
        "\n"
        "# Enable temperature as a sensor. If disabled, all other thermal inputs will be\n"
        "# ignored.\n"
        "enabled = false\n"
        "\n"
        "# CSV file containing a list of temperature sensor locations.\n"
        "sensorLocations = thermalSensors.csv\n"
        "\n"
        "# CSV file containing the temperatures corresponding to each sensor location\n"
        "# defined in the sensorLocations input file\n"
        "sensorReadings = thermalReadings.csv\n"
        "\n"
        "# The surface temperature boundary condition (in kelvin) to use in the forward\n"
        "# model simulations\n"
        "surfaceTemperature = 290.15\n"
        "\n"
        "# The lower boundary condition. This is in Kelvin, unless\n"
        "# lowerBoundaryIsHeatFlow is true, in which case it is interpreted as a heat\n"
        "# flow in milliwatts per m^2\n"
        "lowerBoundary = 0.1\n"
        "\n"
        "# interpert lower boundary condition as heat flow rather than temperature\n"
        "lowerBoundaryIsHeatFlow = true\n"
        "\n"
        "# voxelisation resolution of the thermal forward model\n"
        "# xres yres zres\n"
        "gridResolution = 13 13 25\n"
        "\n"
        "# voxelisation supersample; horizontal antialiasing; depricated; do not use\n"
        "supersample = 0\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Alpha parameter\n"
        "noiseAlpha = 10\n"
        "\n"
        "# noise inverse gamma: The parameters of the inverse gamma distribution used to\n"
        "# compute the likelihood function for MT. Beta parameter\n"
        "noiseBeta = 0.2\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # Field observation sensors -- disabled at present

    config_output += (
        "######################\n"
        "# Field Observations #\n"
        "######################\n"
        "# Section containing the variables for the geological field observation\n"
        "# sensor and model\n"
        "[fieldobs]\n"
        "\n"
        "\n"
        "# Enable field observations as a sensor. If disabled, all other field\n"
        "# observation inputs will be ignored.\n"
        "enabled = true\n"
        "\n"
        "# CSV file containing a list of field observation sensor locations.\n"
        "sensorLocations = fieldobsSensors.csv\n"
        "\n"
        "# CSV file containing the formations (layer boundary index) corresponding\n"
        "# to each sensor location defined in the sensorLocations input file\n"
        "sensorReadings = fieldobsReadings.csv\n"
        "\n"
        "# noise probability: The probability that a single visual identification of\n"
        "# a formation (categorical layer index) will be incorrect\n"
        "noiseProb = 0.01\n"
        "\n")

    # ------------------------------------------------------------------------
    # MCMC control settings and hyperparameters

    config_output += (
        "########\n"
        "# MCMC #\n"
        "########\n"
        "# Section controlling how the MCMC is run.\n"
        "[mcmc]\n"
        "\n"
        "#The number of chains in a sequence of temperatures. The first chain has\n"
        "#temperature=1, and then all subsequent chains have a temperature equal to\n"
        "#initialTempFactor times the previous temperature.\n"
        "chains = 20\n"
        "\n"
        "# The number of totally disconnected sets of chains. Total chains is the\n"
        "# product of the chains variable and the stacks variable. Different stacks are\n"
        "# compared to check convergence.\n"
        "stacks = 2\n"
        "\n"
        "# The walltime in seconds of the MCMC run. Obsidian will quit after this number\n"
        "# of seconds has elapsed\n"
        "wallTime = 86400\n"
        "\n"
        "# The number of states added to the highest temperature chain before it\n"
        "# initiates a swap with the next coldest chain. This will propagate all the way\n"
        "# down to the lowest temperature chain.\n"
        "swapInterval = 25\n"
        "\n"
        "# the factor that defines the geometric progression of chain temperatures for\n"
        "# each stack. t_n+1 = t_n * initialTempFactor, with t_1 = 1\n"
        "initialTempFactor = 2.0\n"
        "\n"
        "# The swap rate that the chains are trying to reach. Note that 0.24 is optimal\n"
        "# for an infinite dimensional problem, 0.5 is optimal for a 1 dimensional\n"
        "# problem.\n"
        "betaOptimalSwapRate = 0.24\n"
        "\n"
        "# The rate at which beta (1/temperature) adapts. This is a multiplicative\n"
        "# factor which scales the difference between the optimal swap rate and the real\n"
        "# swap rate.\n"
        "betaAdaptRate = 0.04\n"
        "\n"
        "# The minimum multiplicative factor by which beta can adapt.\n"
        "betaMinFactor = 0.8\n"
        "\n"
        "# The maximum multiplicative factor by which beta can adapt.\n"
        "betaMaxFactor = 1.25\n"
        "\n"
        "# The number of states appended to the chain between adaptions of beta.\n"
        "betaAdaptInterval = 500\n"
        "\n"
        "# The time scale over which the adaption length diminishes. After 10 times this\n"
        "# value, the adaption rate in approximately half. Also is equal to the number\n"
        "# of states used to calculate the adapt and swap rates in a sliding window.\n"
        "adaptionLength = 20000\n"
        "\n"
        "# Then number of states computed before they are written to disk. Worst case\n"
        "# would be to lose this many states upon a hard crash of the obsidian server.\n"
        "cacheLength = 1000\n"
        "\n"
        "\n")

    # ------------------------------------------------------------------------
    # MCMC proposal settings

    config_output += (
        "############\n"
        "# Proposal #\n"
        "############\n"
        "# Section controlling the proposal distribution, its width, and the adaption of\n"
        "# that width\n"
        "[proposal]\n"
        "\n"
        "# Proposal distribution (either Normal or CrankNicolson)\n"
        "distribution = CrankNicolson\n"
        "\n"
        "# Static step size parameter rho for Crank-Nicolson proposal\n"
        "ro = 0.5\n"
        "\n"
        "# The initial value of sigma (the proposal width) for the lowest temperature\n"
        "# chain\n"
        "initialSigma = 0.1\n"
        "\n"
        "# the factor that defines the geometric progression of chain sigmas for\n"
        "# each stack. s_n+1 = s_n * initialSigmaFactor, with s_1 = initialSigma\n"
        "initialSigmaFactor = 1.4\n"
        "\n"
        "# The maximum multiplicative factor by which sigma can adapt\n"
        "maxFactor = 1.25\n"
        "\n"
        "# The minimum multiplicative factor by which sigma can adapt\n"
        "minFactor = 0.8\n"
        "\n"
        "# The acceptance rate that the chains are trying to reach. Note that 0.24 is optimal\n"
        "# for an infinite dimensional problem, 0.5 is optimal for a 1 dimensional\n"
        "# problem.\n"
        "optimalAccept = 0.24\n"
        "\n"
        "# The rate at which sigma adapts. This is a multiplicative factor which scales\n"
        "# the difference between the optimal accept rate and the real accept rate.\n"
        "adaptRate = 0.2\n"
        "\n"
        "# The number of states appended to the chain between adaptions of sigma\n"
        "adaptInterval = 250\n"
        "\n"
        "\n")

    # Finally, write the string to disk!
    with open("config.obsidian", 'w') as configfile:
        configfile.write(config_output)

    # ========================================================================
    #               Part 2:  The Layer-By-Layer Config Files
    # ========================================================================

    # Layer geometry:  In the Curtin formation boundary problem, we have no
    # really strong constraints on the depths of particular layer boundaries
    # across the modeled volume, since we have no contact points that I know
    # about, only a 2-D slice of seismic, and that seismic slice shows that
    # many layer boundaries slant in at an angle.  So we're going to use
    # very weak priors on the layer depths, and only assume a general order
    # that will allow us to pinch layers out as needed.

    # ------------------------------------------------------------------------
    # Control point mean offset files

    for li, lfn in zip(layers.index, offset_fnames):
        # Format the layer offsets as a CSV file
        lname, ltype, zmin, zmax, lnx, lny = layers.loc[li]
        offsets = pd.DataFrame(0.5*(zmax + zmin)*np.ones((lnx, lny)))
        offsets[offsets < 0] = 0.0
        offsets[offsets > maxdepth] = maxdepth
        offsetstr = offsets.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# Offset: This is a mean heightmap giving the prior mean of a particular\n"
                "# boundary. Control points deform this offset. This file can be high resolution\n"
                "# (much greater than the number of control points for instance), because linear\n"
                "# interpolation is used to create voxelisations.\n"
                "# The top left of this file is\n"
                "# the northern-most, western-most depth value. Positive number indicate\n"
                "# increasing depth. Units are in metres, except if BoundariesAreTimes flag is\n"
                "# true (and it is not the first layer), in which case they are in seconds and\n"
                "# represent two-way times. The distances are computed through the P-wave\n"
                "# velocity parameter.\n"
                "\n"
                "{offsetstr}".format(offsetstr=offsetstr))

    # ------------------------------------------------------------------------
    # Control mask files

    for lnx, lny, lfn in zip(layers.nx, layers.ny, ctrlmask_fnames):
        # Format the layer offsets as a CSV file
        offmasks = pd.DataFrame(np.ones((lnx, lny), dtype=int))
        offmaskstr = offmasks.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# Mask file: Ones or zeros in the shape of the control points for a particular\n"
                "# layer. Ones indicate that control point is allowed to vary, while a zero\n"
                "# indicates it is fixed as a constant and ignored by the inference. This file\n"
                "# is how the number of control points in each layer are specified. Add more\n"
                "# numbers to this mask to increase the number of control points. Control points\n"
                "# are always distributed evenly over a grid the size of the world dimensions\n"
                "# input in the obsidian file.\n"
                "# The top left of this file is the northern-most, western-most control point.\n"
                "\n"
                "{offmaskstr}\n".format(offmaskstr=offmaskstr))

    # ------------------------------------------------------------------------
    # Control point minimum offset values

    for li, lfn in zip(layers.index, ctrlmin_fnames):
        # Format the layer offsets as a CSV file
        lname, ltype, zmin, zmax, lnx, lny = layers.loc[li]
        offmins = pd.DataFrame(-0.5*(zmax - zmin)*np.ones((lnx, lny)))
        offminstr = offmins.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# CtrlMin: gives the minimum value a control point can move away from the given\n"
                "# offset. Note this is not the same as the min of the boundary's movement,\n"
                "# which is a more complex function of all the control points in the boundary.\n"
                "# The top left of this file is the northern-most, western-most control point.\n"
                "# All numbers are metres, and positive numbers indicate increasing depth.\n"
                "\n"
                "{offminstr}\n".format(offminstr=offminstr))

    # ------------------------------------------------------------------------
    # Control point maximum offset values

    for li, lfn in zip(layers.index, ctrlmax_fnames):
        # Format the layer offsets as a CSV file
        lname, ltype, zmin, zmax, lnx, lny = layers.loc[li]
        offmaxs = pd.DataFrame(0.5*(zmax - zmin)*np.ones((lnx, lny)))
        offmaxstr = offmaxs.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# CtrlMax: gives the maximum value a control point can move away from the given\n"
                "# offset. Note this is not the same as the max of the boundary's movement,\n"
                "# which is a more complex function of all the control points in the boundary.\n"
                "# The top left of this file is the northern-most, western-most control point.\n"
                "# All numbers are metres, and positive numbers indicate increasing depth.\n"
                "\n"
                "{offmaxstr}\n".format(offmaxstr=offmaxstr))

    # Rock properties:  We would've ideally liked a hierarchical prior where
    # the properties of the rock are conditioned on a rock type drawn from a
    # categorical distribution, but it looks like the Obsidian guys didn't
    # quite get this far.  In the Curtin formation boundary problem, we've
    # decided to put Gaussian priors on the rock properties using the sample
    # covariances of petrophysical measurements Hugo gave us; ideally we'd
    # consider these data too but let's leave that alone for now.

    # ------------------------------------------------------------------------
    # Rock property mean values -- from petrophysical measurements
    
    for ln, lfn in zip(layers.name, rockmean_fnames):
        # Format the layer offsets as a CSV file
        rockmeans = rockpriormu[ln]
        rockmeanstr = rockmeans.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# RockMeans: The mean values of the rock properties for a particular layer.\n"
                "# In Order: \n"
                "# Density in kg/m^3\n"
                "# LogSusceptibility  in Log(unitless) \n"
                "# ThermalConductivity in Watts/(metres Kelvin)\n"
                "# ThermalProductivity in Watts/metre^3\n"
                "# LogResistivityX in Log(ohm metres)\n"
                "# LogResistivityY in Log(ohm metres)\n"
                "# LogResistivityZ in Log(ohm metres)\n"
                "# ResistivityPhase in degrees between 0 and 90\n"
                "# PWaveVelocity in metres/second \n"
                "#\n"
                "# All logs are assumed to be base 10.\n"
                "\n"
                "{rockmeanstr}\n".format(rockmeanstr=rockmeanstr))

    # ------------------------------------------------------------------------
    # Rock property covariances -- from petrophysical measurements

    for ln, lfn in zip(layers.name, rockcov_fnames):
        # Format the layer offsets as a CSV file
        rockcovs = rockpriorcov[ln]
        rockcovstr = rockcovs.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# RockCov: This file represents the covariance matrix for the multivariate\n"
                "# Gaussian prior over all the rock properties. The order of rock properties is\n"
                "# the same in this file as in the mean file.\n"
                "# In Order: \n"
                "# Density in kg/m^3\n"
                "# LogSusceptibility  in Log(unitless) \n"
                "# ThermalConductivity in Watts/(metres Kelvin)\n"
                "# ThermalProductivity in Watts/metre^3\n"
                "# LogResistivityX in Log(ohm metres)\n"
                "# LogResistivityY in Log(ohm metres)\n"
                "# LogResistivityZ in Log(ohm metres)\n"
                "# ResistivityPhase in degrees between 0 and 90\n"
                "# PWaveVelocity in metres/second \n"
                "#\n"
                "# All logs are assumed to be base 10.\n"
                "\n"
                "{rockcovstr}\n".format(rockcovstr=rockcovstr))

    # ------------------------------------------------------------------------
    # Rock property minimums -- be conservative

    for ln, lfn in zip(layers.name, rockmin_fnames):
        # Format the layer offsets as a CSV file
        rockmins = rockpriormu[ln] - 10.0*np.sqrt(np.diag(rockpriorcov[ln]))
        rockminstr = rockmins.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# RockMins: The minimum value the rock properties can take in the inversion.\n"
                "# In Order: \n"
                "# Density in kg/m^3\n"
                "# LogSusceptibility  in Log(unitless) \n"
                "# ThermalConductivity in Watts/(metres Kelvin)\n"
                "# ThermalProductivity in Watts/metre^3\n"
                "# LogResistivityX in Log(ohm metres)\n"
                "# LogResistivityY in Log(ohm metres)\n"
                "# LogResistivityZ in Log(ohm metres)\n"
                "# ResistivityPhase in degrees between 0 and 90\n"
                "# PWaveVelocity in metres/second \n"
                "#\n"
                "# All logs are assumed to be base 10.\n"
                "\n"
                "{rockminstr}\n".format(rockminstr=rockminstr))

    # ------------------------------------------------------------------------
    # Rock property maximums -- be conservative

    for ln, lfn in zip(layers.name, rockmax_fnames):
        # Format the layer offsets as a CSV file
        rockmaxs = rockpriormu[ln] + 10.0*np.sqrt(np.diag(rockpriorcov[ln]))
        rockmaxstr = rockmaxs.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# RockMaxs: The maximum value the rock properties can take in the inversion.\n"
                "# In Order: \n"
                "# Density in kg/m^3\n"
                "# LogSusceptibility  in Log(unitless) \n"
                "# ThermalConductivity in Watts/(metres Kelvin)\n"
                "# ThermalProductivity in Watts/metre^3\n"
                "# LogResistivityX in Log(ohm metres)\n"
                "# LogResistivityY in Log(ohm metres)\n"
                "# LogResistivityZ in Log(ohm metres)\n"
                "# ResistivityPhase in degrees between 0 and 90\n"
                "# PWaveVelocity in metres/second \n"
                "#\n"
                "# All logs are assumed to be base 10.\n"
                "\n"
                "{rockmaxstr}\n".format(rockmaxstr=rockmaxstr))

    # ------------------------------------------------------------------------
    # Rock property control masks -- only let density & susceptibility vary

    for ln, lfn in zip(layers.name, rockmask_fnames):
        # Format the layer offsets as a CSV file
        rockmask = pd.DataFrame(np.zeros(rockpriormu[ln].shape[0], dtype=int),
                                index=rockpriormu[ln].index)
        rockmask.loc['Density_kgm3',:] = rockmask.loc['log_MagSusc',:] = 1
        rockmaskstr = rockmask.to_csv(index=False, header=False)
        with open(lfn, 'w') as config_file:
            config_file.write(
                "# Mask file: Ones or zeros as a column vector for the rock properties of a\n"
                "# particular layer. Ones indicate that property is allowed to vary,\n"
                "# while a zero indicates it is fixed as a constant and ignored by the\n"
                "# inference.\n"
                "# In Order:\n"
                "# Density,\n"
                "# LogSusceptibility,\n"
                "# ThermalConductivity,\n"
                "# ThermalProductivity,\n"
                "# LogResistivityX,\n"
                "# LogResistivityY,\n"
                "# LogResistivityZ,\n"
                "# ResistivityPhase,\n"
                "# PWaveVelocity\n"
                "\n"
                "{rockmaskstr}\n".format(rockmaskstr=rockmaskstr))

    # ========================================================================
    #                    Part 3:  The Sensor Data Files
    # ========================================================================

    # ------------------------------------------------------------------------
    # Gravity sensor:  convert to milligal by dividing by 10 (good catch Hugo)

    rhdr = ("# GravReadings: This gives a list of the actual readings from all the sensors\n"
            "# specified in the gravity locations file. The measurements can be considered\n"
            "# free air anomaly or Bouguer anomaly in milligals, depending on whether the\n"
            "# model has explicitly added an air layer at the top of the simulation.\n\n")
    shdr = ("# GravSensors: This file gives a list of location for gravity measurements.\n"
            "# The columns are x,y,z in metres. these are not offsets from the world\n"
            "# boundaries so if your world doesn't start from zero be careful that the\n"
            "# sensors are actually placed inside the world. An error should be flagged if\n"
            "# if they don't. Positive Z values are going into the ground.\n\n")
    write_sensor_data(grav_data, lng, lat, L,
                      stag='grav', rhdr=rhdr, shdr=shdr, zval=-0.5)

    # ------------------------------------------------------------------------
    # Magnetic sensor

    rhdr = ("# MagReadings: This gives a list of the readings from the sensors in the\n"
            "# MagSensors file. They are assumed to be Total magnetic anomaly readings in\n"
            "# nano-Tesla.\n\n")
    shdr = ("# MagSensors: This file gives a list of location for magnetic measurements.\n"
            "# The columns are x,y,z in metres. these are not offsets from the world\n"
            "# boundaries so if your world doesn't start from zero be careful that the\n"
            "# sensors are actually placed inside the world. An error should be flagged if\n"
            "# if they don't. Positive Z values are going into the ground.\n\n")
    write_sensor_data(mag_data, lng, lat, L, axthin=4,
                      stag='mag', rhdr=rhdr, shdr=shdr, zval=-0.5)

    # ------------------------------------------------------------------------
    # Field observation sensor

    rhdr = ("# FieldObsReadings: This gives a list of the readings from the sensors\n"
            "# in the FieldObsSensors file. They are assumed to be visual observations\n"
            "# of formations at the surface (categorical, layer boundary index values).\n"
            "# A value of -1 indicates observation of a layer not listed in the prior.\n")
    shdr = ("# FieldObsSensors: This file gives a list of location for geological\n"
            "# field observations. The columns are x,y in metres; these are not offsets\n"
            "# from the world boundaries, so if your world doesn't start from zero,\n"
            "# be careful that the sensors are actually placed inside the world.\n"
            "# An error should be flagged if they don't.\n\n")
    write_sensor_data(field_data, lng, lat, L,
                      stag='fieldobs', rhdr=rhdr, shdr=shdr, zval=None)
