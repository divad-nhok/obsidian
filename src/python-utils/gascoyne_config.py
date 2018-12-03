#!/usr/bin/env python

"""
RS 2018/02/26:  Config Settings for Gascoyne Province Dataset A

In order to abstract the particulars of the configuration away from the
actual Obsidian configuration file writer, I'm going to have that config
writer evaluate a lot of variables from a separate file, which itself
will be python code.  Perhaps the easiest way to do that is with a
pickle actually, since the config files are all small and are basically
standard Python objects or pandas dataframes.

If we end up wanting to commit any of the final data to the repo, then
we'll probably want to make a separate sub-path of obsidian/examples
and dump them in there.
"""

import copy
import numpy as np
import pandas as pd
from config_writer import write_config

# ============================================================================
#                USER-CONFIGURABLE OPTIONS WITHIN THIS BLOCK
# ============================================================================

# First we need to define a configuration, which is going to consist of
# latitude and longitude boundaries of the area, as well as a list of
# (name, min, max) for different layers.  We have to be careful about the
# order in which these layers are specified because the configuration
# fixes a strict order in which those layers can appear in the model,
# even if some layers are absent at certain locations.  Example below.

_datadir = "../../../data/"
config_petrofn = _datadir + "Petrophysics_Gascoyne.csv"
ground_truthfn = _datadir + "Formation_data_Gascoyne.csv"
gravdata_fn = _datadir + "gravity_400m_Gascoyne.txt"
magdata_fn = _datadir + "mag_TMI_gascoyne.txt"

# Column format for sensor files
sensor_colnames = ['id', 'val', 'lat', 'lng']

# Ambient magnetic field from the IGRF model, which can be queried at
#     https://www.ngdc.noaa.gov/geomag-web/#igrfwmm
H_IGRF = np.array([1.419e+2, 2.8739e+4, -4.62667e+4])

config_layers = pd.DataFrame(
        # layer name, layer type, (min, max) depth in m, (nx, ny) control pts
        [('Halfway Gneiss', 'normal', 0.0, 1.0e+4, 5, 5),
         ('Durlacher Supersuite',  'normal', 0.0, 1.5e+4, 5, 5),
         ('XXX',  'normal', 0.0, 2.5e+4, 5, 5),
         ('YYY',       'normal', 0.0, 2.5e+4, 5, 5),],
        columns=['name','type','zmin','zmax','nx','ny'])

config_params = { 'lng': 116.10, 'lat': -24.85, 'L': 2.0e+4,
                  'maxdepth': 1.0e+4, 'layers': config_layers,
                  'H_IGRF': H_IGRF, }

# ============================================================================
# REST OF CODE:  calculates rock priors, reads in data, & writes configuration
# ============================================================================

def compute_rock_priors(rockfname):
    """
    Compute distributions of rock properties for geological layers from
    petrophysical data provided by Hugo.  The prior has 9 components:
        Density in kg/m^3
        log(Magnetic susceptibility)
        Thermal conductivity
        Thermal productivity
        log(Resistivity x-component)
        log(Resistivity y-component)
        log(Resistivity z-component)
        P-wave velocity

    :param rockfname:  name of CSV file with rock measurements from Hugo
    :returns:  two dicts, rockprior_mu and rockprior_cov, containing the
        sample mean and covariance of the rock property measurements for
        each rock layer (indexed by the rock layer's name)
    """

    # Read in the petrophysics rock data Hugo gave us.
    rockdata = pd.read_csv(rockfname)
    rockpriormu = { }
    rockpriorcov = { }
    # Default janky prior just in case we don't have enough physical samples.
    # The first two terms are derived from the petrophysics data aggregated
    # over all layers.  The other terms are made up because they have to be
    # filled, but if we want to use other sensors we need better priors.
    rockprops = ['Density_kgm3', 'log_MagSusc',
                 'ThermConduct', 'ThermProduct',
                 'log_Resist_x', 'log_Resist_y', 'log_Resist_z',
                 'Resist_Phase', 'PWaveVelocity']
    rockpriormu_def = pd.Series(
            [ 2.7e+3, 1.5, 2.0, 2e-6, 0.0, 0.0, 0.0, 0.0, 4000 ],
            index=rockprops)
    rockpriorcov_def = pd.DataFrame(np.diag(
            [ 5.0e+4, 0.5, 0.25, 1e-12, 1.0, 1.0, 1.0, 1.0, 2.5e+5 ]),
            index=rockprops, columns=rockprops)
    
    # Go through all the formations we have data for.
    forms = np.unique(rockdata.FormationName)
    for i, f in enumerate(forms):
        df = rockdata[rockdata.FormationName==f]
        rockpriormu[f] = copy.deepcopy(rockpriormu_def)
        rockpriorcov[f] = copy.deepcopy(rockpriorcov_def)
        # If we have enough data points (I say 5, somewhat arbitrarily),
        # construct an empirical Gaussian prior from Hugo's data.
        if len(df) >= 5:
            # Grab last two columns -- susceptibility and density
            dc = np.zeros(shape=(df.shape[0], 2))
            # Convert magnetic susceptibility to log scale
            dc[:,0] = 1000.0*df.loc[:,'Density_g_cm-3']
            # Convert density to kg/m^3
            dc[:,1] = np.log10(df.loc[:,'Magnetic_susceptibility'])
            # Store sample covariance in hash
            rockpriormu[f].iloc[:2] = np.mean(dc, axis=0)
            rockpriorcov[f].iloc[:2,:2] = np.cov(dc.T)
    return rockpriormu, rockpriorcov

def display_ground_truth(lng, lat, L):
    """
    Displays geological ground-truth labels in a given area.  Put here
    until I find a better home for it.
    :param lng:  longitude of modeled area centre in decimal degrees E
    :param lat:  latitude of modeled area centre in decimal degrees E
    :param L:  length of side of (square) modeled area in metres
    """
    # Read in Hugo's data and extract the desired volume
    formdata = pd.read_csv(ground_truthfn)
    labels = pd.DataFrame(
            formdata.iloc[:,2:5].values, columns=['lat', 'lng', 'val'])
    x, y, v = select_volume(labels, lng, lat, L)
    # Plot the ground truth
    for f in np.unique(v):
        idx = (v == f)
        plt.plot(x[idx], y[idx], ls='None', marker='o', ms=3, label=f)
    plt.legend()
    plt.title("${:.1f} \\times {:.1f}$ km$^2$ area centered on "
              "lng = ${:.3f}$, lat = ${:.3f}$"
              .format(L/1e+3, L/1e+3, lng, lat))
    plt.xlabel('Eastings (m)')
    plt.ylabel('Northings (m)')
    plt.show()

def main():
    """
    The main config-writer routine
    """

    # Generate rock priors
    rockpriormu, rockpriorcov = compute_rock_priors(config_petrofn)

    # Read in grav data and convert to the standard format.
    # For Hugo's data, divide by 10 to convert from um/s^2 to mgal.
    gravdata = pd.read_csv(gravdata_fn, names=sensor_colnames,
                           dtype=float, skiprows=1)
    gravdata.val /= 10.0

    # Read in mag data and convert to the standard format.
    magdata = pd.read_csv(magdata_fn, names=sensor_colnames,
                          dtype=float, skiprows=1)

    # Fill remaining fields in config_params

    config_params.update({ 'grav_data': gravdata,
                           'mag_data': magdata,
                           'rockpriormu': rockpriormu,
                           'rockpriorcov': rockpriorcov, })

    write_config(**config_params)

if __name__ == "__main__":
    main()
