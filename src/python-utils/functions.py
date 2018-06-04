import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyproj

msg_str0 = 'fpath: {}'
msg_str1 = '  Latitude min: {}\n  Latitude max: {}'
msg_str2 = '  Grid code min: {}\n  Grid code max: {}'
msg_str3 = '  Data shape: {}'
msg_str4 = '  X shape: {}\n  Y shape: {}'
msg_str5 = '  Data columns: {}'

def get_vars(dict_data_set, sub_key):
    dir_data = dict_data_set['dir_data']
    fname_data = dict_data_set[sub_key]['fname']
    fpath_data = os.path.join(dir_data, fname_data)
    key_lat = dict_data_set[sub_key]['key_lat']
    key_lon = dict_data_set[sub_key]['key_lon']
    key_y = dict_data_set[sub_key]['key_y']
    return(fpath_data, key_lon, key_lat, key_y)

def get_data(
    fpath,
    ftype = 'txt'
):
    msg_str = msg_str0.format(fpath)
    print(msg_str)
    if (ftype == 'csv') or (ftype == 'txt'):
        data = pd.read_csv(fpath, header=0)
    elif (ftype == 'xlsx'):
        data = pd.read_excel(fpath, header=0)
    elif (ftype == 'asc'):
        data = pd.read_csv(fpath, sep = ' ')
    msg_str = msg_str3.format(data.shape)
    print(msg_str)
    msg_str = msg_str5.format(data.columns)
    print(msg_str)
    return(data)

def get_data_that_is_in_square_around_centre(
    data, 
    centres,
    keys,
    lengths
):
    """
    Subtract the centre coordinates of the region
    and then see whether the "centred coordinates"
    are within the region specified by length.
    """
    msg_str = 'centres: {}\nkeys: {}\nlengths: {}'.format(centres, keys, lengths)
    print(msg_str)
    print('data.shape before: {}'.format(data.shape))
    for length, key, centre in zip(lengths, keys, centres):
        cond = np.abs(np.abs(data[key]) - np.abs(centre)) < length
        #cond = np.abs(data[key] - centre) < length
        data = data[cond]
    data.reset_index(inplace = True)
    print('data.shape after: {}'.format(data.shape))
    return(data)

def run_gp(
    data, key_lat, key_lon, key_y,
    ARD = False
):
    print('Running GP')
    X = np.array([data[key_lat], data[key_lon]]).T
    Y = np.array([data[key_y]]).T
    kernel = GPy.kern.Matern32(2, ARD = ARD)
    msg_str = msg_str4.format(X.shape, Y.shape)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=True)
    fig = plt.figure(figsize = (10, 10))
    print(model)
    f = model.plot()
    return(X, Y, model)
