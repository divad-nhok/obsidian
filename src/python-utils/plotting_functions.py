import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.plotting import autocorrelation_plot
import re
import visvis as vv

import geovis_notebook_version
from diagplots2 import fieldobs_lookup, display_ground_truth

fig_height = 10
fig_width = 10
fontsize = 20
tick_fontsize = 10

def run_diagnostic_plots(
    fpath_in_list,
    dir_out_list,
    param_key_list,
    param_idx_list,
    plot_function_list,
    plot_function_names
):
    for experiment_idx, (fpath_in, dir_out) in enumerate(zip(fpath_in_list, dir_out_list)):
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if os.path.exists(fpath_in):
            samples_i = np.load(fpath_in)
            for key in param_key_list:
                layer_params = samples_i[key]
                for param_idx in param_idx_list:
                    param = layer_params[:, param_idx]
                    fname_fig_param = '{}-param{}.eps'.format(key,param_idx)
                    for plot_func, plot_func_name in zip(plot_function_list, plot_function_names):
                        fname_fig = '{}-{}'.format(plot_func_name, fname_fig_param)
                        fpath_out = os.path.join(dir_out, fname_fig)
                        plot_title = '{} param {}'.format(key, param_idx)
                        plot_func(
                            param, 
                            fpath_out = fpath_out, 
                            plot_title = plot_title
                        )

def plot_acf(
    data,
    img_format = 'png',
    fpath_out = None,
    plot_title = None,
    thinned_amount = '??'
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    x_label = 'MCMC iteration (thinned $\\times {}$)'.format(thin_amount)
    y_label = 'Autocorrelation'
    f = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    #plt.plot(acf_list)
    _ = autocorrelation_plot(data, ax = ax)
    #yticks = np.arange(-0.2, 1.1, 0.1)
    #ax.set_yticks(yticks)
    #ax.axhline(0, color = 'g')
    #ax.axhline(0.1, color = 'k')
    #ax.axhline(-0.1, color = 'k')
    _ = plt.yticks(
        list(plt.yticks()[0]) + [-0.1, 0, 0.1],
        fontsize = tick_fontsize
    )
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.xlabel(x_label, fontsize = fontsize)
    _ = plt.ylabel(y_label, fontsize = fontsize)
    #print(plot_title)
    if plot_title: plt.title(plot_title, fontsize = fontsize)
    plt.savefig(fpath_out)
    plt.clf()

def plot_trace(
    data,
    img_format = 'png',
    fpath_out = None,
    plot_title = None
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    x_label = 'MCMC iteration (thinned $\\times 100$)'
    y_label = 'Parameter value'
    f = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    _ = ax.plot(data)
    #yticks = np.arange(-0.2, 1.1, 0.1)
    #ax.set_yticks(yticks)
    #ax.axhline(0, color = 'g')
    #ax.axhline(0.1, color = 'k')
    #ax.axhline(-0.1, color = 'k')
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.xlabel(x_label, fontsize = fontsize)
    _ = plt.ylabel(y_label, fontsize = fontsize)
    #print(plot_title)
    if plot_title: plt.title(plot_title, fontsize = fontsize)
    plt.savefig(fpath_out)
    plt.clf()

def plot_fieldobs(
    fpath_samples_list,
    dir_out_list,
    fpath_csv_list,
    data_names_list,
):
    sensor_name = 'fieldobs'
    for fpath_samples, dir_out, fpath_csv_list_i in zip(fpath_samples_list, dir_out_list, fpath_csv_list):
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if os.path.exists(fpath_samples): 
            samples = np.load(fpath_samples)
            print('samples path {} exists'.format(fpath_samples))
            #print(samples.keys())
        else: 
            print('samples path {} does not exist'.format(fpath_samples))
            continue
        name_list = [['x', 'y'], ['val']]
        dict_data = {
            key: pd.read_csv(path, names=names, comment='#')
            for key, path, names in 
            zip(data_names_list, fpath_csv_list_i, name_list)
            if os.path.exists(path)
        }
        sensor_key = '{}Sensors'.format(sensor_name)
        readings_key = '{}Readings'.format(sensor_name)
        if 'fieldReadings' not in samples.keys():
            print('fieldReadings key not in samples.keys()')
            continue
        sensors = dict_data[sensor_key]
        readings = dict_data[readings_key]
        fieldLabels = sensors.assign(val=fieldobs_lookup(readings.val))
        actual = readings.val
        
        fig = plt.figure(figsize=(10,10))
        display_ground_truth(fieldLabels, show=False)
        plt.title('Field Observations')
        out_path = os.path.join(dir_out, 'boundary_data.png')
        plt.savefig(out_path)

        # Now show samples
        fig = plt.figure(figsize=(10,10))
        i = len(samples['fieldReadings'])
        readings = samples['fieldReadings'][i-1]
        fieldLabels.val = fieldobs_lookup(readings)
        predicted = readings
        display_ground_truth(fieldLabels, show=False)
        plt.title('Forward-Modeled Field Observations, '
                  'Sample {} from MCMC Chain'.format(i))
        plt.savefig(os.path.join(dir_out,'boundary_fwdmodel_endchain.png'))
        plt.close()

        # residuals
        fig = plt.figure(figsize=(10,10))
        resid = actual - predicted
        plt.hist(resid)
        plt.title('field obs residual')
        plt.savefig(os.path.join(dir_out,'field-obs-resid.png'))
        plt.close()

def plot_sensor_output(
    fpath_samples_list,
    dir_out_list,
    fpath_csv_list,
    data_names_list,
    sensor_name_list = ['grav', 'mag']
):
    for fpath_samples, dir_out, fpath_csv_list_i in zip(fpath_samples_list, dir_out_list, fpath_csv_list):
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if os.path.exists(fpath_samples): 
            samples = np.load(fpath_samples)
            print('samples path {} exists'.format(fpath_samples))
            #print(samples.keys())
        else: 
            print('samples path {} does not exist'.format(fpath_samples))
            continue
        dict_data = {
            key: np.loadtxt(path, delimiter=',')
            for key, path in zip(data_names_list, fpath_csv_list_i)
            if os.path.exists(path)
        }
        print(dict_data.keys())
        for sensor_name in sensor_name_list:
            sensor_key = '{}Sensors'.format(sensor_name)
            readings_key = '{}Readings'.format(sensor_name)
            if readings_key not in samples.keys():
                print('readings key {} not in dict_data'.format(readings_key))
                continue
            sensors = dict_data[sensor_key]
            readings = dict_data[readings_key]
            print(sensors.shape)
            print(readings.shape)
            x, y, z = sensors.T
            actual = readings - readings.mean()
            print(actual.shape)
            N, N2 = samples[readings_key].shape
            print(N, N2)
            chain = samples[readings_key][N/2:]
            print(chain.shape)
            fwd_model = chain.mean(axis=0) - chain.mean()
            print(fwd_model.shape)
            resid = actual - fwd_model
            abs_resid = np.abs(resid)
            # contour map of actual readings
            plot_data_list = [
                (plot_contour, (x, y, actual), 'data'),
                (plot_contour, (x, y, fwd_model), 'posterior'),
                (plot_contour, (x, y, resid), 'resid'),
                (plot_contour, (x, y, abs_resid), 'abs-val-resid'),
                (plot_hist, actual, 'data'),
                (plot_hist, fwd_model, 'posterior'),
                (plot_hist, resid, 'resid'),
                (plot_scatter, (actual, resid), 'actual-vs-resid'),
                (plot_scatter, (fwd_model, resid), 'fwdmodel-vs-resid'),
                (plot_scatter, (actual, fwd_model), 'actual-vs-fwdmodel'),
            ]
            for plot_func, plot_data, plot_key in plot_data_list:
                if type(plot_data) is tuple:
                    plot_func(*plot_data, sensor_name, plot_key, dir_out)
                else:
                    plot_func(plot_data, sensor_name, plot_key, dir_out)

def plot_contour(
    x, y, data,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = 10,
    fig_height = 10,
    xlab = 'Eastings (m)',
    ylab = 'Northings (m)',
):
    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    _ = plt.tricontourf(x, y, data, alpha=0.5);
    _ = plt.colorbar()
    _ = plt.xlabel(xlab)
    _ = plt.ylabel(ylab)
    fname = '{}-{}-contour.png'.format(sensor_prefix, plot_key)
    fpath_fig = os.path.join(dir_output, fname)
    plt.savefig(fpath_fig)
    plt.clf()

def plot_hist(
    data,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = 10,
    fig_height = 10,
    bins = 20
):
    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    weights = np.ones_like(data)/float(len(data))
    _ = plt.hist(data, bins = bins, weights = weights)
    _ = plt.xlabel('Parameter value bin')
    _ = plt.ylabel('Probability')
    fname = '{}-{}-hist.png'.format(sensor_prefix, plot_key)
    fpath_fig = os.path.join(dir_output, fname)
    plt.savefig(fpath_fig)
    plt.clf()

def plot_scatter(
    x, y,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = 10,
    fig_height = 10,
):
    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    _ = plt.scatter(x, y)
    re_result = re.search('(.*)-vs-(.*)', plot_key)
    _ = plt.xlabel(re_result.group(1))
    _ = plt.ylabel(re_result.group(2))
    fname = '{}-{}-scatter.png'.format(sensor_prefix, plot_key)
    fpath_fig = os.path.join(dir_output, fname)
    plt.savefig(fpath_fig)
    plt.clf()

def get_fname_list(dir_parent, search_str):
    fname_list = [
        os.path.join(dir_parent, fname)
        for fname in os.listdir(dir_parent) 
        if os.path.isfile(os.path.join(dir_parent, fname))
        and search_str in fname
    ]
    return(fname_list)

def run_geovis_app(
    layer_voxels,
    fname_save = None
):
    app = vv.use()
    vv.figure(1)
    vv.xlabel('Eastings (units)')
    vv.ylabel('Northings (units)')
    vv.zlabel('Depth (units)')
    a = vv.gca()
    a.camera.fov = 200
    a.daspect = 1, 1, -1
    t = vv.volshow(layer_voxels, cm=vv.CM_JET)
    vv.ColormapEditor(a)
    if fname_save:
        vv.screenshot(
            fname_save,
            sf = 2,
            bg = 'w'
        )
        vv.cla()
        vv.clf()
        vv.closeAll()

def plot_layer_posteriors(
    fname_list,
    dir_out = '',
    img_save_format = 'png',
    fname_save_template_str = 'mean-posterior-layer-{:02}.{}'
):
    print(fname_list)
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)
    for idx, fn in enumerate(fname_list):
        if idx == 0:
            view = geovis_notebook_version.MasonView(fn)
        else:
            view.add_samples(geovis_notebook_version.MasonView(fn))
    n_layers = len(view.layers)
    for layer_idx in range(n_layers):
        print(layer_idx)
        layer_voxels = view.meanlayer(layer_idx)
        fname_img = fname_save_template_str.format(layer_idx, img_save_format)
        fname_save = os.path.join(dir_out, fname_img)
        print(fname_save)
        run_geovis_app(layer_voxels, fname_save)
        plt.clf()
    return(view)
