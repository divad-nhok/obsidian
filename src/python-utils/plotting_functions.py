import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import visvis as vv
from pandas.compat import range, lrange, lmap, zip
import matplotlib
#import seaborn as sns

import geovis_notebook_version
from diagplots2 import fieldobs_lookup, display_ground_truth

fig_height = 5
fig_width = 5
fontsize = 20
tick_fontsize = 20

def make_chainstat_table(
    chain_list, fpath_out = None, plot_title = None
):
    index = [
        'Mean', 
        'Median', 
        'Standard deviation',
    ]
    columns = ['Chain {}'.format(idx) for idx in range(len(chain_list))]
    func_list = [
        np.mean,
        np.median,
        lambda x: np.std(x),
    ]
    table_data = [
        [func(chain) for func in func_list]
        for chain in chain_list
    ]
    table = pd.DataFrame(table_data,columns=index,index=columns).T
    if fpath_out:
        table.to_csv(fpath_out)

def make_fieldobs_table(
    actual, predicted, fpath_out = None, plot_title = None
):
    index = [
        'True positive', 
        'False positive', 
        'True negative',
        'False negative'
    ]
    columns = ['']
    func_list = [
        lambda x, y: ((x == 1) & (x == y)).sum(),
        lambda x, y: ((x == 1) & (x != y)).sum(),
        lambda x, y: ((x == 0) & (x == y)).sum(),
        lambda x, y: ((x == 0) & (x != y)).sum(),
    ]
    table_data = [
        [func(actual, predicted) for func in func_list]
    ]
    table = pd.DataFrame(table_data,columns=index,index=columns).T
    if fpath_out:
        table.to_csv(fpath_out)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def make_chains(
    samples, stack_length_list, n_burn, n_thin,
    make_uniform_length = True
):
    thinned_lengths = [(l - n_burn)/n_thin for l in stack_length_list]
    thinned_lengths2 = [np.floor(l) for l in thinned_lengths]
    smallest_length = np.min(thinned_lengths2)
    cum_lengths = np.cumsum(thinned_lengths2)
    chains = []
    start = 0
    for end in cum_lengths:
        data = samples[start:end]
        if make_uniform_length:
            data = data[:smallest_length - 1]
        chains.append(data)
        start = end
    if make_uniform_length:
        chains = np.stack(chains, axis = 1).T
    return(chains)

def run_diagnostic_plots(
    fpath_in_list,
    dir_out_list,
    layer_list,
    param_idx_list,
    plot_function_list,
    extension_list,
    plot_function_names,
    layer_title_list = None,
    param_title_list = None,
    n_burn = 0,
    n_thin = 0,
    stack_length_list = []
):
    for experiment_idx, (fpath_in, dir_out) in enumerate(zip(fpath_in_list, dir_out_list)):
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if os.path.exists(fpath_in):
            samples_i = np.load(fpath_in)
            for layer_idx, layer in enumerate(layer_list):
                layer_params = samples_i[layer]
                layer_title = layer
                if layer_title_list:
                    layer_title = layer_title_list[layer_idx]
                for param_idx in param_idx_list:
                    param = layer_params[:, param_idx]
                    if np.any(stack_length_list):
                        param = make_chains(param, stack_length_list, n_burn, n_thin, make_uniform_length = False)
                    param_title = 'param {}'.format(param_title_list[param_idx])
                    if param_title_list:
                        param_title = param_title_list[param_idx].lower()
                    fname_fig_param = '{}-param{}'.format(layer,param_idx)
                    for plot_func, plot_func_name, extension in zip(plot_function_list, plot_function_names, extension_list):
                        fname_fig = '{}-{}.{}'.format(plot_func_name, fname_fig_param, extension)
                        fpath_out = os.path.join(dir_out, fname_fig)
                        plot_title = '{} {}'.format(layer_title, param_title)
                        #print('param.shape: {}'.format(param.shape))
                        data = param[n_burn:]
                        #print('data.shape: {}'.format(data.shape))
                        plot_func(
                            param[n_burn:], 
                            fpath_out = fpath_out, 
                            plot_title = plot_title
                        )

def plot_acf(
    data,
    img_format = 'png',
    fpath_out = None,
    plot_title = None,
    thin_amount = None,
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    x_label = 'Lag'
    y_label = 'Autocorrelation'
    f = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    data = [reject_outliers(data_i) for data_i in data]
    for data_i in data:
        _ = autocorrelation_plot(data_i, ax = ax)
    plt.setp(ax.lines, linewidth = 0.5)

    #line = ax.lines[5]
    #line.set_color('black')

    # x axis
    start = 0
    end = np.max([len(data_i) for data_i in data])
    middle = (end - start) / 2
    new_xt = [start, middle, end]
    ax.set_xticks(new_xt)
    ax.set_xticklabels([int(num) for num in new_xt])
    ax.set_xlim([start, end])
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.xlabel(x_label, fontsize = fontsize)

    # y axis
    start = np.min([np.min(line.get_ydata()) for line in ax.lines])
    end = np.max([np.max(line.get_ydata()) for line in ax.lines])
    middle = (end + start) / 2
    new_yt = [start, middle, end]
    new_yt = np.around(new_yt, decimals = 2)
    new_yt = np.append(new_yt,0)
    new_yt = np.sort(new_yt)
    ax.set_yticks(new_yt)
    ax.set_yticklabels(new_yt)
    _ = plt.yticks(fontsize = tick_fontsize)
    _ = plt.ylabel(y_label, fontsize = fontsize)

    plt.tight_layout()

    ax.grid(False)
    if plot_title: plt.title(plot_title, fontsize = fontsize)
    plt.tight_layout()
    plt.savefig(fpath_out)
    plt.clf()

def plot_trace(
    data,
    img_format = 'png',
    fpath_out = None,
    plot_title = None,
    thin_amount = None,
    lw = 0.5
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    x_label = 'MCMC iteration'
    if thin_amount:
        x_label = 'MCMC iteration (thinned $\\times {}$)'.format(thin_amount)
    y_label = 'Parameter value'
    f = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    data = [reject_outliers(data_i) for data_i in data]
    for data_i in data:
        _ = ax.plot(data_i, linewidth = lw,) #color='black')
    #mean = np.mean(data)

    # x axis
    start = 0
    end = np.max([len(data_i) for data_i in data])
    middle = (end - start) / 2
    new_xt = [start, middle, end]
    ax.set_xticks(new_xt)
    ax.set_xticklabels([int(num) for num in new_xt])
    ax.set_xlim([start, end])
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.xlabel(x_label, fontsize = fontsize)

    # y axis
    start = np.min([np.min(data_i) for data_i in data])
    end = np.max([np.max(data_i) for data_i in data])
    middle = (end + start) / 2
    new_yt = [start, middle, end]
    new_yt = np.around(new_yt, decimals = 2)
    ax.set_yticks(new_yt)
    ax.set_yticklabels(new_yt)

    #ax.set_ylim([start, end])
    plt.tight_layout()
    _ = plt.ylabel(y_label, fontsize = fontsize)
    #print(plot_title)
    #print('new_xt: {}\nnew_yt: {}'.format(new_xt, new_yt))
    if plot_title: plt.title(plot_title, fontsize = fontsize)
    plt.tight_layout()
    plt.savefig(fpath_out)
    plt.clf()

def plot_hist_diagnostic(
    data,
    fpath_out,
    plot_title = None,
    fig_width = fig_width,
    fig_height = fig_height,
    bins = 10,
    width = 0.1,
    tick_fontsize = tick_fontsize,
    xlab = 'Parameter bin value',
    mean_labels = True,
    mean_line = True,
    xlabels = []
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    data = [reject_outliers(data_i) for data_i in data]
    colors = ['red', 'green', 'blue', 'orange']
    colors = [matplotlib.colors.to_rgba(color, alpha=0.5) for color in colors]
    colors = [(1,0,0,0.5), (0,1,0,0.5), (0,0,1,0.5), (1,215/255,0,0.5)]
    idx = 0
    for data_i in data:
        weights = np.ones_like(data_i)/float(len(data_i))
        h = plt.hist(
            data_i, 
            bins = bins, 
            weights = weights,
            fc = colors[idx],
            ec='black',
        )
        idx += 1
    
    """
    plt.clf()
    labels = pd.rolling_mean(h[1], 2)[1:]
    counts = h[0]

    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    #width = 0.5
    mini = np.min([np.min(data_i) for data_i in data])
    maxi = np.max([np.max(data_i) for data_i in data])
    width = np.round(np.abs((maxi - mini)/(bins + 2 )), 2)
    plt.bar(
        labels, counts, align='center',
        edgecolor='black', 
        width = width, 
        fill = False
    )

    if mean_line:
        for data_i in data:
            ax.axvline(data_i.mean(), color = 'r')

    if np.any(xlabels):
        label_strs = ['{:.1f}'.format(l) for l in xlabels]
        _ = ax.set_xticks(xlabels)
        _ = ax.set_xticklabels(label_strs)

    mean_labels = False
    if mean_labels:
        closest_mean_idx = np.argmin([np.abs(l - data.mean())  for l in labels])
        selected_labels = [labels[0], labels[closest_mean_idx], labels[-1]]
        label_strs = ['{:.1f}'.format(l) for l in selected_labels]
        _ = ax.set_xticks(selected_labels)
        _ = ax.set_xticklabels(label_strs)

    yt = ax.get_yticks()
    ytl = ['{:.0f}\%'.format(yt_i * 100) for yt_i in yt]
    ax.set_yticklabels(ytl)
    """

    _ = plt.xlabel(xlab, fontsize = fontsize)
    _ = plt.ylabel('Probability', fontsize = fontsize)
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.yticks(fontsize = tick_fontsize)
    if plot_title:
        _ = plt.title(plot_title, fontsize = fontsize)
    plt.tight_layout()
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
        
        # scatter plot of actual observations
        plot_scatter_fieldobs(
            fieldLabels,
            'fieldobs', 'actual',
            dir_out,
        )

        # Now show samples
        i = len(samples['fieldReadings'])
        readings = samples['fieldReadings'][i-1]
        fieldLabels.val = fieldobs_lookup(readings)
        predicted = readings
        plot_scatter_fieldobs(
            fieldLabels,
            'fieldobs', 'predicted',
            dir_out,
        )

        # histogram of residuals
        resid = actual - predicted
        xlabels = np.sort(np.unique(resid))
        plot_hist(
            resid,
            'fieldobs', 'resid',
            dir_out,
            xlab = 'Residual',
            width = 0.5,
            mean_labels = False,
            mean_line = False,
            xlabels=xlabels,
        )
        fpath_out = os.path.join(dir_out, 'fieldobs-prediction-eval.csv')
        make_fieldobs_table(actual, predicted, fpath_out)
        return(sensors, actual)


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
        resid_list = []
        for sensor_name in sensor_name_list:
            sensor_key = '{}Sensors'.format(sensor_name)
            readings_key = '{}Readings'.format(sensor_name)
            if readings_key not in samples.keys():
                print('readings key {} not in dict_data'.format(readings_key))
                continue
            sensors = dict_data[sensor_key]
            readings = dict_data[readings_key]
            #print(sensors.shape)
            #print(readings.shape)
            x, y, z = sensors.T
            actual = readings - readings.mean()
            #print(actual.shape)
            N, N2 = samples[readings_key].shape
            #print(N, N2)
            chain = samples[readings_key][N/2:]
            #print(chain.shape)
            fwd_model = chain.mean(axis=0) - chain.mean()
            #print(fwd_model.shape)
            resid = actual - fwd_model
            resid_list.append(resid)
            abs_resid = np.abs(resid)
            if sensor_name == 'mag': 
                width = 100
                hist_xlab = 'Data - Model (nT)'
            if sensor_name == 'grav': 
                width = 0.5
                hist_xlab = 'Data - Model (mgal)'
            plot_data_list = [
                #(plot_contour, (x, y, actual), 'posterior', dict(data2 = fwd_model, cmap = 'viridis')),
                #(plot_contour, (x, y, actual), 'resid', dict(data2 = resid, cmap = 'magma')),
                #(plot_hist, fwd_model, 'posterior', dict(width = width, xlab = hist_xlab)),
                #(plot_hist, resid, 'resid', dict(width = width, xlab = hist_xlab)),
            ]
            for plot_func, plot_data, plot_key, plot_args in plot_data_list:
                if type(plot_data) is tuple:
                    plot_func(*plot_data, sensor_name, plot_key, dir_out, **plot_args)
                else:
                    plot_func(plot_data, sensor_name, plot_key, dir_out, **plot_args)
        make_resid_table(resid_list, dir_out_list[0])


def make_resid_table(resid_list, dir_out):
    col_names = ['Gravity', 'Magnetism']
    index = [
        'Mean', 'Median', 
        '2 x Standard deviation',
        '2std % of total range',
        '5% percentile', '95% percentile'
    ]
    func_list = [
        np.mean,
        np.median,
        lambda x: np.std(x) * 2,
        lambda x: (np.std(x) * 2) / np.ptp(x) * 100,
        lambda x: np.percentile(x, 5),
        lambda x: np.percentile(x, 95),
    ]
    table_data = [
        [func(resid) for func in func_list]
        for resid in resid_list
    ]
    table = pd.DataFrame(table_data,columns=index,index=col_names).T
    fname = 'resid-table.csv'
    fout = os.path.join(dir_out, fname)
    table.to_csv(fout)
    print(table)

def plot_contour(
    x, y, data,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = fig_width,
    fig_height = fig_height,
    xlab = 'Eastings (km)',
    ylab = 'Northings (km)',
    data2 = [],
    cmap = 'viridis'
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass

    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    _ = plt.tricontourf(x, y, data, alpha=0.5, cmap=cmap);
    _ = plt.colorbar()
    if np.any(data2):
        _ = plt.tricontour(x, y, data2, colors='k', label='Fwd Model')
    _ = plt.xlabel(xlab, fontsize = fontsize)
    _ = plt.ylabel(ylab, fontsize = fontsize)

    xt = ax.get_xticks()
    xtl = ['{:.0f}'.format(xt_i / 1000) for xt_i in xt]
    ax.set_xticklabels(xtl, fontsize=tick_fontsize)

    yt = ax.get_yticks()
    ytl = ['{:.0f}'.format(yt_i / 1000) for yt_i in yt]
    ax.set_yticklabels(ytl, fontsize=tick_fontsize)

    plt.tight_layout()
    fname = '{}-{}-contour.eps'.format(sensor_prefix, plot_key)
    fpath_fig = os.path.join(dir_output, fname)
    plt.savefig(fpath_fig)
    plt.clf()

def plot_hist(
    data,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = fig_width,
    fig_height = fig_height,
    bins = 10,
    width = 1.0,
    tick_fontsize = tick_fontsize,
    xlab = 'Parameter bin value',
    mean_labels = True,
    mean_line = True,
    xlabels = []
):
    try:
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
    except:
        pass
    weights = np.ones_like(data)/float(len(data))
    h = plt.hist(
        data, 
        bins = bins, 
        weights = weights,
    )
    plt.clf()
    labels = pd.rolling_mean(h[1], 2)[1:]
    counts = h[0]

    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    #width = 0.5
    plt.bar(
        labels, counts, align='center',
        edgecolor='black', 
        width = width, 
        fill = False
    )

    if mean_line:
        ax.axvline(data.mean(), color = 'r')

    if np.any(xlabels):
        label_strs = ['{:.1f}'.format(l) for l in xlabels]
        _ = ax.set_xticks(xlabels)
        _ = ax.set_xticklabels(label_strs)

    if mean_labels:
        closest_mean_idx = np.argmin([np.abs(l - data.mean())  for l in labels])
        selected_labels = [labels[0], labels[closest_mean_idx], labels[-1]]
        label_strs = ['{:.1f}'.format(l) for l in selected_labels]
        _ = ax.set_xticks(selected_labels)
        _ = ax.set_xticklabels(label_strs)

    yt = ax.get_yticks()
    ytl = ['{:.0f}\%'.format(yt_i * 100) for yt_i in yt]
    ax.set_yticklabels(ytl)

    _ = plt.xlabel(xlab, fontsize = fontsize)
    _ = plt.ylabel('Probability', fontsize = fontsize)
    _ = plt.xticks(fontsize = tick_fontsize)
    _ = plt.yticks(fontsize = tick_fontsize)
    plt.tight_layout()

    fname = '{}-{}-hist.eps'.format(sensor_prefix, plot_key)
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

def plot_scatter_fieldobs(
    data,
    sensor_prefix, plot_key,
    dir_output,
    fig_width = 5,
    fig_height = 5,
):
    x, y, v = data.x, data.y, data.val

    fig = plt.figure(figsize = (fig_width, fig_height))
    ax = plt.gca()
    markers = ['o', '*', 's', 'v']
    colors = ['r', 'k', 'b', 'g']
    marker_idx = 0
    for unq in np.unique(v.values):
        print(unq)
        idx = (v == unq)
        _ = plt.scatter(
            x[idx], y[idx], 
            marker=markers[marker_idx],
            label=unq,
            #facecolors=colors[marker_idx],
            facecolors='none',
            edgecolors=colors[marker_idx],
        )
        marker_idx += 1
    _ = plt.xlabel('Eastings (km)')
    _ = plt.ylabel('Northings (km)')
    yt = ax.get_yticks()[1:-1]
    ytl = ['{}'.format(yt_i/1000) for yt_i in yt]
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)
    xt = ax.get_xticks()[1:-1]
    xtl = ['{}'.format(xt_i/1000) for xt_i in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl)

    plt.legend(loc='upper left')
    plt.tight_layout()

    fname = '{}-{}-scatter.eps'.format(sensor_prefix, plot_key)
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

def autocorrelation_plot(series, ax=None, extra_lines=False, **kwds):
    """Autocorrelation plot for time series.

    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method

    Returns:
    -----------
    ax: Matplotlib axis object
    """
    import matplotlib.pyplot as plt
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    if extra_lines:
        ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey')
        ax.axhline(y=z95 / np.sqrt(n), color='grey')
        ax.axhline(y=0.0, color='black')
        ax.axhline(y=-z95 / np.sqrt(n), color='grey')
        ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey')
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    return ax

