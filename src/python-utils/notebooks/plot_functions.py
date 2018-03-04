import matplotlib.pyplot as plt
import numpy as np
import os

def make_convergence_plot(
    converged_all,
    fig_width = 10,
    fig_height = 10,
    fontsize = 20,
    xstr = 'Iteration',
    ystr = 'Gelman-Rubin Rhat',
    title_str = 'Gelman-Rubin R for different windows of iterations for param {}',
    save_str = 'rhat_full_param{}.png',
    plot_type = 1,
    param_range = 10,
    save_dir = ''
):
    for param_idx in range(param_range):
        y = converged_all[param_idx,:]
        y = y[~np.isnan(y)]
        x = range(len(y))

        if plot_type == 1:
            fig = plt.figure(figsize = (fig_width, fig_height))
            ax = plt.gca()
            idx = slice(1000, len(y))
            plt.plot(x[idx], y[idx])

        elif plot_type == 2:
            fig, axes = plt.subplots(
                2, 2, 
                #sharex=True, 
                #sharey=True, 
                figsize = (fig_width, fig_height)
            )
            fig.add_subplot(111, frameon=False)
            plt.tick_params(
                labelcolor='none',
                top='off',
                bottom='off',
                left='off',
                right='off'
            )
            plt.grid(False)
            axes[0][0].plot(x, y)
            idx = slice(0, 100)
            axes[0][1].plot(x[idx], y[idx])
            idx = slice(100, 1000)
            axes[1][0].plot(x[idx], y[idx])
            idx = slice(1000, len(x))
            axes[1][1].plot(x[idx], y[idx])

        plt.xlabel(
            xstr,
            fontsize = fontsize
        )
        plt.ylabel(
            ystr,
            fontsize = fontsize
        )
        plt.title(
            title_str.format(param_idx),
            fontsize = fontsize
        )
        out_path = os.path.join(
            save_dir, 
            save_str.format(param_idx)
        )
        plt.savefig(
            out_path
        )
        plt.clf()