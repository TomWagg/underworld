import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import gaussian_filter


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7*fs,
          'legend.title_fontsize': 0.8*fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def plot_side_on_density(xs, zs, labels, xlim=20, zlim=12, n_bins=200, sigma=1.0, apply_smoothing=True,
                         contours=[1, 10, 100, 1000],
                         fig=None, ax=None, show=True):
    """Plot side-on density distribution of objects given x and z coordinates.

    Parameters
    ----------
    xs : list of array-like
        List of x-coordinate arrays for different populations to plot.
    zs : list of array-like
        List of z-coordinate arrays for different populations to plot.
    labels : list of str
        List of labels corresponding to each population.
    xlim : float, optional
        Limit for x-axis in kpc. Default is 20.
    zlim : float, optional
        Limit for z-axis in kpc. Default is 12.
    n_bins : int, optional
        Number of bins for the histogram. Default is 200.
    sigma : float, optional
        Standard deviation for Gaussian smoothing. Default is 1.0.
    apply_smoothing : bool, optional
        Whether to apply Gaussian smoothing to the histogram. Default is True.
    contours : list of float, optional
        Contour levels to plot on top of the density map. Default is [1, 10, 100, 1000]. To disable,
        set to None or an empty list.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, a new figure is created. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, new axes are created. Default is None.
    show : bool, optional
        Whether to display the plot immediately. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    upper_lim = 0
    to_plot = []

    for x, z, extent in zip([np.abs(xs[0].to(u.kpc).value), -np.abs(xs[1].to(u.kpc).value)],
                            [zs[0].to(u.kpc).value, zs[1].to(u.kpc).value],
                            [[0, xlim, -zlim, zlim], [-xlim, 0, -zlim, zlim]]):
        mask = (np.abs(z) < zlim) & (np.abs(x) < xlim)
        x = x[mask]
        z = z[mask]

        range_ex = [[extent[0], extent[1]], [extent[2], extent[3]]]

        hist, x_edges, y_edges = np.histogram2d(x, z, range=range_ex, bins=n_bins)

        if apply_smoothing:
            smoothed_hist = gaussian_filter(hist, sigma=sigma)
        else:
            smoothed_hist = hist

        max_count = smoothed_hist.max()
        max_count_logged = 10**np.floor(np.log10(max_count))
        max_count_rounded = int(np.ceil(max_count / max_count_logged) * max_count_logged)

        upper_lim = max(upper_lim, max_count_rounded)

        to_plot.append((smoothed_hist, extent))

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    for plotting, label in zip(to_plot, labels):
        smoothed_hist, extent = plotting

        im = ax.imshow(
            smoothed_hist.T,
            origin='lower',
            extent=extent,
            cmap='magma',
            norm=mpl.colors.LogNorm(vmin=1, vmax=upper_lim)
        )

        if contours is not None and len(contours) > 0:
            cont = ax.contour(
                smoothed_hist.T,
                levels=contours,
                colors='white',
                linewidths=0.5,
                origin='lower',
                extent=extent,
                alpha=0.5,
            )
            ax.clabel(cont, inline=True, fontsize=0.4*fs, fmt='%1.0f')

        ax.annotate(
            label,
            xy=(0.02 if extent[0] == 0 else 0.98, 0.05),
            ha='left' if extent[0] == 0 else 'right',
            xycoords='axes fraction',
            color='white',
            fontsize=0.8*fs,
            weight='bold'
        )

    fig.colorbar(im, label='Number of objects', ax=ax)

    ax.set(
        xlim=(-xlim, xlim),
        ylim=(-zlim, zlim),
        xlabel='x [kpc]',
        ylabel='z [kpc]',
    )
    ax.set_facecolor('black')

    if show:
        plt.show()

    return fig, ax