import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


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
                         contours=[1, 10, 100, 1000], norm="log",
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

    for x, z, extent in zip([np.abs(xs[0]), -np.abs(xs[1])],
                            [zs[0], zs[1]],
                            [[0, xlim, -zlim, zlim], [-xlim, 0, -zlim, zlim]]):
        if hasattr(x, 'unit'):
            x = x.to(u.kpc).value
        if hasattr(z, 'unit'):
            z = z.to(u.kpc).value
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

        if norm == 'log':
            kwargs = {
                'norm': mpl.colors.LogNorm(vmin=1, vmax=upper_lim)
            }
        else:
            kwargs = {
                'vmin': 1,
                'vmax': upper_lim / 2
            }

        im = ax.imshow(
            smoothed_hist.T,
            origin='lower',
            extent=extent,
            cmap='magma',
            **kwargs
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
        xlabel=r'Galactocentric $x$ [kpc]',
        ylabel=r'Galactocentric $z$ [kpc]',
    )
    ax.set_facecolor('black')

    if show:
        plt.show()

    return fig, ax


def nice_transparent_hist(ax, data, bins, label, colour, density, lw=2, alpha=0.4, cumulative=False):
    ax.hist(data, bins=bins, color=colour, lw=lw, histtype='step', density=density, label=label, cumulative=cumulative)
    ax.hist(data, bins=bins, color=colour, alpha=alpha, density=density, cumulative=cumulative)


def compare_table_quantity(pops, quantity, kstar, bins, xlabel, ylabel, density=True, table_name="final_bpp",
                           lw=2, cumulative=False,
                           fig=None, ax=None, show=True, **settings):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if not isinstance(kstar, (list, np.ndarray)):
        kstar = [kstar]

    for pop in pops:
        data = np.concatenate((
            getattr(pop, table_name)[f"{quantity}_1"][pop.final_bpp["kstar_1"].isin(kstar)],
            getattr(pop, table_name)[f"{quantity}_2"][pop.final_bpp["kstar_2"].isin(kstar)],
        ))

        nice_transparent_hist(
            ax=ax, data=data, bins=bins,
            label=f"{pop.label}\nN={len(data)}", colour=pop.colour,
            density=density, lw=lw, cumulative=cumulative
        )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        **settings
    )

    ax.legend()

    if show:
        plt.show()

    return fig, ax


def plot_mass_histogram(pops, data, bins, co_type, xlabel, ylabel, density=True, lw=2, alpha=0.4,
                        fig=None, ax=None, show=True, labels=None, colours=None, legend_title=None,
                        cumulative=False,
                        **settings):
    labels = [f"{pop.label}\nN={len(data[pop.label]['mass'][co_type])}"
              for pop in pops] if labels is None else labels
    colours = [pop.colour for pop in pops] if colours is None else colours

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    for pop, colour in zip(pops, colours):
        nice_transparent_hist(
            ax=ax, data=data[pop.label]['mass'][co_type], bins=bins,
            label=None, colour=colour,
            density=density, lw=lw, alpha=alpha, cumulative=cumulative
        )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        **settings
    )

    # construct a legend with rectangles matching the histogram colors (lines have same colour, fill has
    # the same alpha)
    handles = [mpl.patches.Patch(edgecolor=colour, facecolor=mpl.colors.to_rgba(colour, alpha=alpha),
                                 label=label, lw=lw) for colour, label in zip(colours, labels)]
    leg = ax.legend(handles=handles, title=legend_title, frameon=False)
    leg.get_title().set_multialignment('center')

    if show:
        plt.show()

    return fig, ax


def exponential(x, a, b):
    return a * np.exp(-b * x)


def double_exponential(x, a, w, b1, b2):
    """Double exponential function for fitting."""
    return a * (w * np.exp(-b1 * x) + (1 - w) * np.exp(-b2 * x))


def exp_plus_sech2(x, a, w, b1, b2):
    """Exponential plus sech^2 function for fitting."""
    return a * (w * np.exp(-b1 * x) + (1 - w) * 1 / (np.cosh(b2 * x))**2)

def double_exp_plus_sech2(x, a, w1, w2, b1, b2, b3):
    """Double exponential plus sech^2 function for fitting."""
    return a * (w1 * np.exp(-b1 * x) + w2 * np.exp(-b2 * x) + (1 - w1 - w2) * 1 / (np.cosh(b3 * x))**2)


def estimate_scale_height(z, bins=np.linspace(0, 3, 201),
                          plot=False, fig=None, ax=None, show=True,
                          label="", colour="black", model="exp_plus_sech2",
                          scale_height_loc=None, zorder=None,
                          R=None, Rlims=(7.5, 8.5), verbose=False,
                          **kwargs):
    """Estimate the scale height of a distribution given z-positions."""
    z = np.abs(z)

    if R is not None:
        R = R.to(u.kpc).value if hasattr(R, 'unit') else R
        mask = (R >= Rlims[0]) & (R < Rlims[1])
        if verbose:
            print(len(z), "objects before Rlims")
        z = z[mask]
        if verbose:
            print(len(z), "objects in Rlims")

    # remove units for calculation if they exist
    if hasattr(z, 'unit'):
        z = z.to(u.kpc).value

    hist, bin_edges = np.histogram(z, bins=bins, range=kwargs["xlim"], density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # hist = hist/hist.max()

    scale_height = bin_centres[hist < hist.max() / np.e][0]

    smooth_hist = gaussian_filter(hist, sigma=0.5)

    z_range = np.linspace(0, bin_edges.max(), 50000)

    if model == "exp":
        p0 = [smooth_hist.max(), 1 / scale_height]
        popt, pcov = curve_fit(exponential, bin_centres, smooth_hist, p0=p0)

        fit_pdf = exponential(z_range, *popt)
        fit_pdf /= fit_pdf.max()
        scale_height = z_range[fit_pdf <= (1 / np.e)][0]
        scale_height_err = np.sqrt(np.diag(pcov))[1]

    elif model == "double_exp":
        p0 = [smooth_hist.max(), 0.5, 1 / scale_height, 1 / scale_height]
        popt, pcov = curve_fit(double_exponential, bin_centres, smooth_hist, p0=p0,
                               bounds=([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf]))
        scale_height = [1 / popt[2], 1 / popt[3]]

        fit_pdf = double_exponential(z_range, *popt)
        fit_pdf /= fit_pdf.max()
        scale_height = z_range[fit_pdf <= (1 / np.e)][0]
        scale_height_err = max(np.sqrt(np.diag(pcov))[2:4])

    elif model == "exp_plus_sech2":
        p0 = [smooth_hist.max(), 0.5, 2, 2]
        popt, pcov = curve_fit(exp_plus_sech2, bin_centres, smooth_hist, p0=p0,
                               bounds=([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf]))
        scale_height = [1 / popt[2], 1 / popt[3]]
        # print(np.sqrt(np.diag(pcov)) * 1000)

        fit_pdf = exp_plus_sech2(z_range, *popt)
        fit_pdf /= fit_pdf.max()
        scale_height = z_range[fit_pdf <= (1 / np.e)][0]
        scale_height_err = max(np.sqrt(np.diag(pcov))[2:4])
    elif model == "double_exp_plus_sech2":
        p0 = [smooth_hist.max(), 0.3, 0.3, 2, 2, 2]
        popt, pcov = curve_fit(double_exp_plus_sech2, bin_centres, smooth_hist, p0=p0,
                               bounds=([0, 0, 0, 0, 0, 0], [np.inf, 1, 1, np.inf, np.inf, np.inf]))
        scale_height = [1 / popt[3], 1 / popt[4], 1 / popt[5]]

        fit_pdf = double_exp_plus_sech2(z_range, *popt)
        fit_pdf /= fit_pdf.max()
        scale_height = z_range[fit_pdf <= (1 / np.e)][0]
        scale_height_err = max(np.sqrt(np.diag(pcov))[3:6])
    else:
        raise ValueError(f"Model {model} not recognized. Choose from 'exp', 'exp_plus_sech2', or 'double_exp_plus_sech2'.")

    print(f"Fitting {label} using model: {model}")
    print(f"  Optimal parameters: {popt}")
    print(f"  Parameter errors: {np.sqrt(np.diag(pcov)) * 1000}")
    print(f" Estimated scale height: {scale_height * 1000:.2f} pc ± {scale_height_err * 1000:.2f} pc")

    if plot:

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(bin_centres, hist, label=label, color=colour, lw=2, zorder=zorder)

        plot_func_dict = {
            "exp": exponential,
            "double_exp": double_exponential,
            "exp_plus_sech2": exp_plus_sech2,
            "double_exp_plus_sech2": double_exp_plus_sech2
        }
        plot_func = plot_func_dict[model]
        ax.plot(bin_centres, plot_func(bin_centres, *popt), color=colour, ls='--', alpha=0.5, zorder=zorder)

        if scale_height_loc is not None:
            loc = smooth_hist.max() if scale_height_loc is None else scale_height_loc
            ax.axvline(scale_height, color=colour, ls='dotted', alpha=0.5, zorder=zorder)
            ax.annotate(f"{scale_height * 1000:.0f} pc", xy=(scale_height, loc), fontsize=0.7*fs,
                        rotation=0, color=colour, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour), zorder=1000)

        ax.set(
            xlabel="Distance from the Galactic plane, |z| (kpc)",
            ylabel=kwargs.pop('ylabel', r'Density'),
            **kwargs
        )
        ax.set_xlabel(ax.get_xlabel(), fontsize=0.8*fs)
        ax.legend(fontsize=0.7*fs)

        if show:
            plt.show()

        return scale_height, scale_height_err, fig, ax
    else:
        return scale_height, scale_height_err, None, None


def estimate_scale_height_cdf(
        z, plot=False, fig=None, ax=None, show=True,
        label="", colour="black", scale_height_loc=None,
        R=None, Rlims=(7.5, 8.5), verbose=False, weight_bins=False, **kwargs):
    """Estimate the scale height of a distribution given z-positions using the cumulative distribution function (CDF).
    This method does not assume a model, but instead just finds the z-value at which the CDF reaches 1 - 1/e ~ 0.63, which corresponds to the scale height for an exponential distribution."""
    z = np.abs(z)
    if R is not None:
        R = R.to(u.kpc).value if hasattr(R, 'unit') else R
        mask = (R >= Rlims[0]) & (R < Rlims[1])
        if verbose:
            print(len(z), "objects before Rlims")
        z = z[mask]
        if verbose:
            print(len(z), "objects in Rlims")

    if hasattr(z, 'unit'):
        z = z.to(u.kpc).value

    # calculate empirical CDF without any binning
    sorted_z = np.sort(z)
    cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)

    scale_height = sorted_z[cdf >= (1 - 1 / np.e)][0]


    if plot:
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # interpolate cdf for plotting
        subsample_z = np.geomspace(sorted_z.min(), sorted_z.max(), 1000)
        cdf_interp = np.interp(subsample_z, sorted_z, cdf)

        ax.plot(subsample_z, cdf_interp, label=label, color=colour, lw=2)

        if scale_height_loc is not None:
            loc = 0.5 if scale_height_loc is None else scale_height_loc
            ax.axvline(scale_height, color=colour, ls='--', alpha=0.5)
            ax.annotate(f"{scale_height * 1000:.0f} pc", xy=(scale_height, loc), fontsize=0.7*fs,
                        rotation=0, color=colour, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour))

        ax.set(
            xlabel="Distance from the Galactic plane, |z| (kpc)",
            ylabel=kwargs.pop('ylabel', r'$F(z)$'),
            xscale=kwargs.pop('xscale', 'log'),
            **kwargs
        )
        ax.set_xlabel(ax.get_xlabel(), fontsize=0.8*fs)
        ax.legend(fontsize=0.7*fs)

        if show:
            plt.show()

        return scale_height, None, fig, ax
    else:
        return scale_height, None, None, None


def absolute_galactocentric_height(pops, kinematics, co_type="CO", fig=None, axes=None, show=True):
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    scales = ['linear', 'log']
    bin_list = [np.linspace(0, 10, 501), np.geomspace(8e-4, 4e4, 120)]
    labels = ['Inner 10 kpc', 'Full population']

    for ax, scale, bins, label in zip(axes, scales, bin_list, labels):
        print(label)
        for pop in pops:
            co_pos = kinematics[pop.label]["pos"][co_type]

            pos_val = co_pos[:, 2].to(u.kpc).value if hasattr(co_pos, 'unit') else co_pos[:, 2]

            mask = np.abs(pos_val) < bins[-1]
            print(f"  {pop.label} fraction within 0.5 kpc {(np.abs(pos_val) < 0.5).sum() / mask.sum():1.2f}")

            ax.hist(np.abs(pos_val), bins=bins, histtype='step', lw=2, color=pop.colour)
            ax.hist(np.abs(pos_val), bins=bins, alpha=0.4, color=pop.colour,
                    label=f'{pop.label} (N={len(co_pos[mask])})')

        ax.set(
            xscale=scale,
            xlabel=r'$|z|$ [kpc]',
            ylabel=r'$N_{\rm CO}$',
        )
        ax.legend(title=label, fontsize=fs*0.5)

    if show:
        plt.show()

    return fig, axes


def plot_avg_mass_vs_z(mean_masses, labels, colours, z_maxes, z_range,
                       fig=None, ax=None, show=True, save=None, legend_kwargs={}, ax_kwargs={}):
    # plot the average mass as a function of |z|
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    for mean_bh, label, c, z_max in zip(mean_masses, labels, colours, z_maxes):
        mask = z_range < z_max
        ax.plot(z_range[mask], mean_bh[mask], lw=2, label=label, color=c)

    ax.axvspan(3, 10, color='gray', alpha=0.3, lw=0)

    ax.set(
        xscale="log",
        xlabel=r'Distance from Galactic plane, $|z|$ [kpc]',
        ylabel='Average BH Mass [M$_\odot$]',
        xlim=(z_range.min(), z_range.max()),
        **ax_kwargs
    )

    ax.set_xlabel(ax.get_xlabel(), fontsize=0.9*fs)

    ax.legend(title=legend_kwargs.pop('title', "Black holes"),
              fontsize=legend_kwargs.pop('fontsize', 0.7*fs),
              title_fontsize=legend_kwargs.pop('title_fontsize', 0.7*fs),
              **legend_kwargs)

    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))

    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax
