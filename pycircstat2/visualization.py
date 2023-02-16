import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from pycircstat2.descriptive import (
    circ_mean,
    compute_smooth_params,
    nonparametric_density_estimation,
)


def circ_plot(
    circ_data,
    ax=None,
    **kwargs,
):

    # process kwargs and assign defaults
    outward = kwargs.pop("outward", True)
    figsize = kwargs.pop("figsize", (5, 5))
    projection = kwargs.pop("projection", "polar")
    marker = kwargs.pop("marker", "o")
    marker_color = kwargs.pop("marker_color", "black")
    marker_size = kwargs.pop("marker_size", 10)

    bins = kwargs.pop("bins", 12)

    plot_counts = kwargs.pop("plot_counts", False)
    plot_rlabel = kwargs.pop("plot_rlabel", False)
    plot_grid = kwargs.pop("plot_grid", True)
    plot_spine = kwargs.pop("plot_spine", False)
    plot_axis = kwargs.pop("plot_axis", True)

    plot_density = kwargs.pop("plot_density", True)
    plot_rose = kwargs.pop("plot_rose", True)

    plot_mean = kwargs.pop("plot_mean", True)
    if np.isclose(circ_data.r, 0):
        plot_mean = False
        # print("Mean is not plotted because `r` is close to 0")
    mean_kwargs = kwargs.pop(
        "mean_kwargs", {"color": "black", "linestyle": "-", "kind": "arrow"}
    )

    plot_mean_ci = kwargs.pop("plot_mean_ci", True)
    if not hasattr(circ_data, "mean_ub") or np.isnan(circ_data.mean_ub):
        plot_mean_ci = False
        # print("Mean CI is not plotted because it is not computed")

    plot_median = kwargs.pop("plot_median", True)
    if plot_median and np.isnan(circ_data.median):
        plot_median = False
        # print("Median is not plotted because `median` is nan")

    median_kwargs = kwargs.pop(
        "median_kwargs", {"color": "black", "linestyle": "dotted"}
    )

    plot_median_ci = kwargs.pop("plot_median_ci", True)
    if not hasattr(circ_data, "median_ub") or np.isnan(circ_data.median_ub):
        plot_median_ci = False
        # print("Median CI is not plotted because it is not computed.")

    zero_location = kwargs.pop("zero_location", "N")
    clockwise = kwargs.pop("clockwise", -1)
    r_max_scatter = kwargs.pop("r_max_scatter", 1)
    rticks = kwargs.pop("rticks", [0, 1])
    rlim_max = kwargs.pop("rlim_max", None)

    # check axes
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": projection}, layout="constrained"
        )

    # plot
    if not circ_data.grouped:

        # plot scatter
        alpha, counts = np.unique(circ_data.alpha.round(3), return_counts=True)
        alpha = np.repeat(alpha, counts)

        if outward:
            radii = np.hstack(
                [
                    r_max_scatter + 0.05 + np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )
        else:
            radii = np.hstack(
                [
                    r_max_scatter - 0.05 - np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )
        ax.scatter(alpha, radii, color=marker_color, marker=marker, s=marker_size)

        # plot density
        if plot_density:  # and not np.isclose(circ_data.r, 0):

            kwargs_density = kwargs.pop("kwargs_density", {})
            density_method = kwargs_density.pop("method", "nonparametric")
            density_color = kwargs_density.pop("color", "black")
            density_linestyle = kwargs_density.pop("linestyle", "-")

            if density_method == "nonparametric":
                h0 = kwargs_density.pop(
                    "h0", compute_smooth_params(circ_data.r, circ_data.n)
                )
                x, f = nonparametric_density_estimation(circ_data.alpha, h0)

            elif density_method == "MoVM":

                x = np.linspace(0, 2 * np.pi, 100)
                f = circ_data.mixture_opt.predict_density(x=x, unit="radian")

            else:
                raise ValueError(
                    f"`{kwargs_density['method']}` in `kwargs_density` is not supported."
                )

            # save density to circ_data
            circ_data.density_x = x
            circ_data.density_f = f
            f_ = f + 1.05  # add the radius of the plotted circle
            ax.plot(
                x,
                f_,
                color=density_color,
                linestyle=density_linestyle,
            )
            if rlim_max is None:
                ax.set_ylim(0, f_.max())
            else:
                ax.set_ylim(0, rlim_max)
        else:
            if rlim_max is None:
                ax.set_ylim(0, radii.max() + 0.025)
            else:
                ax.set_ylim(0, rlim_max)

    # plot rose diagram
    if plot_rose:

        if not circ_data.grouped:
            alpha = circ_data.alpha
            w, beta = np.histogram(
                alpha, bins=bins, range=(0, 2 * np.pi)
            )  # np.histogram return bin edges
            beta = 0.5 * (beta[:-1] + beta[1:])
        else:
            w = circ_data.w
            beta = circ_data.alpha

        w_sqrt = np.sqrt(w)
        w_norm = w_sqrt / w_sqrt.max()

        width = kwargs.pop("width", 2 * np.pi / len(beta))

        bars = ax.bar(
            beta,
            w_norm,
            width=width,
            color="gray",
            ec="black",
            alpha=0.8,
            bottom=0,
            zorder=2,
        )
        if plot_counts:
            for i, v in enumerate(w):

                angle = rotation = beta[i].round(3)
                if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                    alignment = "right"
                    rotation = rotation + np.pi
                else:
                    alignment = "left"

                if v != 0:
                    ax.text(
                        x=angle,
                        y=bars[i].get_height() - 0.075,
                        s=str(v),
                        ha="center",
                        va="center",
                        rotation=rotation,
                        rotation_mode="anchor",
                        color="black",
                    )

        if circ_data.grouped and plot_density:
            x = np.linspace(0, 2 * np.pi, 100)
            f = circ_data.mixture_opt.predict_density(x=x, unit="radian") + 1
            ax.plot(x, f, color="black", linestyle="-")
            if rlim_max is None:
                ax.set_ylim(0, f.max())
            else:
                ax.set_ylim(0, rlim_max)
    else:
        w = circ_data.w
        rticks = [1]  # overwrite

    if plot_mean:

        radius = circ_data.r

        ax.plot(
            [0, circ_data.mean],
            [0, radius],
            color=mean_kwargs.pop("color", "black"),
            ls=mean_kwargs.pop("linestyle", "-"),
            label="mean",
            zorder=5,
        )

    if plot_mean and plot_mean_ci:

        if circ_data.mean_lb < circ_data.mean_ub:
            x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
        else:
            x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub + 2 * np.pi, num=50)

        # plot arc
        ax.plot(
            x1,
            np.ones_like(x1) * radius,
            ls="-",
            color="black",
            zorder=5,
            lw=2,
        )
        # plot arc cap
        ax.errorbar(x1[0], radius, yerr=0.03, capsize=0, color="black", lw=2)
        ax.errorbar(x1[-1], radius, yerr=0.03, capsize=0, color="black", lw=2)

    if plot_median:
        ax.plot(
            [0, circ_data.median],
            [0, 0.95],
            color=median_kwargs.pop("color", "black"),
            ls=median_kwargs.pop("linestyle", "dotted"),
            label="median",
            zorder=5,
        )

    if plot_median and plot_median_ci:
        if circ_data.median_lb < circ_data.median_ub:
            x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
        else:
            x1 = np.linspace(
                circ_data.median_lb, circ_data.median_ub + 2 * np.pi, num=50
            )
        # plot arc
        ax.plot(
            x1,
            np.ones_like(x1) - 0.05,
            ls="dotted",
            color="black",
            zorder=5,
            lw=2,
        )
        # plot arc cap
        ax.errorbar(x1[0], 0.95, yerr=0.03, capsize=0, color="black", lw=2)
        ax.errorbar(x1[-1], 0.95, yerr=0.03, capsize=0, color="black", lw=2)

    ax.set_theta_zero_location(zero_location)
    ax.set_theta_direction(clockwise)
    ax.grid(plot_grid)
    ax.axis(plot_axis)
    ax.spines["polar"].set_visible(plot_spine)
    ax.set_rgrids(rticks, ["" for _ in range(len(rticks))], fontsize=16)

    if circ_data.unit == "hour":
        position_major = np.arange(0, 2 * np.pi, 2 * np.pi / 8)
        position_minor = np.arange(0, 2 * np.pi, 2 * np.pi / 24)
        labels = [f"{i}:00" for i in np.arange(0, circ_data.n_intervals, 3)]
        ax.xaxis.set_major_locator(ticker.FixedLocator(position_major))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(position_minor))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

    gridlines = ax.yaxis.get_gridlines()
    gridlines[-1].set_color("k")
    gridlines[-1].set_linewidth(1)

    if plot_mean or plot_median:
        ax.legend(frameon=False)

    return ax
