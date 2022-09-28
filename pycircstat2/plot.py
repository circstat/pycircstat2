import matplotlib.pyplot as plt
import numpy as np

from .utils import angrange


def plot_scatter(circ_data, ax=None, outward=True, **kwargs):

    assert circ_data.grouped is False, "Scatter plot is for groupped data only."

    alpha, freq = np.unique(circ_data.alpha, return_counts=True)
    alpha = np.repeat(alpha, freq, axis=0)
    if outward:
        radii = np.hstack(
            [1 + np.arange(0, 0.05 * int(f), 0.05)[: int(f)] for f in freq]
        )
    else:
        radii = np.hstack(
            [1 - np.arange(0, 0.05 * int(f), 0.05)[: int(f)] for f in freq]
        )

    if ax is None:

        figsize = kwargs["figsize"] if "figsize" in kwargs else (8, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="polar")

    if "rticks" in kwargs:
        rticks = kwargs["rticks"]
        ax.set_rticks(rticks)
    else:
        rticks = [0, 1]
        ax.set_rticks(rticks)

    zero_location = kwargs["zero_location"] if "zero_location" in kwargs else "N"
    clockwise = kwargs["clockwise"] if "clockwise" in kwargs else -1
    plot_rlabel = kwargs["plot_rlabel"] if "plot_rlabel" in kwargs else False
    plot_spine = kwargs["plot_spine"] if "plot_spine" in kwargs else False
    plot_axis = kwargs["plot_axis"] if "plot_axis" in kwargs else True

    if "plot_mean" in kwargs:
        plot_mean = kwargs["plot_mean"]
    else:
        if np.isclose(circ_data.r, 0):
            plot_mean = False
        else:
            plot_mean = True

    if "plot_mean_ci" in kwargs:
        plot_mean_ci = kwargs["plot_mean_ci"]
    else:
        if hasattr(circ_data, "mean_ub"):
            plot_mean_ci = True
        else:
            plot_mean_ci = False

    if "plot_median" in kwargs:
        plot_median = kwargs["plot_median"]
    else:
        if np.isnan(circ_data.median):
            plot_median = False
        else:
            plot_median = True

    if "plot_median_ci" in kwargs:
        plot_median_ci = kwargs["plot_median_ci"]
    else:
        if hasattr(circ_data, "median_ub"):
            plot_median_ci = True
        else:
            plot_median_ci = False

    mean_kwargs = (
        kwargs["mean_kwargs"]
        if "mean_kwargs" in kwargs
        else {"color": "red", "linestyle": "--", "kind": "arrow"}
    )
    median_kwargs = (
        kwargs["median_kwargs"]
        if "median_kwargs" in kwargs
        else {"color": "black", "linestyle": "--"}
    )
    xticks = (
        kwargs["xticks"]
        if "xticks" in kwargs
        else [x * np.pi for x in np.arange(0, 2, 0.25)]
    )

    ax.set_theta_zero_location(zero_location)
    ax.set_theta_direction(clockwise)
    ax.axis(plot_axis)
    ax.spines["polar"].set_visible(plot_spine)
    ax.set_xticks(xticks)

    if plot_rlabel is False:
        ax.set_yticklabels([])

    if plot_mean is True:
        if mean_kwargs["kind"] == "line":
            ax.plot(
                [0, circ_data.mean],
                [0, circ_data.r],
                color=mean_kwargs["color"],
                ls=mean_kwargs["linestyle"],
                label="circ mean",
            )
        elif mean_kwargs["kind"] == "arrow":
            ax.arrow(
                circ_data.mean,
                0,
                0,
                circ_data.r,
                width=0.015,
                color=mean_kwargs["color"],
                ls=mean_kwargs["linestyle"],
                length_includes_head=True,
                label="circ mean",
            )

    if plot_mean_ci is True:
        x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
        ax.plot(x1, np.ones_like(x1) * circ_data.r, ls="-", color="red", alpha=0.75)

    if plot_median_ci is True:
        x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
        ax.plot(
            x1,
            np.ones_like(x1) - 0.05,
            ls="--",
            color="black",
            alpha=0.75,
            zorder=5,
        )

    if plot_median is True:
        ax.plot(
            [0, circ_data.median],
            [0, 1],
            color=median_kwargs["color"],
            ls=median_kwargs["linestyle"],
            label="circ median",
        )
        ax.plot(
            [0, circ_data.median + np.pi],
            [0, 1],
            color=median_kwargs["color"],
            ls=median_kwargs["linestyle"],
        )

    ax.scatter(alpha, radii, color="black")
    ax.set_ylim(0, radii.max() + 0.05)

    gridlines = ax.yaxis.get_gridlines()
    idx = rticks.index(1)
    gridlines[idx].set_color("k")
    gridlines[idx].set_linewidth(1)

    if plot_mean or plot_median:
        ax.legend(frameon=False)

    return ax


def plot_rose(circ_data, ax=None, **kwargs):

    assert circ_data.grouped, "Rose plot is for grouped data only."

    if ax is None:

        figsize = kwargs["figsize"] if "figsize" in kwargs else (8, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="polar")

    width = (
        kwargs["width"]
        if "width" in kwargs
        else 2 * np.pi / len(circ_data.alpha)
        # else 0.999 * 2 * np.pi / len(circ_data.alpha)
    )
    bottom = kwargs["bottom"] if "bottom" in kwargs else 0
    zero_location = kwargs["zero_location"] if "zero_location" in kwargs else "N"
    clockwise = kwargs["clockwise"] if "clockwise" in kwargs else -1

    rticks = (
        kwargs["rticks"]
        if "rticks" in kwargs
        else np.arange(0, circ_data.w.max(), circ_data.w.max() / 5)
    )

    plot_rlabel = kwargs["plot_rlabel"] if "plot_rlabel" in kwargs else True
    plot_grid = kwargs["plot_grid"] if "plot_grid" in kwargs else True
    plot_spine = kwargs["plot_spine"] if "plot_spine" in kwargs else True
    plot_axis = kwargs["plot_axis"] if "plot_axis" in kwargs else True

    if "plot_mean" in kwargs:
        plot_mean = kwargs["plot_mean"]
    else:
        if np.isclose(circ_data.r, 0):
            plot_mean = False
        else:
            plot_mean = True

    if "plot_mean_ci" in kwargs:
        plot_mean_ci = kwargs["plot_mean_ci"]
    else:
        if hasattr(circ_data, "mean_ub"):
            plot_mean_ci = True
        else:
            plot_mean_ci = False

    if "plot_median_ci" in kwargs:
        plot_median_ci = kwargs["plot_median_ci"]
    else:
        if hasattr(circ_data, "median_ub"):
            plot_median_ci = True
        else:
            plot_median_ci = False

    if "plot_median" in kwargs:
        plot_median = kwargs["plot_median"]
    else:
        if np.isnan(circ_data.median):
            plot_median = False
        else:
            plot_median = True
    mean_kwargs = (
        kwargs["mean_kwargs"]
        if "mean_kwargs" in kwargs
        else {"color": "red", "linestyle": "--", "kind": "arrow"}
    )

    ax.set_theta_zero_location(zero_location)
    ax.set_theta_direction(clockwise)
    ax.grid(plot_grid)
    ax.axis(plot_axis)
    ax.spines["polar"].set_visible(plot_spine)
    ax.set_rticks(rticks)

    if plot_rlabel is False:
        ax.set_yticklabels([])

    if plot_mean is True:
        ax.plot(
            [0, circ_data.mean],
            [0, circ_data.w.max()],
            color="red",
            ls="--",
            zorder=3,
            label="circ mean",
        )

    if plot_mean_ci is True:
        x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
        ax.plot(
            x1,
            np.ones_like(x1) * circ_data.w.max() - 1,
            ls="-",
            color="red",
            alpha=0.75,
            zorder=5,
        )

    if plot_median is True:
        ax.plot(
            [0, circ_data.median],
            [0, circ_data.w.max()],
            color="black",
            ls="--",
            zorder=3,
            label="circ median",
        )
        # anti median
        ax.plot(
            [0, angrange(circ_data.median + np.pi)],
            [0, circ_data.w.max()],
            color="black",
            ls="--",
            zorder=3,
        )

    if plot_median_ci is True:
        x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
        ax.plot(
            x1,
            np.ones_like(x1) * circ_data.w.max() - 1,
            ls="--",
            color="black",
            alpha=0.75,
            zorder=5,
        )

    ax.bar(
        circ_data.alpha,
        circ_data.w,
        width=width,
        color="gray",
        ec="black",
        alpha=0.8,
        bottom=bottom,
        zorder=2,
    )
    ax.set_ylim(0, circ_data.w.max() + 0.05)
    if plot_mean or plot_median:
        ax.legend(frameon=False)

    return ax
