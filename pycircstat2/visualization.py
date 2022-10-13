from typing import Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from pycircstat2.utils import (compute_smooth_params,
                               nonparametric_density_estimation)

from .utils import angrange


def circ_plot(
    circ_data,
    ax=None,
    **kwargs,
):

    # process kwargs and assign defaults
    outward = kwargs.pop("outward", True)
    figsize = kwargs.pop("figsize", (8, 8))
    projection = kwargs.pop("projection", "polar")
    marker_color = kwargs.pop("marker_color", "black")
    marker = kwargs.pop("marker", "o")
    bins = kwargs.pop("bins", 12)

    plot_rlabel = kwargs.pop("plot_rlabel", False)
    plot_grid = kwargs.pop("plot_grid", True)
    plot_spine = kwargs.pop("plot_spine", False)
    plot_axis = kwargs.pop("plot_axis", True)

    plot_density = kwargs.pop("plot_density", True)
    plot_histogram = kwargs.pop("plot_histogram", True)

    plot_mean = kwargs.pop("plot_mean", True)
    if np.isclose(circ_data.r, 0):
        plot_mean = False
        print("Mean is not plotted because `r` is close to 0")
    mean_kwargs = kwargs.pop(
        "mean_kwargs", {"color": "C3", "linestyle": "-", "kind": "arrow"}
    )

    plot_mean_ci = kwargs.pop("plot_mean_ci", True)
    if not hasattr(circ_data, "mean_ub"):
        plot_mean_ci = False
        print("Mean CI is not plotted because `r` is close to 0")

    plot_median = kwargs.pop("plot_median", True)
    if np.isnan(circ_data.median):
        plot_median = False
        print("Median is not plotted because `median` is nan")

    median_kwargs = kwargs.pop("median_kwargs", {"color": "C0", "linestyle": "-"})

    plot_median_ci = kwargs.pop("plot_median_ci", True)
    if not hasattr(circ_data, "median_ub") or np.isnan(circ_data.median_ub):
        plot_median_ci = False
        print("Median CI is not plotted because it is not computed.")

    zero_location = kwargs.pop("zero_location", "N")
    clockwise = kwargs.pop("clockwise", -1)
    rticks = kwargs.pop("rticks", [0, 1])

    # check axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=projection)

    # plot
    if circ_data.grouped is not True:

        # plot scatter
        alpha, counts = np.unique(circ_data.alpha, return_counts=True)
        alpha = np.repeat(alpha, counts)

        if outward:
            radii = np.hstack(
                [
                    rticks[1] + 0.05 + np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )
        else:
            radii = np.hstack(
                [
                    rticks[1] + 0.05 - np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )

        ax.scatter(alpha, radii, color=marker_color, marker=marker)
        # ax.set_ylim(0, radii.max() + 0.05)

        if plot_density and not np.isclose(circ_data.r, 0):
            h0 = compute_smooth_params(circ_data.r, circ_data.n)
            # for h in [1.5 * h0, h0, 0.25 * h0]:
            x, f = nonparametric_density_estimation(circ_data.alpha, 0.75 * h0, 1.05)
            ax.plot(x, f, color="black", linestyle="-")

            ax.set_ylim(0, f.max())
        else:
            ax.set_ylim(0, radii.max() + 0.025)

    # plot histogram
    if plot_histogram:
        if not circ_data.grouped:
            alpha = circ_data.alpha
            w, beta = np.histogram(
                alpha, bins=bins, range=(0, 2 * np.pi)
            )  # np.histogram return bin edges
            beta = 0.5 * (beta[:-1] + beta[1:])
            w = np.sqrt(w)
            w = w / w.max()
        else:
            beta = circ_data.alpha
            w = circ_data.w
            rticks = np.linspace(0, np.ceil(np.max(w) / 5) * 5 + 5, 5)

        width = kwargs.pop("width", 2 * np.pi / len(beta))

        ax.bar(
            beta,
            w,
            width=width,
            color="gray",
            ec="black",
            alpha=0.8,
            bottom=0,
            zorder=2,
        )
    else:
        w = circ_data.w
        rticks = [1]  # overwrite

    if plot_mean:

        radius = circ_data.r if not circ_data.grouped else np.max(w) + 0.025

        ax.plot(
            [0, circ_data.mean],
            [0, radius],
            color=mean_kwargs.pop("color", "C3"),
            ls=mean_kwargs.pop("linestyle", "-"),
            label="circ mean",
        )

    if plot_mean_ci is True:

        if circ_data.mean_lb < circ_data.mean_ub:
            x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
        else:
            x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub + 2 * np.pi, num=50)

        ax.plot(
            x1,
            np.ones_like(x1) * radius,
            ls="-",
            color="C3",
            alpha=0.75,
            zorder=9,
        )

    if plot_median:
        ax.plot(
            [0, circ_data.median],
            [0, np.max(w) - 0.025],
            color=median_kwargs["color"],
            ls=median_kwargs["linestyle"],
            label="circ median",
        )

    if plot_median_ci is True:
        if circ_data.median_lb < circ_data.median_ub:
            x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
        else:
            x1 = np.linspace(
                circ_data.median_lb, circ_data.median_ub + 2 * np.pi, num=50
            )
        ax.plot(
            x1,
            np.ones_like(x1) * np.ones_like(x1) - 0.025,
            ls="-",
            color="C0",
            alpha=0.75,
            zorder=5,
        )

    ax.set_theta_zero_location(zero_location)
    ax.set_theta_direction(clockwise)
    ax.grid(plot_grid)
    ax.axis(plot_axis)
    ax.spines["polar"].set_visible(plot_spine)
    ax.set_rticks(rticks)

    if circ_data.unit == "hour":
        position_major = np.arange(0, 2 * np.pi, 2 * np.pi / 8)
        position_minor = np.arange(0, 2 * np.pi, 2 * np.pi / 24)
        labels = [f"{i}:00" for i in np.arange(0, circ_data.k, 3)]
        ax.xaxis.set_major_locator(ticker.FixedLocator(position_major))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(position_minor))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

    gridlines = ax.yaxis.get_gridlines()
    gridlines[-1].set_color("k")
    gridlines[-1].set_linewidth(1)

    if plot_mean or plot_median:
        ax.legend(frameon=False)

    return ax


# def plot_scatter(circ_data, ax=None, **kwargs):

#     assert circ_data.grouped is False, "Scatter plot is for groupped data only."

#     alpha, counts = np.unique(circ_data.alpha, return_counts=True)
#     alpha = np.repeat(alpha, counts, axis=0)

#     outward = kwargs["outward"] if "outward" in kwargs else True
#     if outward:
#         radii = np.hstack(
#             [1 + np.arange(0, 0.05 * int(f), 0.05)[: int(f)] for f in counts]
#         )
#     else:
#         radii = np.hstack(
#             [1 - np.arange(0, 0.05 * int(f), 0.05)[: int(f)] for f in counts]
#         )

#     if ax is None:
#         figsize = kwargs["figsize"] if "figsize" in kwargs else (8, 8)
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(projection="polar")

#     rticks = kwargs["rticks"] if "rticks" in kwargs else [0, 1]
#     ax.set_rticks(rticks)

#     zero_location = kwargs["zero_location"] if "zero_location" in kwargs else "N"
#     clockwise = kwargs["clockwise"] if "clockwise" in kwargs else -1
#     plot_rlabel = kwargs["plot_rlabel"] if "plot_rlabel" in kwargs else False
#     plot_spine = kwargs["plot_spine"] if "plot_spine" in kwargs else False
#     plot_axis = kwargs["plot_axis"] if "plot_axis" in kwargs else True

#     if "plot_mean" in kwargs:
#         plot_mean = kwargs["plot_mean"]
#     else:
#         if np.isclose(circ_data.r, 0):
#             plot_mean = False
#         else:
#             plot_mean = True

#     if "plot_mean_ci" in kwargs:
#         plot_mean_ci = kwargs["plot_mean_ci"]
#     else:
#         if hasattr(circ_data, "mean_ub"):
#             plot_mean_ci = True
#         else:
#             plot_mean_ci = False

#     if "plot_median" in kwargs:
#         plot_median = kwargs["plot_median"]
#     else:
#         if np.isnan(circ_data.median):
#             plot_median = False
#         else:
#             plot_median = True

#     if "plot_median_ci" in kwargs:
#         plot_median_ci = kwargs["plot_median_ci"]
#     else:
#         if hasattr(circ_data, "median_ub"):
#             plot_median_ci = True
#         else:
#             plot_median_ci = False

#     mean_kwargs = (
#         kwargs["mean_kwargs"]
#         if "mean_kwargs" in kwargs
#         else {"color": "red", "linestyle": "--", "kind": "arrow"}
#     )
#     median_kwargs = (
#         kwargs["median_kwargs"]
#         if "median_kwargs" in kwargs
#         else {"color": "black", "linestyle": "--"}
#     )
#     xticks = (
#         kwargs["xticks"]
#         if "xticks" in kwargs
#         else [x * np.pi for x in np.arange(0, 2, 0.25)]
#     )

#     ax.set_theta_zero_location(zero_location)
#     ax.set_theta_direction(clockwise)
#     ax.axis(plot_axis)
#     ax.spines["polar"].set_visible(plot_spine)
#     ax.set_xticks(xticks)

#     if plot_rlabel is False:
#         ax.set_yticklabels([])

#     if plot_mean is True:
#         if mean_kwargs["kind"] == "line":
#             ax.plot(
#                 [0, circ_data.mean],
#                 [0, circ_data.r],
#                 color=mean_kwargs["color"],
#                 ls=mean_kwargs["linestyle"],
#                 label="circ mean",
#             )
#         elif mean_kwargs["kind"] == "arrow":
#             ax.arrow(
#                 circ_data.mean,
#                 0,
#                 0,
#                 circ_data.r,
#                 width=0.015,
#                 color=mean_kwargs["color"],
#                 ls=mean_kwargs["linestyle"],
#                 length_includes_head=True,
#                 label="circ mean",
#             )

#     if plot_mean_ci is True:
#         x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
#         ax.plot(x1, np.ones_like(x1) * circ_data.r, ls="-", color="red", alpha=0.75)

#     if plot_median_ci is True:
#         x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
#         ax.plot(
#             x1,
#             np.ones_like(x1) - 0.05,
#             ls="--",
#             color="black",
#             alpha=0.75,
#             zorder=5,
#         )

#     if plot_median is True:
#         ax.plot(
#             [0, circ_data.median],
#             [0, 1],
#             color=median_kwargs["color"],
#             ls=median_kwargs["linestyle"],
#             label="circ median",
#         )
#         ax.plot(
#             [0, circ_data.median + np.pi],
#             [0, 1],
#             color=median_kwargs["color"],
#             ls=median_kwargs["linestyle"],
#         )

#     ax.scatter(alpha, radii, color="black")
#     ax.set_ylim(0, radii.max() + 0.05)

#     gridlines = ax.yaxis.get_gridlines()
#     idx = rticks.index(1)
#     gridlines[idx].set_color("k")
#     gridlines[idx].set_linewidth(1)

#     if plot_mean or plot_median:
#         ax.legend(frameon=False)

#     return ax


# def plot_rose(circ_data, ax=None, **kwargs):

#     assert circ_data.grouped, "Rose plot is for grouped data only."

#     if ax is None:

#         figsize = kwargs["figsize"] if "figsize" in kwargs else (8, 8)
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(projection="polar")

#     width = (
#         kwargs["width"]
#         if "width" in kwargs
#         else 2 * np.pi / len(circ_data.alpha)
#         # else 0.999 * 2 * np.pi / len(circ_data.alpha)
#     )
#     bottom = kwargs["bottom"] if "bottom" in kwargs else 0
#     zero_location = kwargs["zero_location"] if "zero_location" in kwargs else "N"
#     clockwise = kwargs["clockwise"] if "clockwise" in kwargs else -1

#     rticks = (
#         kwargs["rticks"]
#         if "rticks" in kwargs
#         else np.arange(0, circ_data.w.max(), circ_data.w.max() / 5)
#     )

#     plot_rlabel = kwargs["plot_rlabel"] if "plot_rlabel" in kwargs else True
#     plot_grid = kwargs["plot_grid"] if "plot_grid" in kwargs else True
#     plot_spine = kwargs["plot_spine"] if "plot_spine" in kwargs else True
#     plot_axis = kwargs["plot_axis"] if "plot_axis" in kwargs else True

#     if "plot_mean" in kwargs:
#         plot_mean = kwargs["plot_mean"]
#     else:
#         if np.isclose(circ_data.r, 0):
#             plot_mean = False
#         else:
#             plot_mean = True

#     if "plot_mean_ci" in kwargs:
#         plot_mean_ci = kwargs["plot_mean_ci"]
#     else:
#         if hasattr(circ_data, "mean_ub"):
#             plot_mean_ci = True
#         else:
#             plot_mean_ci = False

#     if "plot_median_ci" in kwargs:
#         plot_median_ci = kwargs["plot_median_ci"]
#     else:
#         if hasattr(circ_data, "median_ub"):
#             plot_median_ci = True
#         else:
#             plot_median_ci = False

#     if "plot_median" in kwargs:
#         plot_median = kwargs["plot_median"]
#     else:
#         if np.isnan(circ_data.median):
#             plot_median = False
#         else:
#             plot_median = True
#     mean_kwargs = (
#         kwargs["mean_kwargs"]
#         if "mean_kwargs" in kwargs
#         else {"color": "red", "linestyle": "--", "kind": "arrow"}
#     )

#     ax.set_theta_zero_location(zero_location)
#     ax.set_theta_direction(clockwise)
#     ax.grid(plot_grid)
#     ax.axis(plot_axis)
#     ax.spines["polar"].set_visible(plot_spine)
#     ax.set_rticks(rticks)

#     if plot_rlabel is False:
#         ax.set_yticklabels([])

#     if plot_mean is True:
#         ax.plot(
#             [0, circ_data.mean],
#             [0, circ_data.w.max()],
#             color="red",
#             ls="--",
#             zorder=3,
#             label="circ mean",
#         )

#     if plot_mean_ci is True:
#         x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
#         ax.plot(
#             x1,
#             np.ones_like(x1) * circ_data.w.max() - 1,
#             ls="-",
#             color="red",
#             alpha=0.75,
#             zorder=5,
#         )

#     if plot_median is True:
#         ax.plot(
#             [0, circ_data.median],
#             [0, circ_data.w.max()],
#             color="black",
#             ls="--",
#             zorder=3,
#             label="circ median",
#         )
#         # anti median
#         ax.plot(
#             [0, angrange(circ_data.median + np.pi)],
#             [0, circ_data.w.max()],
#             color="black",
#             ls="--",
#             zorder=3,
#         )

#     if plot_median_ci is True:
#         x1 = np.linspace(circ_data.median_lb, circ_data.median_ub, num=50)
#         ax.plot(
#             x1,
#             np.ones_like(x1) * circ_data.w.max() - 1,
#             ls="--",
#             color="black",
#             alpha=0.75,
#             zorder=5,
#         )

#     ax.bar(
#         circ_data.alpha,
#         circ_data.w,
#         width=width,
#         color="gray",
#         ec="black",
#         alpha=0.8,
#         bottom=bottom,
#         zorder=2,
#     )
#     ax.set_ylim(0, circ_data.w.max() + 0.05)
#     if plot_mean or plot_median:
#         ax.legend(frameon=False)

#     return ax
