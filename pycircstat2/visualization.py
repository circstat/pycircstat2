import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from pycircstat2.descriptive import (
    compute_smooth_params,
    nonparametric_density_estimation,
)


# Recursive function to merge user config into defaults
def _merge_dicts(defaults, overrides):
    """Recursively merge overrides into defaults without modifying the originals.
    
    - If `overrides[key] == False`: The feature is explicitly disabled.
    - If `overrides[key] == True`: The feature is enabled with default settings.
    - If `overrides[key]` is a dictionary, it merges with the defaults, **overwriting any `False` values**.
    """
    merged = copy.deepcopy(defaults)  # Ensure defaults remain unchanged
    for key, value in overrides.items():
        if value is False:  # Explicitly disable the feature
            merged[key] = False  
        elif value is True:  # Enable feature with default settings
            merged[key] = copy.deepcopy(defaults[key])
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Merge and ensure False values are correctly overridden
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and isinstance(merged[key].get(sub_key), dict):
                    merged[key][sub_key] = _merge_dicts(merged[key][sub_key], sub_value)
                else:
                    merged[key][sub_key] = sub_value  # Overwrite
        else:
            merged[key] = value  # Override with new value
    return merged


DEFAULT_CIRC_PLOT_CONFIG = {

    "figsize": (5, 5),
    "projection": "polar",
    "grid": True,
    "spine": False,
    "axis": True,
    "outward": True,
    "zero_location": "N",
    "clockwise": -1,
    "radius": {       
        "ticks": [0, 1],
        "lim_max": None,
    },
    "scatter": {
        "marker": "o",
        "color": "black",
        "size": 10,
        "r_start": 1,
        },
    "rose": {
        "bins": 12,
        "counts": False,
        "color": "gray",
        "edgecolor": "black",
        "alpha": 0.5,
    },
    "density": {
        "method": "nonparametric",
        "color": "black",
        "linestyle": "-",
    },
    "mean": {
        "color": "black",
        "linestyle": "-",
        "kind": "arrow",
        "ci": True,
    },
    "median": {
        "color": "black",
        "linestyle": "dotted",
        "ci": True,
    },
}


def circ_plot(
    circ_data,
    ax=None,
    config=None,
):

    """Plots circular data with various visualization options.

    Parameters
    ----------
    circ_data : Circular
        A Circular object containing the data to plot.
    ax : matplotlib.axes._axes.Axes, optional
        The axis to plot on. If None, a new figure is created.
    config : dict, optional
        Configuration dictionary that overrides defaults.

        - **"figsize"** : tuple, default=(5, 5)  
            Size of the figure in inches.
        
        - **"projection"** : str, default="polar"  
            Type of projection used for the plot.

        - **"grid"** : bool, default=True  
            Whether to display grid lines.

        - **"spine"** : bool, default=False  
            Whether to show the polar spine.

        - **"axis"** : bool, default=True  
            Whether to display the axis.

        - **"outward"** : bool, default=True  
            Determines whether scatter points are plotted outward or inward.

        - **"zero_location"** : str, default="N"  
            The reference direction for 0 degrees (e.g., "N", "E", "S", "W").

        - **"clockwise"** : int, default=-1  
            Direction of angle increase: -1 for clockwise, 1 for counterclockwise.

        - **"radius"** : dict  
            Controls radial axis settings:
            - **"ticks"** : list, default=[0, 1]  
                Radial tick values.
            - **"lim_max"** : float or None, default=None  
                Maximum radial axis limit.

        - **"scatter"** : dict  
            Controls scatter plot settings:
            - **"marker"** : str, default="o"  
                Marker style for scatter points.
            - **"color"** : str, default="black"  
                Color of scatter points.
            - **"size"** : int, default=10  
                Size of scatter markers.
            - **"r_start"** : float, default=1  
                Starting radius for scatter points.

        - **"rose"** : dict  
            Controls rose diagram settings:
            - **"bins"** : int, default=12  
                Number of bins for histogram.
            - **"counts"** : bool, default=False  
                Whether to display counts on bars.

        - **"density"** : dict or bool  
            Controls density estimation settings:
            - **If False**, disables density plotting.
            - **If True**, uses default settings.
            - **If dict**, allows customization:
                - **"method"** : str, default="nonparametric"  
                    Method for density estimation ("nonparametric" or "MovM").
                - **"color"** : str, default="black"  
                    Color of the density line.
                - **"linestyle"** : str, default="-"  
                    Line style of the density plot.

        - **"mean"** : dict or bool  
            Controls mean direction plotting:
            - **If False**, disables mean plot.
            - **If True**, uses default settings.
            - **If dict**, allows customization:
                - **"color"** : str, default="black"  
                    Color of the mean line.
                - **"linestyle"** : str, default="-"  
                    Line style of the mean plot.
                - **"kind"** : str, default="arrow"  
                    Type of mean representation.
                - **"ci"** : bool, default=True  
                    Whether to display mean confidence intervals.

        - **"median"** : dict or bool  
            Controls median direction plotting:
            - **If False**, disables median plot.
            - **If True**, uses default settings.
            - **If dict**, allows customization:  
                - **"color"** : str, default="black"  
                    Color of the median line.
                - **"linestyle"** : str, default="dotted"  
                    Line style of the median plot.
                - **"ci"** : bool, default=True  
                    Whether to display median confidence intervals.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object containing the plot.
    """

    # Merge user config with defaults recursively
    config = _merge_dicts(DEFAULT_CIRC_PLOT_CONFIG, config or {})
    
    # check axes
    if ax is None:
        fig, ax = plt.subplots(
            figsize=config["figsize"],
            subplot_kw={"projection": config["projection"]},
            layout="constrained",
        )

    # plot
    if not circ_data.grouped:

        # plot scatter
        alpha, counts = np.unique(circ_data.alpha.round(3), return_counts=True)
        alpha = np.repeat(alpha, counts)

        if config["outward"]:
            radii = np.hstack(
                [
                    config["scatter"]["r_start"]
                    + 0.05
                    + np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )
        else:
            radii = np.hstack(
                [
                    config["scatter"]["r_start"]
                    - 0.05
                    - np.arange(0, 0.05 * int(f), 0.05)[: int(f)]
                    for f in counts
                ]
            )
        ax.scatter(
            alpha, radii, 
            marker=config["scatter"]["marker"], 
            color=config["scatter"]["color"], 
            s=config["scatter"]["size"]
        )

        # plot density
        if config["density"]:  # and not np.isclose(circ_data.r, 0):

            density_method = config["density"].get("method", "nonparametric")
            density_color = config["density"].get("color", "black")
            density_linestyle = config["density"].get("linestyle", "-")

            if density_method == "nonparametric":
                h0 = config["density"].get(
                    "h0", compute_smooth_params(circ_data.r, circ_data.n)
                )
                x, f = nonparametric_density_estimation(circ_data.alpha, h0)

            elif density_method == "MovM":

                x = np.linspace(0, 2 * np.pi, 100)
                f = circ_data.mixture_opt.predict_density(x=x, unit="radian")

            else:
                raise ValueError(
                    f"`{config['density']['method']}` in `density` is not supported."
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
            if config["radius"]["lim_max"] is None:
                ax.set_ylim(0, f_.max())
            else:
                ax.set_ylim(0, config["radius"]["lim_max"])
        else:
            if config["radius"]["lim_max"] is None:
                ax.set_ylim(0, radii.max() + 0.025)
            else:
                ax.set_ylim(0, config["radius"]["lim_max"])

    # plot rose diagram
    if config["rose"]:

        if not circ_data.grouped:
            alpha = circ_data.alpha
            w, beta = np.histogram(
                alpha, bins=config["rose"]["bins"], range=(0, 2 * np.pi)
            )  # np.histogram return bin edges
            beta = 0.5 * (beta[:-1] + beta[1:])
        else:
            w = circ_data.w
            beta = circ_data.alpha

        w_sqrt = np.sqrt(w)
        w_norm = w_sqrt / w_sqrt.max()

        width = config.get("width", 2 * np.pi / len(beta))

        bars = ax.bar(
            beta,
            w_norm,
            width=width,
            color=config["rose"]["color"],
            ec=config["rose"]["edgecolor"],
            alpha=config["rose"]["alpha"],
            bottom=0,
            zorder=2,
        )
        if config["rose"]["counts"]:
            
            for i, v in enumerate(w):

                angle = rotation = beta[i].round(3)
                if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                    # alignment = "right"
                    rotation = rotation + np.pi
                # else:
                #     alignment = "left"

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

        if circ_data.grouped and config["density"]:
            x = np.linspace(0, 2 * np.pi, 100)
            f = circ_data.mixture_opt.predict_density(x=x, unit="radian") + 1
            ax.plot(x, f, color="black", linestyle="-")
            if config["rlim_max"] is None:
                ax.set_ylim(0, f.max())
            else:
                ax.set_ylim(0, config["rlim_max"])
    else:
        w = circ_data.w
        config["radius"]["ticks"] = [1]  # overwrite

    if config["mean"]:

        radius = circ_data.r

        ax.plot(
            [0, circ_data.mean],
            [0, radius],
            color=config["mean"].get("color", "black"),
            ls=config["mean"].get("linestyle", "-"),
            label="mean",
            zorder=5,
        )

        if config["mean"]["ci"]:

            if circ_data.mean_lb < circ_data.mean_ub:
                x1 = np.linspace(circ_data.mean_lb, circ_data.mean_ub, num=50)
            else:
                x1 = np.linspace(
                    circ_data.mean_lb, circ_data.mean_ub + 2 * np.pi, num=50
                )

            # plot arc
            ax.plot(
                x1,
                np.ones_like(x1) * radius,
                ls="-",
                color=config["mean"]["color"],
                zorder=5,
                lw=2,
            )
            # plot arc cap
            ax.errorbar(x1[0], radius, yerr=0.03, capsize=0, color=config["mean"]["color"], lw=2)
            ax.errorbar(x1[-1], radius, yerr=0.03, capsize=0, color=config["mean"]["color"], lw=2)

    if config["median"]:
        ax.plot(
            [0, circ_data.median],
            [0, 0.95],
            color=config["median"]["color"],
            ls=config["median"].get("linestyle", "dotted"),
            label="median",
            zorder=5,
        )

        if config["median"]["ci"]:
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
                color=config["median"]["color"],
                zorder=5,
                lw=2,
            )
            # plot arc cap
            ax.errorbar(x1[0], 0.95, yerr=0.03, capsize=0, color=config["median"]["color"], lw=2)
            ax.errorbar(x1[-1], 0.95, yerr=0.03, capsize=0, color=config["median"]["color"], lw=2)

    ax.set_theta_zero_location(config["zero_location"])
    ax.set_theta_direction(config["clockwise"])
    ax.grid(config["grid"])
    ax.axis(config["axis"])
    ax.spines["polar"].set_visible(config["spine"])
    ax.set_rgrids(config["radius"]["ticks"], ["" for _ in range(len(config["radius"]["ticks"]))], fontsize=16)

    if circ_data.unit == "hour":
        position_major = np.arange(0, 2 * np.pi, 2 * np.pi / 8)
        position_minor = np.arange(0, 2 * np.pi, 2 * np.pi / 24)
        labels = [f"{i}:00" for i in np.arange(0, circ_data.full_cycle, 3)]
        ax.xaxis.set_major_locator(ticker.FixedLocator(position_major))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(position_minor))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

    gridlines = ax.yaxis.get_gridlines()
    gridlines[-1].set_color("k")
    gridlines[-1].set_linewidth(1)

    if config["median"] or config["mean"]:
        ax.legend(frameon=False)

    return ax