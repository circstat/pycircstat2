![logo](https://raw.githubusercontent.com/circstat/pycircstat2/main/docs/docs/images/logo.png)

# PyCircStat2: Circular statistics with Python

[![PyPI version](https://badge.fury.io/py/pycircstat2.svg)](https://badge.fury.io/py/pycircstat2)

A rework of [pycircstat](https://github.com/circstat/pycircstat).

[**Key Features**](#key-features) |
[**Installlation**](#installation) | 
[**API Reference**](#api-reference) |
[**Examples**](#example-notebooks) (
[**Books**](#books) |
[**Topics**](#topics)
)

## Key Features

- **One-Stop Circular Data Analysis Pipeline with `Circular` Class**  

    The `Circular` class simplifies circular data analysis by providing automatic data transformation, descriptive statistics, hypothesis testing, and visualization tools—all in one place.  

    ```python
    from pycircstat2 import Circular
    data = [30, 60, 90, 120, 150]
    circ = Circular(data, unit="degree")
    print(circ.summary())
    circ.plot(plot_rose=True, plot_mean=True)
    ```

- **Compatibility with Legacy APIs**  

  APIs for descriptive statistics and hypothesis testing follow the conventions established by the original [circstat-matlab](https://github.com/circstat/circstat-matlab) and [pycircstat](https://github.com/circstat/pycircstat), ensuring ease of use for existing users.


- **Wide-Ranging Circular Distributions**  

  The package supports a variety of circular distributions, including but not limited to:  
  - **Symmetric distributions**: Circular Uniform, Cardioid, Cartwright, Wrapped Normal, Wrapped Cauchy, von Mises (and its flat-top extension), and Jones-Pewsey.
  - **Asymmetric distributions**: Sine-skewed Jones-Pewsey, Asymmetric Extended Jones-Pewsey, Inverse Batschelet.

Also see the full feature checklist [here](https://circstat.github.io/pycircstat2/feature-checklist/).

## Installation

To install the latest tagged version:

```
pip install pycircstat2
```

Or to install the development version, clone the repository and install it with `pip install -e`:

```
git clone https://github.com/circstat/pycircstat2
pip install -e pycircstat2
```

## API Reference

The API reference is available [here](https://circstat.github.io/pycircstat2/reference/base/).

## Example notebooks

In the notebooks below, we reproduce examples and figures from a few textbooks on circular statistics.

### Books

-   [Statistical Analysis of Circular Data](https://github.com/circstat/pycircstat2/blob/main/examples/B1-Fisher-1993.ipynb) (Fisher, 1993)
-   [Chapter 26 and 27](https://github.com/circstat/pycircstat2/blob/main/examples/B2-Zar-2010.ipynb) from Biostatistical Analysis (Zar, 2010).
-   [Circular Statistics in R](https://github.com/circstat/pycircstat2/blob/main/examples/B3-Pewsey-2014.ipynb) (Pewsey, et al., 2014)

And a few more examples on selective topics:

### Topics

-   [Utils](https://github.com/circstat/pycircstat2/blob/main/examples/T0-utils.ipynb)
-   [Descriptive Statistics](https://github.com/circstat/pycircstat2/blob/main/examples/T1-descriptive-statistics.ipynb)
-   [Hypothesis Testing](https://github.com/circstat/pycircstat2/blob/main/examples/T2-hypothesis-testing.ipynb)
-   [Circular Models](https://github.com/circstat/pycircstat2/blob/main/examples/T3-circular-distributions.ipynb)
-   [Regression](https://github.com/circstat/pycircstat2/blob/main/examples/T4-regression.ipynb)
