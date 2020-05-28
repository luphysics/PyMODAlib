#  PyMODAlib, a Python implementation of the algorithms from MODA (Multiscale Oscillatory Dynamics Analysis).
#  Copyright (C) 2020 Lancaster University
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Functions for plotting the results of PyMODAlib algorithms.
"""
import warnings

import numpy as np
from numpy import ndarray


def contourf(
    axes,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    levels: int = 256,
    vmin=None,
    vmax=None,
    cmap=None,
    subsample: bool = True,
    subsample_width: int = 3840,
    log=False,
    *args,
    **kwargs,
) -> "matplotlib.contour.QuadContourSet":
    """
    Plots a contour plot in PyMODA style. Useful for easily plotting a wavelet transform.

    This function is a wrapper around matplotlib's 'contourf'.

    .. note::
        Most of this documentation was copied from the relevant matplotlib function, `matplotlib.pyplot.contourf`.

    .. highlight:: python
    .. code-block:: python

        \"""
        Example of plotting the wavelet transform of a signal using this function.
        \"""
        import numpy as np
        from matplotlib import pyplot as plt
        import pymodalib

        # Load the signal from a data file.
        signal = np.load("some_data_file.npy")

        # Sampling frequency of 10Hz.
        fs = 10

        # Time values for the signal.
        times = pymodalib.generate_times(signal, fs)

        # Calculate the wavelet transform.
        wt, freq = pymodalib.wavelet_transform(signal, fs)

        # Amplitude of the wavelet transform.
        amp = np.abs(wt)

        # Create the Axes object.
        fig, ax = plt.subplots()

        # Plot the wavelet transform.
        mesh1, mesh2 = np.meshgrid(times, freq)
        pymodalib.contourf(ax, mesh1, mesh2, amp)

        # Set log scale, labels etc.
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Amplitude of wavelet transform")

        # Show the plot.
        plt.show()

    Parameters
    ----------
    axes
        The Axes object to plot on.
    x, y : ndarray
        The coordinates of the values in Z.

        X and Y must both be 2-D with the same shape as Z (e.g. created via numpy.meshgrid),
        or they must both be 1-D such that len(X) == M is the number of columns in Z and
        len(Y) == N is the number of rows in Z.
    z : ndarray
        The height values over which the contour is drawn.
    levels : int
        Determines the number and positions of the contour lines / regions.

        If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.

        If array-like, draw contour lines at the specified levels. The values must be in increasing order.
    vmin : float, None
        The minimum value, used to calibrate the colormap. If None, the minimum value of the array will be used.
    vmax : float, None
        The maximum value, used to calibrate the colormap. If None, the maximum value of the array will be used.
    cmap : str, Colormap, None
        The colormap to use. If left to None, the PyMODAlib colormap will be used.
    subsample : bool
        (Default = True) Whether to subsample the data, greatly improving plotting performance.
    subsample_width : int
        (Default = 3840) The target width of the subsampled data. If this width is more than the width of the
        screen in pixels, the effect of subsampling will be negligible.
    log : bool
        (Default = False) Whether to use a logarithmic scale on the y-axis.
        This is useful when plotting a wavelet transform.
    *args : optional
        Arguments to pass to matplotlib's `contourf` function.
    *kwargs : optional
        Keyword arguments to pass to matplotlib's `contourf` function.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The value returned by matplotlib's `contourf`.
    """
    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    if cmap is None:
        cmap = colormap()

    if log:
        axes.set_yscale("log")

    try:
        if subsample:
            x = _subsample2d(x, subsample_width)
            y = _subsample2d(y, subsample_width)
            z = _subsample2d(z, subsample_width)
    except:
        warnings.warn(
            f"Could not subsample when x, y, z have dimensions {x.shape}, {y.shape}, {z.shape}. "
            f"Please use 'np.meshgrid' to ensure that the shapes are identical.",
            RuntimeWarning,
        )

    return axes.contourf(
        x, y, z, levels, vmin=vmin, vmax=vmax, cmap=cmap, *args, **kwargs
    )


def _subsample2d(arr: ndarray, width: int) -> ndarray:
    try:
        x, y = arr.shape
    except ValueError:
        return arr

    if width > arr.shape[1]:
        return arr

    factor = int(np.ceil((y / width)))
    new_shape = (x, int(np.ceil(y / factor)))

    result = np.empty(new_shape, dtype=arr.dtype)
    result[:, :] = arr[:, ::factor]

    return result


def colormap() -> "LinearSegmentedColormap":
    """
    Loads the colormap used by PyMODA.

    .. highlight:: python
    .. code-block:: python

        \"""
        Example usage of 'colormap()'.

        Assume that the data from a wavelet transform has already been calculated. 
        \"""
        import pymodalib
        from matplotlib import pyplot as plt

        cmap = pymodalib.colormap()

        # Pass the colormap to a matplotlib function using the 'cmap' keyword argument.
        # Note that this is matplotlib's 'contourf', not PyMODAlib's 'contourf'.
        plt.contourf(mesh1, mesh2, amp_wt, cmap=cmap)


    Returns
    -------
    LinearSegmentedColormap
        The colormap, as an object which can be passed to matplotlib functions.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.io import loadmat
    import os

    here = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(here, "colormap.mat")
    cmap = loadmat(filename).get("cmap")

    if cmap is None:
        warnings.warn(
            "Could not load colormap. The colormap data is not supplied with "
            "source-distributions of PyMODAlib.",
            RuntimeWarning,
        )
        return "jet"

    return LinearSegmentedColormap.from_list("colours", cmap, N=len(cmap), gamma=1.0)
