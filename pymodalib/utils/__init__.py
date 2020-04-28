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

from numpy import ndarray


def reorient(array: ndarray) -> ndarray:
    """
    Re-orientates an array to follow PyMODAlib's preferred standard:
    arrays whose first dimension is greater than their second dimension.

    Parameters
    ----------
    array : ndarray
        The array to re-orient.

    Returns
    -------
    ndarray
        An array whose first dimension is greater than, or equal to, the second dimension.
    """
    try:
        x, y = array.shape
    except ValueError:
        return array

    if y > x:
        array = array.T

    if array.shape[1] == 1:
        return array.reshape(len(array))

    return array
