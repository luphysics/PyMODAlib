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
from pymodalib.utils import matlab_runtime


def matlabwrapper(func):
    """
    Decorator which marks a MATLAB wrapper: a function which calls a function from a MATLAB-packaged library.
    """

    def wrapper(*args, **kwargs):
        runtime_valid = matlab_runtime.is_runtime_valid()

        if not runtime_valid:
            matlab_runtime.raise_invalid_exception()

        return func(*args, **kwargs)

    return wrapper


def deprecated(func):
    """
    Decorator that marks a function as deprecated, i.e. that it should no
    longer be used and will be removed in future.
    """

    def wrapper(*args, **kwargs):
        print(f"Warning: calling deprecated function '{func.__name__}'.")
        return func(*args, **kwargs)

    return wrapper
