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
from collections import abc
from typing import List, Union, Dict, Any, Tuple

import numpy as np
from numpy import ndarray


def matlab_to_numpy(data: Union[ndarray, Dict]) -> Union[ndarray, Dict]:
    """
    Converts a matlab array to a numpy array. Can be much faster than simply calling "np.asarray()",
    for real arrays.

    Parameters
    ----------
    data : array_like, Dict
        The MATLAB array to convert. If a dict, all values which are MATLAB arrays will be converted.

    Returns
    -------
    ndarray, Dict
        If an array was passed, returns an `ndarray`; otherwise, a dict whose values
        have been converted to `ndarray` if necessary.
    """
    if isinstance(data, dict):
        return __dict_convert(data)

    return __array_convert(data)


def __dict_convert(data: Dict) -> Dict:
    """
    Recursively converts a dict's values to `ndarray` types, if they are Matlab array types.

    Works with:
        - Dictionaries where only some of their values are Matlab array types.
        - Nested dictionaries.
        - Values which are Matlab array types.
        - Values which are lists of Matlab array types.

    Does not work with values which are nested lists of Matlab array types.

    Parameters
    ----------
    data : Dict
        Dictionary containing values, some of which may be Matlab array types. Can be a nested dictionary.

    Returns
    -------
    Dict
        Dictionary whose Matlab-type values have been converted to `ndarray` types.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = __dict_convert(value)

        elif is_matlab_array(value):
            data[key] = matlab_to_numpy(value)

        elif isinstance(value, abc.Iterable) and all(
            [is_matlab_array(i) for i in value]
        ):
            data[key] = [matlab_to_numpy(i) for i in value]

    return data


def is_matlab_array(value: Any) -> bool:
    """
    Returns whether a variable is a Matlab type.
    """
    return "mlarray" in f"{type(value)}"


def __array_convert(data: "mlarray") -> ndarray:
    """
    Converts a Matlab array type to a `ndarray`.
    """
    try:
        # Should work for real arrays, maybe not for complex arrays.
        result = np.array(data._data).reshape(data.size, order="F")
    except:
        result = np.array(data)

    return result


def multi_matlab_to_numpy(*args) -> Tuple[Any, ...]:
    """
    Converts multiple matlab arrays to numpy arrays using `matlab_to_numpy()`.

    This allows code like the following:

    `x, y = multi_matlab_to_numpy(x, y)`

    Parameters
    ----------
    args : array_like
        The arrays to convert.

    Returns
    -------
    List[ndarray]
        Ordered list containing the Numpy equivalent of each MATLAB array.
    """
    out = []

    for arr in args:
        out.append(matlab_to_numpy(arr))

    return (*out,)


def multi_nested_matlab_to_numpy(*args) -> Tuple[Any]:
    return (*[nested_matlab_to_numpy(a) for a in args],)


def nested_matlab_to_numpy(item) -> Any:
    if is_matlab_array(item):
        converted = __array_convert(item)
    elif isinstance(item, Dict):
        converted = __dict_convert(item)
    elif not isinstance(item, ndarray) and isinstance(item, abc.Iterable):
        converted = nested_matlab_to_numpy(*item)
    else:
        converted = item

    return converted
