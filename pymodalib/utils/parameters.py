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
from typing import Any, Iterable, Dict, Optional


def verify_parameter(param: Any, possible_values: Iterable[Any]) -> None:
    """
    Checks whether a parameter is in a list of possible values. This is useful when parsing parameters which have particular string values.

    :param param: the parameter
    :param possible_values: the possible values that the parameter could take
    :return: whether the parameter is a valid value
    """
    for possible_value in possible_values:
        if param == possible_value:
            return

    raise BadParametersException(
        f"Parameter '{param}' is not one of the accepted values: {possible_values}."
    )


def sanitise(dictionary: Dict) -> Dict:
    """
    Creates a sanitised copy of a dictionary, removing all None items.
    Does not modify the existing dictionary.

    :param dictionary: the dictionary to remove None items from
    :return: the new dictionary
    """
    new = {}

    for key, value in dictionary.items():
        if key is not None and value is not None:
            new[key] = value

    return new


def float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class BadParametersException(Exception):
    pass
