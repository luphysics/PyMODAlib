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


import os
import uuid
import warnings
from os.path import join
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import ndarray

CACHE_FOLDER = os.environ.get("PYMODALIB_CACHE") or join(f"{Path.home()}", ".pymodalib")
os.makedirs(CACHE_FOLDER, exist_ok=True)

ssd_warning = (
    f"The cache folder is located at {CACHE_FOLDER}. It is highly recommended that you ensure the "
    f"cache folder is not located on an SSD. For more information, please see the developer guide."
)
warnings.filterwarnings("once", "The cache folder is located at")


def __generate_name() -> str:
    """
    Generates a random name for a cache file, using a UUID.
    """
    return f"{uuid.uuid4()}.dat"


def cachedarray(shape: Tuple[int, ...], dtype) -> ndarray:
    """
    Creates a Numpy 'memmap' array which is able to cache its values to disk.

    :param shape: the shape of the array
    :param dtype: the data type of the array
    :return: a 'memmap' array
    """
    warnings.warn(ssd_warning, ResourceWarning)

    name = __generate_name()
    while name in os.listdir(CACHE_FOLDER):
        name = __generate_name()

    filename = join(CACHE_FOLDER, name)
    return np.memmap(filename, shape=shape, dtype=dtype, mode="w+")


def cleanup() -> None:
    """
    Deletes all files in the cache folder.
    """
    for file in os.listdir(CACHE_FOLDER):
        if file.endswith(".dat"):
            try:
                os.remove(join(CACHE_FOLDER, file))
            except Exception as e:
                print(e)
