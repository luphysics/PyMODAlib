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

from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyMODAlib",
    version="0.1.0b1",
    packages=["pymodalib"],
    python_requires="~=3.6",
    install_requires=["numpy>=1.18.1", "scipy>=1.4.1"],
    description="Library providing Python implementations of MODA's algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luphysics/PyMODAlib",
    author="Lancaster University Physics",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="moda pymoda wavelet-transform time-frequency-analysis",
    project_urls={"Source": "https://github.com/luphysics/PyMODAlib"},
)
