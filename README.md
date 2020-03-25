# PyMODAlib

[![DOI](https://zenodo.org/badge/243930888.svg)](https://zenodo.org/badge/latestdoi/243930888)
[![License: GPL](https://img.shields.io/badge/License-GPLv3-10b515.svg)](https://github.com/luphysics/PyMODAlib/blob/master/LICENSE)
[![PyPI: version](https://img.shields.io/pypi/v/PyMODAlib)](https://pypi.org/project/PyMODAlib)
[![PyPI: Python version](https://img.shields.io/pypi/pyversions/PyMODAlib)](https://pypi.org/project/PyMODAlib)
[![Code style: Black](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)

PyMODAlib is a Python library containing the algorithms used by [PyMODA](https://github.com/luphysics/PyMODA). With PyMODAlib, you can write Python scripts to perform the same calculations as PyMODA.

Some of PyMODAlib's algorithms are MATLAB-packaged libraries, while some are Python translations of algorithms belonging to [MODA](https://github.com/luphysics/MODA).

## User Guide

This section describes how to use PyMODAlib in your Python scripts.

### Installing PyMODAlib

PyMODAlib can be installed using `pip`. Open a terminal and run:

```bash
pip install pymodalib
```

> **Tip:** on macOS/Linux, replace `pip` with the correct command for your system (e.g. `pip3`).

### Updating PyMODAlib

PyMODAlib may be updated regularly. To update your installed version, open a terminal and run:

```bash
pip install -U pymodalib
```

### Current status

`PyMODAlib` is still early in development. Currently, the features implemented are:

- Wavelet transform.
- Wavelet phase coherence.
- Group coherence for one or two groups, with inter-subject surrogates.
- Detecting harmonics.
- Downsampling.

### Examples

There are examples of using PyMODAlib's functionality in the [examples](https://github.com/luphysics/PyMODAlib/tree/master/examples) directory.

To download the dependencies required to run the examples, open a terminal and run:

```bash
pip install -U pymodalib matplotlib
```

To try the examples, download the PyMODAlib repository [as a zip file](https://github.com/luphysics/PyMODAlib/zipball/master) or by using `git clone`, then run Python files from the `examples` subfolders.

### License

You may use, distribute and modify this software under the terms of the [GNU General Public License v3.0](https://opensource.org/licenses/GPL-3.0). See [LICENSE](/LICENSE).

## Developer guide

This guide is aimed at developers interested in contributing to PyMODAlib. 

### Developing PyMODAlib

When developing PyMODAlib, you can test your changes by installing the library locally in "editable" mode. From the root of the repository, run:

```bash
pip install -e .
```

> **Note:** After making changes to PyMODAlib, you don't need to run the `pip install` command again. Any Python script which imports `pymodalib` will reflect the changes immediately.

Switching back to the release version of PyMODAlib is simple:

```bash
pip uninstall pymodalib -y
pip install -U pymodalib
```

### Project structure

The public-facing API is located in the `algorithms` package. This package contains wrappers for the actual implementations, which can be found in the `implementations` package.

This structure allows the implementation to be easily changed, while ensuring that the API remains backwards-compatible.

#### Implementations

The `implementations` package contains a `matlab` package and a `python` package. The `matlab` package contains wrappers for algorithms supplied by MATLAB-packaged libraries, while the `python` package contains algorithms implemented purely in Python.

### MATLAB-packaged libraries

Currently, the MATLAB-packaged libraries are not required if functionality that depends on them is not used. MATLAB-packaged libraries are installed by downloading PyMODA and installing its dependencies.

Functions that require the MATLAB Runtime will be marked with the `matlabwrapper` decorator. This will check if the correct version of the MATLAB Runtime is installed.

> **Note:** MATLAB libraries are still incompatible with Python 3.8. When MATLAB R2020a releases, Python 3.8 support will be added but the current required version of the MATLAB Runtime will no longer be supported (users will need to upgrade to the newer Runtime).

### Packaging the project

This section describes how to publish an update to PyPI. 

From the [documentation](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project):

```bash
rm -r dist/
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
