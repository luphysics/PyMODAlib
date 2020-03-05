# PyMODAlib

![License: GPL](https://img.shields.io/badge/License-GPLv3-blue.svg)
![PyPI: version](https://img.shields.io/pypi/v/PyMODAlib)
![PyPI: Python version](https://img.shields.io/pypi/pyversions/PyMODAlib)

PyMODAlib is a Python library containing the algorithms used by [PyMODA](https://github.com/luphysics/PyMODA). PyMODAlib can be used to programmatically perform time-frequency analysis.

Some of PyMODAlib's algorithms are MATLAB-packaged libraries, while some are Python translations of algorithms belonging to [MODA](https://github.com/luphysics/MODA).

## Installing PyMODAlib

PyMODAlib can be installed using `pip`:

```bash
pip install pymodalib
```

## Current status

`PyMODAlib` is still early in development. Currently, the features implemented are:

- Detecting harmonics.

## Examples

There are examples of using PyMODAlib's functionality in the `examples` directory.

## License

You may use, distribute and modify this software under the terms of the [GNU General Public License v3.0](https://opensource.org/licenses/GPL-3.0). See [LICENSE](/LICENSE).

## Developer notes

### Developing PyMODAlib

When developing PyMODAlib, you can test your changes by installing the library locally in "editable" mode. From the root of the repository, run:

```bash
pip install -e .
```

> **Tip:** After making changes to PyMODAlib, you don't need to run the `pip install` command again.

Switching back to the release version of PyMODAlib is simple:

```bash
pip uninstall pymodalib -y
pip install pymodalib
```

### Packaging the project

From the [documentation](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project):

```bash
rm -r dist/
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
