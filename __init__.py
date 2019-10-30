
version_prefix = '0.0.1'

# Get the __version__ string from the auto-generated _version.py file, if exists.
try:
    from PyDMN._version import __version__
except ImportError:
    __version__ = version_prefix

from PyDMN import means, models, util, plot

__all__ = [
    '__version__',
    'means',
    'models',
    'util',
    'plot',
]
