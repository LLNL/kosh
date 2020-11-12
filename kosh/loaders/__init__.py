from .core import KoshLoader, KoshFileLoader  # noqa
from .jsons import JSONLoader  # noqa
try:
    from .pil import PILLoader  # noqa
except ImportError:
    pass  # we do not have PIL
from .pgm import PGMLoader  # noqa
try:
    from .hdf5 import HDF5Loader  # noqa
except ImportError:
    pass  # we do not have h5py
try:
    from .UltraLoader import UltraLoader  # noqa
except ImportError:
    pass  # we do not have pydv
