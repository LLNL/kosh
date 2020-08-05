from .loaders import KoshLoader  # noqa
from .core import KoshStore  # noqa
from .arrays import KoshAxis  # noqa
from .utils import create_new_db  # noqa
from .schema import KoshSchema  # noqa
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("kosh").version
except Exception:
    __version__ = "???"
