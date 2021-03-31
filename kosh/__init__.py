import os
kosh_cache_dir = os.path.join(os.environ["HOME"], ".cache", "kosh")  # noqa
from .loaders import KoshLoader  # noqa
from .core import KoshStore  # noqa
from .utils import create_new_db  # noqa
from .schema import KoshSchema  # noqa
from .operators import KoshOperator  # noqa
import pkg_resources  # noqa

try:
    __version__ = pkg_resources.get_distribution("kosh").version
except Exception:
    __version__ = "???"
