import os
kosh_cache_dir = os.path.join(os.environ["HOME"], ".cache", "kosh")  # noqa
from .loaders import KoshLoader, KoshSinaLoader  # noqa
from .utils import create_new_db, walk_dictionary_keys, version  # noqa
from .schema import KoshSchema  # noqa
from .operators import KoshOperator  # noqa
import pkg_resources  # noqa
from .store import KoshStore, connect  # noqa
from .dataset import KoshDataset  # noqa
try:
    d = pkg_resources.get_distribution("kosh")
    __version__ = d.version
    metadata = list(d._get_metadata(d.PKG_INFO))
    __sha__ = None
    for meta in metadata:
        if "Summary:" in meta:
            __sha__ = meta.split("(sha: ")[-1][:-1]
            break
    if __sha__ is not None:
        __version__ += "."+__sha__
except Exception:
    __version__ = "???"
    __sha__ = None
