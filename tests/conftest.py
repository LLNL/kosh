import os
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
_cache_root = os.path.join(tempfile.gettempdir(
), f"kosh-mpl-cache-{os.getuid() if hasattr(os, 'getuid') else os.getpid()}")
os.environ.setdefault("MPLCONFIGDIR", _cache_root)
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
os.makedirs(_cache_root, exist_ok=True)


def _ensure_repo_kosh_on_path() -> None:
    """Ensure tests exercise the repo checkout, not a preinstalled `kosh`.

    On some HPC/module environments, `kosh` can already be present in a global
    site-packages and may take precedence under `srun`/MPI. Force the repo root
    (containing the `kosh/` package directory) to the front of `sys.path`.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    while repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    loaded = sys.modules.get("kosh")
    if loaded is None:
        return

    loaded_file = getattr(loaded, "__file__", "") or ""
    expected_prefix = os.path.join(repo_root, "kosh") + os.sep
    if not loaded_file.startswith(expected_prefix):
        sys.modules.pop("kosh", None)


_ensure_repo_kosh_on_path()


def pytest_sessionstart(session):
    """Optional debug to verify which `kosh` is imported under MPI launchers."""
    if os.environ.get("KOSH_DEBUG_IMPORT") != "1":
        return
    try:
        from mpi4py import MPI  # type: ignore
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except Exception:
        rank = "?"
        size = "?"
    try:
        import kosh  # noqa: F401
        kosh_file = getattr(kosh, "__file__", None)
    except Exception as exc:
        kosh_file = f"<import failed: {exc!r}>"
    print(
        f"DEBUG rank={rank}/{size} pid={os.getpid()} kosh.__file__={kosh_file}", flush=True)
    print(f"DEBUG rank={rank}/{size} sys.path[0:5]={sys.path[:5]}", flush=True)
