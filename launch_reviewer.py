"""Launch the local desktop app (or its isolated analysis worker)."""
from pathlib import Path
import os
import sys
import multiprocessing


def bootstrap():
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    dependencies = root / ".desktop-deps"
    if dependencies.exists():
        sys.path.insert(0, str(dependencies))
    workspace = root
    if "--workspace" in sys.argv:
        workspace = Path(sys.argv[sys.argv.index("--workspace") + 1]).resolve()
    elif getattr(sys, "frozen", False):
        workspace = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EEGReviewer" / "Workspace"
    cache = workspace / "desktop_data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # Numba otherwise tries writing beside read-only conda packages on import.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(cache))
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    for stream in ("stdout","stderr"):
        if getattr(sys,stream) is None:
            setattr(sys,stream,open(os.devnull,"w",encoding="utf-8"))
    return workspace


if __name__ == "__main__":
    multiprocessing.freeze_support()
    workspace = bootstrap()
    if "--worker" in sys.argv:
        from eeg_reviewer.worker import main
        raise SystemExit(main(sys.argv[sys.argv.index("--worker") + 1]))
    if "--smoke" in sys.argv:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from eeg_reviewer.app import smoke
        raise SystemExit(smoke(workspace))
    from eeg_reviewer.app import main
    raise SystemExit(main(workspace))
