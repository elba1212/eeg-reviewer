"""Build a standalone Windows app without modifying the analysis environment."""
from pathlib import Path
import importlib.metadata
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import launch_reviewer
launch_reviewer.bootstrap()
sys.path.insert(0,str(ROOT/".build-deps"))
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT/".build-deps"),str(ROOT/".desktop-deps"),str(ROOT),os.environ.get("PYTHONPATH","")])


def main():
    import PyInstaller.__main__
    excluded = ["PyQt5","PyQt6","PySide2","IPython","notebook","jupyter","pytest","tkinter","vtk","pyvista","pyvistaqt","tensorflow","tensorboard",
                "PySide6.QtWebEngineCore","PySide6.QtWebEngineWidgets","PySide6.QtWebEngineQuick","PySide6.QtQml","PySide6.QtQuick","PySide6.QtPdf","PySide6.QtPdfWidgets"]
    # MNE's lazy imports require collection of its signal IO/filter modules.
    mne_modules = ["mne.io.fiff.raw","mne.io.edf.edf","mne.io.array._array","mne.io.base",
                   "mne._fiff.meas_info","mne._fiff.pick","mne.annotations","mne.filter",
                   "mne.channels.montage","mne.transforms","mne.time_frequency"]
    arguments = [str(ROOT/"launch_reviewer.py"),"--name","EEGReviewer","--onedir","--windowed","--noconfirm",
                 "--distpath",str(ROOT/"dist"),"--workpath",str(ROOT/"desktop_data/build"),"--specpath",str(ROOT/"desktop_data/build"),
                 "--paths",str(ROOT/".build-deps"),"--paths",str(ROOT),"--paths",str(ROOT/".desktop-deps"),"--collect-data","yasa","--collect-data","mne","--collect-data","matplotlib",
                 "--hidden-import","eeg_reviewer.worker","--hidden-import","eeg_reviewer.pipeline","--hidden-import","eeg_reviewer.reporting",
                 "--hidden-import","eeg_reviewer.recordings","--hidden-import","eeg_reviewer.storage","--hidden-import","eeg_reviewer.scoring","--hidden-import","eeg_reviewer.seizures",
                 "--hidden-import","PySide6.QtPrintSupport"]
    for distribution in ["mne","yasa","numpy","scipy","pandas","scikit-learn","torch","wonambi"]:
        try:
            importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        arguments += ["--copy-metadata",distribution]
    for name in excluded:
        arguments += ["--exclude-module",name]
    for name in mne_modules:
        arguments += ["--hidden-import",name]
    model = ROOT/"desktop_data/models/seizuretransformer"
    if (model/"manifest.json").exists():
        arguments += ["--add-data",str(model)+os.pathsep+"models/seizuretransformer"]
    if "--clean" in sys.argv:
        arguments += ["--clean"]
    PyInstaller.__main__.run(arguments)
    print(f"Built: {ROOT/'dist/EEGReviewer/EEGReviewer.exe'}",flush=True)


if __name__ == "__main__":
    main()
