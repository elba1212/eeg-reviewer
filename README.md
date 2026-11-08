# EEG Reviewer

A local Windows desktop application for reviewing sleep EEG. It supports sleep-stage estimates, spike-wave candidate review, N2/N3 SWI summaries, possible-seizure screening when a compatible model is installed, artifact and stage review, electrode distribution summaries, and local report exports.

This is research software for expert review. It is not a diagnostic device. Automated sleep stages and event flags need review by a qualified EEG reader. Scalp electrode distribution describes where a signal was recorded; it does not localize an intracranial source or establish a seizure-onset zone.

## Included formats

- EDF / EDF+
- BDF
- FIF
- Micromed VWR (requires Wonambi)
- Supported XLTEK tab-delimited text exports

For recordings in other proprietary formats, export EDF/EDF+ or a supported XLTEK text file first. Check the electrode mapping in the app before analysis, especially when source channel labels are generic or numbered.

## Run from source

Use Python 3.11 on Windows. From this folder, create a virtual environment, then install the desktop dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe launch_reviewer.py --workspace "$env:LOCALAPPDATA\EEGReviewer\Workspace"
```

Micromed VWR import needs Wonambi, which is listed in `requirements.txt`. If you do not need VWR, you can remove that dependency.

The app stores study records, conversions, model files, review edits, and reports in the selected workspace. The launch command above uses `%LOCALAPPDATA%\EEGReviewer\Workspace`, outside this source folder. Raw EDF/BDF/FIF recordings are read from their selected paths; converted VWR and XLTEK signals are stored in the workspace. The app does not upload EEG. Keep the workspace on a drive with access controls appropriate for health information.

## Seizure screening model

The application includes an experimental integration for the official SeizureTransformer research checkpoint. **The checkpoint is not included.** Without a locally installed, checksum-verified model bundle, the app reports seizure screening as unavailable; it does not treat missing results as a negative finding.

To prepare the optional model, install a compatible CPU version of PyTorch using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then run:

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'EEGReviewer\Workspace'
$upstream = Join-Path $workspace 'desktop_data\models\upstream'
.venv\Scripts\python.exe scripts/prepare_seizure_model.py --inspect
.venv\Scripts\python.exe scripts/prepare_seizure_model.py --download --destination $upstream
.venv\Scripts\python.exe scripts/prepare_seizure_model.py --export --destination $upstream --workspace $workspace
```

The preparation script checks the upstream license and preprocessing/inference parity before exporting a local CPU bundle. Screening requires an unambiguous 19-channel referential scalp montage. Missing or unsuitable channels make screening unavailable. Technical parity is not clinical validation; performance for pediatric epilepsy recordings is not established. The model's thresholded events are candidates and require expert review. Model files and weights are ignored by Git.

## Build and install a Windows app

Install the build dependency after the desktop requirements:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.venv\Scripts\python.exe scripts/build_desktop.py
```

The build output is `dist\EEGReviewer\EEGReviewer.exe`. To install it under the current user's local application data and create a Start-menu shortcut, run:

```powershell
.\Install EEG Reviewer.ps1
```

`Start EEG Reviewer.cmd` runs the built app when available, or launches the Python source version otherwise. The source fallback uses `.venv` when present; otherwise make sure `python` resolves to an environment with the desktop requirements installed.

## Git hygiene

`desktop_data/`, model weights, build outputs, virtual environments, and caches are excluded by `.gitignore`. Do not add recordings, patient exports, reports, or workspace files to this repository. This folder does not include a project license; choose a license you are authorized to grant before inviting reuse or contributions. The optional upstream model is separately licensed and is not bundled here.

