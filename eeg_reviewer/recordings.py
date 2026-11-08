"""Recording adapters, preserving source units, channel identity and gaps."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import uuid

import numpy as np

from .storage import Store, Study

SCALP = {"Fp1", "Fp2", "Fpz", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"}
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "A1": "M1", "A2": "M2"}
STANDARD = {ch.upper(): ch for ch in SCALP | {"M1", "M2"}}


def canonical_electrode(label):
    label = re.sub(r"^EEG\s+", "", str(label).strip(), flags=re.I)
    normalized = re.sub(r"[-_ ](?:REF|LE|AVG|A1|A2|M1|M2)$", "", label, flags=re.I)
    normalized = ALIASES.get(normalized.upper(), normalized)
    return STANDARD.get(normalized.upper(), label)


def role_for(label):
    upper = str(label).upper()
    if canonical_electrode(label) in SCALP or ("-" in label and all(canonical_electrode(p) in SCALP for p in label.split("-"))):
        return "EEG"
    if any(text in upper for text in ("EOG", "ROC", "LOC")):
        return "EOG"
    if "EMG" in upper or "CHIN" in upper:
        return "EMG"
    return "Other"


def default_study(source, kind, sfreq, n_samples, channels, mapping=None, **kwargs):
    mapping = mapping or {ch: canonical_electrode(ch) for ch in channels}
    roles = {ch: role_for(mapping[ch]) for ch in channels}
    picks = [ch for ch in channels if roles[ch] == "EEG"]
    eeg = next((ch for preferred in ("C3", "C4", "Cz") for ch in picks if canonical_electrode(mapping[ch]) == preferred), "")
    return Study(id=kwargs.pop("id", uuid.uuid4().hex), name=kwargs.pop("name", Path(source).stem),
                 source=str(Path(source).resolve()), kind=kind, sfreq=float(sfreq), n_samples=int(n_samples),
                 channels=channels, mapping=mapping, roles=roles, detection_channels=picks, eeg_channel=eeg, **kwargs)


def read_mne(path):
    import mne
    suffix = Path(path).suffix.lower()
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf, ".fif": mne.io.read_raw_fif}
    if suffix not in readers:
        raise ValueError("Supported inputs: EDF/EDF+, BDF, FIF, Micromed VWR, XLTEK text exports.")
    return readers[suffix](str(path), preload=False, verbose="ERROR")


class Recording:
    def __init__(self, study):
        self.study = study
        self.raw = None
        self.array = None
        if study.kind == "mne":
            self.raw = read_mne(study.source)
        else:
            self.array = np.load(study.source, mmap_mode="r", allow_pickle=False)
            if self.array.shape[0] != len(study.channels) or self.array.shape[1] < study.n_samples:
                raise ValueError("The signal array does not match its recording metadata.")

    def get_uv(self, channels, start, end):
        a = max(0, int(round(start * self.study.sfreq)))
        b = min(self.study.n_samples, int(round(end * self.study.sfreq)))
        indices = [self.study.channels.index(ch) for ch in channels]
        if self.raw is not None:
            data = self.raw.get_data(picks=indices, start=a, stop=b) * 1e6
        else:
            data = np.array(self.array[indices, a:b], dtype=np.float64)
            missing = self.study.metadata.get("missing_value")
            if missing is not None:
                data[data == missing] = np.nan
            data *= self.study.metadata.get("scale_to_uv", 1.0)
        for span in self.study.bad_spans:
            if "gap" in span.get("reason","").lower():
                x = max(0,int(np.floor(span["start_sec"]*self.study.sfreq))-a)
                y = min(b-a,int(np.ceil(span["end_sec"]*self.study.sfreq))-a)
                if y > x:
                    data[:,x:y] = np.nan
        return data

    def close(self):
        if self.raw is not None:
            self.raw.close()
        if self.array is not None and getattr(self.array, "_mmap", None) is not None:
            self.array._mmap.close()


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def import_recording(path, store, progress=lambda *args: None, identifier=None, name=None):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    progress(5, "Reading recording information")
    if suffix in {".edf", ".bdf", ".fif"}:
        raw = read_mne(path)
        try:
            bad = [{"start_sec": float(a), "end_sec": min(float(a + b), raw.n_times / raw.info["sfreq"]), "reason": str(c), "origin": "source"}
                   for a, b, c in zip(raw.annotations.onset - raw.first_time, raw.annotations.duration, raw.annotations.description) if str(c).lower().startswith("bad")]
            origin = raw.info.get("meas_date")
            study = default_study(path, "mne", raw.info["sfreq"], raw.n_times, list(raw.ch_names),
                                  id=identifier or uuid.uuid4().hex, name=name or path.stem,
                                  first_timestamp=origin.isoformat() if origin else None, bad_spans=bad,
                                  excluded=list(raw.info.get("bads", [])), metadata=source_identity(path))
            study.detection_channels = [ch for ch in study.detection_channels if ch not in study.excluded]
            for ch, kind in zip(raw.ch_names, raw.get_channel_types()):
                if kind in ("eog", "emg"):
                    study.roles[ch] = kind.upper()
        finally:
            raw.close()
    elif suffix == ".vwr":
        from wonambi.ioeeg.micromed import Micromed
        dataset = Micromed(str(path))
        _, origin, sfreq, channels, n_samples, _ = dataset.return_hdr()
        study = default_study(path, "numpy", sfreq, n_samples, list(channels), name=name or path.stem,
                              first_timestamp=origin.isoformat() if origin else None,
                              metadata={**source_identity(path), "original_source": str(path), "scale_to_uv": 1.0})
        destination = store.directory(study) / "signals_uv.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=(len(channels), n_samples))
        chunk = max(1, int(sfreq * 60))
        for start in range(0, n_samples, chunk):
            end = min(n_samples, start + chunk)
            data[:, start:end] = dataset.return_dat(list(range(len(channels))), start, end)
            progress(10 + int(85 * end / n_samples), "Converting Micromed recording")
        data.flush()
        del data
        study.source = str(destination)
    elif suffix == ".txt":
        study = import_xltek(path, store, progress)
    else:
        raise ValueError("This format needs a signal export. Export EDF/EDF+ or the supported XLTEK text format; a report PDF is not EEG signal data.")
    store.save(study)
    progress(100, "Recording imported; check electrode mapping before analysis")
    return study


def import_xltek(path, store, progress):
    from scripts.convert_xltek_txt_export import _parse_header, _sampling_rate, _source_units, _scale_factor, _detect_encoding, DATA_LINE_RE, _parse_channel_values
    header, channels, indices, _ = _parse_header(path)
    sfreq = _sampling_rate(header)
    scale = _scale_factor(_source_units(header), "uV")
    encoding = _detect_encoding(path)
    first = last = None
    timestamp = None
    count = 0
    with path.open(encoding=encoding) as handle:
        for line in handle:
            if not DATA_LINE_RE.match(line):
                continue
            parts = line.rstrip("\r\n").split("\t")
            stamp = int(parts[2])
            if last is not None and stamp <= last:
                raise ValueError("XLTEK sample stamps must increase; export a single continuous study.")
            if first is None:
                first = stamp
                from datetime import datetime
                timestamp = datetime.strptime(parts[0], "%m/%d/%Y %H:%M:%S").isoformat()
            last = stamp
            count += 1
    if first is None:
        raise ValueError("No signal samples found in export.")
    progress(30, "Preserving XLTEK acquisition timeline")
    mapping = {ch: ch for ch in channels}
    mapping_source = None
    for ent in path.parent.rglob("*.ent"):
        if ent.stem.casefold() not in header.get("Original File", "").casefold():
            continue
        lists = [re.findall(r'"([^"\r\n]+)"', s) for s in re.findall(r'ChanNames"\s*,\s*\(([^)]*)\)', ent.read_bytes().decode("utf-8", errors="replace"))]
        if lists and len(lists[0]) == len(channels) and all(row == lists[0] for row in lists):
            mapping = dict(zip(channels, lists[0]))
            mapping_source = {"file": str(ent), "sha256": hashlib.sha256(ent.read_bytes()).hexdigest()}
            break
    study = default_study(path, "numpy", sfreq, last - first + 1, channels, mapping=mapping,
                          first_timestamp=timestamp, metadata={**source_identity(path), "original_source": str(path), "scale_to_uv": 1.0, "mapping_source": mapping_source})
    destination = store.directory(study) / "signals_uv.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=(len(channels), study.n_samples))
    for start in range(0, study.n_samples, 1_000_000):
        data[:, start:start + 1_000_000] = np.nan
    previous = first - 1
    with path.open(encoding=encoding) as handle:
        for line in handle:
            if not DATA_LINE_RE.match(line):
                continue
            parts = line.rstrip("\r\n").split("\t")
            stamp = int(parts[2])
            if stamp > previous + 1:
                study.bad_spans.append({"start_sec": (previous + 1 - first) / sfreq, "end_sec": (stamp - first) / sfreq, "reason": "Acquisition gap", "origin": "source"})
            values, _ = _parse_channel_values(parts, indices)
            data[:, stamp - first] = values * scale
            previous = stamp
            if (stamp - first) % max(1, int(sfreq * 120)) == 0:
                progress(30 + int(65 * (stamp - first) / study.n_samples), "Converting XLTEK samples")
    data.flush()
    del data
    study.source = str(destination)
    return study


def normalize_events(frame, kind):
    import pandas as pd
    frame = frame.copy()
    fields = ["id", "channel", "start_sec", "end_sec", "status", "origin"]
    fields += ["peak_sec", "amplitude_uv", "robust_z"] if kind == "spike" else ["score"]
    for col in fields:
        if col not in frame:
            frame[col] = pd.Series(index=frame.index, dtype="object")
    frame["id"] = [uuid.uuid4().hex for _ in range(len(frame))]
    frame["status"] = "unreviewed"
    frame["origin"] = "automatic"
    return frame[fields].reset_index(drop=True)
