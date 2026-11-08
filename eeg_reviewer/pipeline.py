"""Analysis in a separate process; a run is committed only on success."""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import numpy as np
import pandas as pd

from .recordings import Recording, normalize_events
from .storage import Store


def finite_spans(valid, sfreq, offset=0.0):
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return [{"start_sec": offset + a / sfreq, "end_sec": offset + b / sfreq}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def run_analysis(store, study, progress=lambda *args: None, run_seizures=True):
    study.validate()
    import mne
    import yasa
    from scripts.swi_analysis import detect_spike_wave_candidates
    recording = Recording(study)
    try:
        progress(3, "Preparing sleep-staging channels")
        staging_picks = list(dict.fromkeys(ch for ch in [study.eeg_channel, study.eog_channel, study.emg_channel] if ch))
        data = recording.get_uv(staging_picks, 0, study.duration)
        staging_valid = np.isfinite(data).all(axis=0)
        raw = mne.io.RawArray(np.nan_to_num(data) * 1e-6,
                              mne.create_info(staging_picks, study.sfreq, [study.roles[ch].lower() for ch in staging_picks]), verbose="ERROR")
        del data
        progress(8, "Predicting 30-second sleep stages")
        # YASA's demographic models require both age and sex. Keep the signal-
        # only model when only age was supplied; never invent a demographic.
        metadata = None
        if study.age is not None and study.metadata.get("sex_male") in (0,1):
            metadata = {"age":study.age,"male":study.metadata["sex_male"]}
        model = yasa.SleepStaging(raw, study.eeg_channel, eog_name=study.eog_channel or None, emg_name=study.emg_channel or None, metadata=metadata)
        prediction = model.predict()
        labels = np.asarray(prediction.hypno if hasattr(prediction, "hypno") else prediction, dtype=object)
        labels = pd.Series(labels).replace({"WAKE": "W", "REM": "R"}).to_numpy()
        try:
            proba = prediction.proba if hasattr(prediction, "proba") else model.predict_proba()
            confidence = np.asarray(proba.max(axis=1), dtype=float)
        except (AttributeError, ValueError, TypeError):
            confidence = np.full(len(labels), np.nan)
        hypno = pd.DataFrame({"start_sec": np.arange(len(labels)) * 30.0, "end_sec": np.minimum((np.arange(len(labels)) + 1) * 30.0, study.duration),
                              "stage": labels, "original_stage": labels.copy(), "confidence": confidence})
        for index, epoch in hypno.iterrows():
            a, b = int(epoch.start_sec * study.sfreq), int(round(epoch.end_sec * study.sfreq))
            if not staging_valid[a:b].all() or any(epoch.start_sec < s["end_sec"] and epoch.end_sec > s["start_sec"] for s in study.bad_spans):
                hypno.loc[index, "stage"] = "UNKNOWN"
        del raw, model, staging_valid
        progress(28, "Detecting spike-wave candidates")
        tables, coverage, missing = [], [], []
        chunk_sec = 1800.0
        for start in np.arange(0, study.duration, chunk_sec):
            end = min(start + chunk_sec, study.duration)
            a, b = max(0.0, start - 2.0), min(study.duration, end + 2.0)
            data = recording.get_uv(study.picks, a, b)
            valid = np.isfinite(data).all(axis=0)
            for span in study.bad_spans:
                x = max(0, int(np.floor((span["start_sec"] - a) * study.sfreq)))
                y = min(len(valid), int(np.ceil((span["end_sec"] - a) * study.sfreq)))
                if y > x:
                    valid[x:y] = False
            missing.extend(finite_spans(~valid, study.sfreq, a))
            # Never filter through a missing-data interval or turn a gap into EEG.
            for span in finite_spans(valid, study.sfreq, a):
                if span["end_sec"] - span["start_sec"] < 3:
                    continue
                x = int(round((span["start_sec"] - a) * study.sfreq))
                y = int(round((span["end_sec"] - a) * study.sfreq))
                raw = mne.io.RawArray(data[:, x:y] * 1e-6, mne.create_info(study.picks, study.sfreq, "eeg"), verbose="ERROR")
                events = detect_spike_wave_candidates(raw, picks=study.picks)
                events[["start_sec", "end_sec", "peak_sec"]] += span["start_sec"]
                events = events[(events.peak_sec >= start) & (events.peak_sec < end)]
                events["start_sec"] = events.start_sec.clip(lower=span["start_sec"])
                events["end_sec"] = events.end_sec.clip(upper=span["end_sec"])
                tables.append(events)
                coverage.append({"start_sec": max(start, span["start_sec"]), "end_sec": min(end, span["end_sec"])})
            progress(28 + int(45 * end / study.duration), f"Spike detection: {end / 60:.1f} / {study.duration / 60:.1f} min")
        spikes = normalize_events(pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), "spike")
        from .seizures import screen
        progress(75, "Checking seizure-screening model")
        seizures, status = screen(store, study, recording, progress) if run_seizures else (normalize_events(pd.DataFrame(), "seizure"), {"seizure_status": "not_run", "seizure_message": "Seizure screening was not requested."})
        versions = {name: importlib.metadata.version(name) for name in ["mne", "yasa", "numpy", "scipy", "pandas"]}
        analysis = {"coverage": merge_intervals(coverage), "source": "Desktop analysis", "detector": {"threshold_z": 6.0, "min_abs_uv": 75.0, "chunk_sec": chunk_sec},
                    "versions": versions, "missing_data_spans": merge_intervals(missing), **status}
        progress(98, "Saving completed analysis")
        store.commit_analysis(study, hypno, spikes, seizures, analysis)
        progress(100, "Analysis saved")
        return study
    finally:
        recording.close()


def merge_intervals(spans):
    result = []
    for span in sorted(spans, key=lambda s: s["start_sec"]):
        a, b = float(span["start_sec"]), float(span["end_sec"])
        if b <= a:
            continue
        if result and a <= result[-1]["end_sec"] + 1e-8:
            result[-1]["end_sec"] = max(result[-1]["end_sec"], b)
        else:
            result.append({"start_sec": a, "end_sec": b})
    return result
