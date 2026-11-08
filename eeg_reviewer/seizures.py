"""Explicitly gated, offline SeizureTransformer integration.

Only a locally installed, provenance-checked TorchScript bundle with a passing
parity record can execute. Absent/unsuitable models are never negative findings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .recordings import canonical_electrode, normalize_events


def unavailable(reason):
    return normalize_events(pd.DataFrame(), "seizure"), {"seizure_status": "unavailable", "seizure_message": reason, "seizure_coverage": []}


def check_bundle(directory):
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise ValueError("No verified seizure model is installed. Review or annotate possible events manually.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model") != "SeizureTransformer" or manifest.get("format") != "torchscript":
        raise ValueError("The installed model is not a supported SeizureTransformer bundle.")
    if not manifest.get("source_url", "").startswith("https://github.com/keruiwu/SeizureTransformer"):
        raise ValueError("The model's official source provenance has not been recorded.")
    if not manifest.get("distribution_terms") or not manifest.get("parity_passed"):
        raise ValueError("Model distribution terms and upstream inference parity have not been verified.")
    if manifest.get("fs") != 256 or manifest.get("window_samples") != 15360 or len(manifest.get("channels", [])) != 19 or manifest.get("montage") != "referential":
        raise ValueError("The official checkpoint requires 19 referential electrodes at 256 Hz in 60-second windows.")
    model_path = Path(directory) / "model.pt"
    if not model_path.exists() or hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("The installed seizure model is missing or its checksum does not match.")
    return manifest, model_path


MODEL_CHANNELS = ["Fp1","F3","C3","P3","O1","F7","T3","T5","Fz","Cz","Pz","Fp2","F4","C4","P4","O2","F8","T4","T6"]


def resolve_channels(study, required):
    mapped = {}
    for ch in study.channels:
        if ch not in study.excluded and study.roles.get(ch) == "EEG":
            mapped.setdefault(canonical_electrode(study.mapping.get(ch,ch)),[]).append(ch)
    result = []
    for channel in required:
        canonical = canonical_electrode(channel)
        matches = mapped.get(canonical,[])
        if len(matches) != 1:
            raise ValueError(f"Seizure screening requires an unambiguous referential {channel} electrode. Missing electrodes are never replaced by zeros.")
        result.append(matches[0])
    return result


def preprocess_window(values):
    """Exact filter order and 60-second padding from the official package."""
    from scipy import signal
    values = np.pad(values,((0,0),(0,15360-values.shape[1])))
    b,a = signal.butter(3,[0.5,120],btype="band",fs=256)
    values = signal.lfilter(b,a,values,axis=1)
    values = signal.lfilter(*signal.iirnotch(1,Q=30,fs=256),values,axis=1)
    values = signal.lfilter(*signal.iirnotch(60,Q=30,fs=256),values,axis=1)
    return values.astype(np.float32)


def resolve_derivations(study, derivations):
    usable = [ch for ch in study.channels if ch not in study.excluded and study.roles.get(ch) == "EEG"]
    mapped = {}
    for ch in usable:
        mapped.setdefault(canonical_electrode(study.mapping.get(ch, ch)), []).append(ch)
    plan = []
    for pair in derivations:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Invalid bipolar derivation in model manifest.")
        left, right = [canonical_electrode(x) for x in pair]
        direct = f"{left}-{right}"
        if len(mapped.get(direct, [])) == 1:
            plan.append((mapped[direct][0], None))
        elif len(mapped.get(left, [])) == 1 and len(mapped.get(right, [])) == 1:
            plan.append((mapped[left][0], mapped[right][0]))
        else:
            raise ValueError(f"Seizure screening requires unambiguous {left}-{right} data; check the electrode mapping.")
    return plan


def screen(store, study, recording, progress=lambda *args: None):
    try:
        import sys
        directory = store.models / "seizuretransformer"
        if not (directory/"manifest.json").exists() and getattr(sys,"frozen",False):
            directory = Path(sys._MEIPASS)/"models/seizuretransformer"
        manifest, path = check_bundle(directory)
        picks = resolve_channels(study, manifest["channels"])
        import torch
    except (ValueError, OSError, ImportError) as exc:
        return unavailable(str(exc))
    from scipy import signal, ndimage
    import tempfile
    try:
        torch.set_num_threads(2)
        with path.open("rb") as handle:
            model = torch.jit.load(handle, map_location="cpu").eval()
        # Stream two passes: global mean/std as upstream, then bounded windows.
        total = np.zeros(19)
        squares = np.zeros(19)
        count = 0
        full_valid = np.zeros(study.n_samples,dtype=bool)
        for start in np.arange(0, study.duration, 300.0):
            end = min(start + 300.0, study.duration)
            values = recording.get_uv(picks, start, end)
            valid = np.isfinite(values).all(axis=0)
            for span in study.bad_spans:
                a = max(0, int((span["start_sec"] - start) * study.sfreq))
                b = min(len(valid), int(np.ceil((span["end_sec"] - start) * study.sfreq)))
                if b > a:
                    valid[a:b] = False
            values = values[:, valid]
            offset = int(round(start*study.sfreq))
            full_valid[offset:offset+len(valid)] = valid
            total += values.sum(axis=1)
            squares += (values * values).sum(axis=1)
            count += values.shape[1]
        if not count:
            return unavailable("No usable EEG time for seizure screening.")
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
        if np.any(std < 1e-10):
            return unavailable("A required seizure-screening channel is flat.")
        duration_samples = int(study.n_samples * 256 / study.sfreq)
        probabilities = np.full(duration_samples, np.nan, dtype=np.float32)
        coverage = []
        from .pipeline import finite_spans, merge_intervals
        cache = store.root/"cache"
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="seizure_",dir=cache) as temporary:
            prepared = np.lib.format.open_memmap(Path(temporary)/"input.npy",mode="w+",dtype="float64",shape=(19,duration_samples))
            blocks = finite_spans(full_valid,study.sfreq)
            for index,ch in enumerate(picks):
                prepared[index,:] = np.nan
                channel = recording.get_uv([ch],0,study.duration)[0]
                for span in blocks:
                    x,y = int(round(span["start_sec"]*study.sfreq)),int(round(span["end_sec"]*study.sfreq))
                    a,b = int(span["start_sec"]*256),min(duration_samples,int(span["end_sec"]*256))
                    if b <= a:
                        continue
                    values = (channel[x:y]-mean[index])/std[index]
                    prepared[index,a:b] = values if study.sfreq == 256 and len(values) == b-a else signal.resample(values,b-a)
                progress(75+int(5*(index+1)/19),"Preparing the checkpoint's referential montage")
            prepared.flush()
            for a in range(0,duration_samples,15360):
                b = min(a+15360,duration_samples)
                values = np.asarray(prepared[:,a:b])
                if not np.isfinite(values).all():
                    continue
                data = preprocess_window(values)
                with torch.inference_mode():
                    output = model(torch.from_numpy(data)[None]).cpu().numpy().reshape(-1)
                if len(output) != 15360 or not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
                    raise ValueError("The seizure model returned invalid probabilities.")
                probabilities[a:b] = output[:b-a]
                coverage.append({"start_sec":a/256,"end_sec":b/256})
                progress(80+int(15*b/duration_samples),f"Seizure screening: {b/256/60:.1f} min")
            # Release Windows file mappings before TemporaryDirectory removes files.
            del values
            prepared._mmap.close()
            del prepared
        if not coverage:
            return unavailable("No complete usable model windows were available; screening produced no result.")
        events = []
        # Postprocess each finite block independently, never join events over gaps.
        for span in finite_spans(np.isfinite(probabilities), 256):
            a, b = int(round(span["start_sec"] * 256)), int(round(span["end_sec"] * 256))
            mask = probabilities[a:b] > float(manifest.get("threshold", 0.8))
            mask = ndimage.binary_opening(mask, structure=np.ones(5))
            mask = ndimage.binary_closing(mask, structure=np.ones(5))
            for event in finite_spans(mask, 256, span["start_sec"]):
                if event["end_sec"] - event["start_sec"] >= 2.0:
                    x, y = int(event["start_sec"] * 256), int(event["end_sec"] * 256)
                    events.append({**event, "channel": "All", "score": float(np.nanmax(probabilities[x:y]))})
        status = {"seizure_status": "completed", "seizure_message": "Experimental research-model screening; events require expert review.",
                  "seizure_coverage": merge_intervals(coverage), "model": {k: manifest[k] for k in manifest if k != "channels"},
                  "model_limitations":"Technical inference parity checked. Pediatric clinical sensitivity and false-alarm rate are not established for these recordings. The upstream non-256-Hz sample-count truncation is corrected to retain recording time."}
        return normalize_events(pd.DataFrame(events), "seizure"), status
    except Exception as exc:
        return unavailable(f"Seizure screening failed: {exc}")
