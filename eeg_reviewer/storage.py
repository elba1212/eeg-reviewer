"""Study manifests and atomic, versioned analysis storage. No EEG dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


@dataclass
class Study:
    id: str
    name: str
    source: str
    kind: str
    sfreq: float
    n_samples: int
    channels: list[str]
    mapping: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)
    detection_channels: list[str] = field(default_factory=list)
    eeg_channel: str = ""
    eog_channel: str = ""
    emg_channel: str = ""
    reference: str = "Unknown / as recorded"
    age: float | None = None
    recording_type: str = "unknown"
    first_timestamp: str | None = None
    bad_spans: list[dict] = field(default_factory=list)
    reviewed_spans: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    active_run: str = ""
    narrative: str = ""
    settings_changed: bool = False

    @property
    def duration(self):
        return self.n_samples / self.sfreq

    @property
    def picks(self):
        return [ch for ch in self.detection_channels if ch not in self.excluded and self.roles.get(ch) == "EEG"]

    def validate(self):
        if not math.isfinite(self.sfreq) or self.sfreq <= 0 or self.n_samples <= 0:
            raise ValueError("Recording sample rate and length must be positive.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Channel names must be unique.")
        if self.age is not None and (not math.isfinite(self.age) or not 0 < self.age < 120):
            raise ValueError("Age must be greater than zero and below 120 years, or left unknown.")
        if self.eeg_channel not in self.channels or self.eeg_channel in self.excluded:
            raise ValueError("Select a usable sleep-staging EEG channel.")
        if self.roles.get(self.eeg_channel) != "EEG":
            raise ValueError("The staging channel must have EEG role.")
        if not self.picks:
            raise ValueError("Select at least one usable EEG channel for spike detection.")
        from .recordings import canonical_electrode, SCALP
        if canonical_electrode(self.mapping.get(self.eeg_channel, "")) not in SCALP:
            raise ValueError("Map the staging channel to a verified scalp electrode first.")
        def mapped_scalp(label):
            return canonical_electrode(label) in SCALP or ("-" in label and all(canonical_electrode(part) in SCALP for part in label.split("-")))
        unmapped = [ch for ch in self.picks if not mapped_scalp(self.mapping.get(ch,""))]
        if unmapped:
            raise ValueError("Verify the electrode mapping for detection channels: " + ", ".join(unmapped))
        for ch, role in [(self.eog_channel, "EOG"), (self.emg_channel, "EMG")]:
            if ch and (ch not in self.channels or ch in self.excluded or self.roles.get(ch) != role):
                raise ValueError(f"Select a usable {role} channel or leave it blank.")


class Store:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "desktop_data"
        self.studies = self.root / "studies"
        self.models = self.root / "models"

    def directory(self, study_or_id):
        identifier = study_or_id.id if isinstance(study_or_id, Study) else study_or_id
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier):
            raise ValueError("Invalid internal study identifier")
        return self.studies / identifier

    def save(self, study):
        write_json(self.directory(study) / "study.json", asdict(study))

    def load(self, identifier):
        values = json.loads((self.directory(identifier) / "study.json").read_text(encoding="utf-8"))
        return Study(**{k: v for k, v in values.items() if k in Study.__dataclass_fields__})

    def list_studies(self):
        result = []
        for path in sorted(self.studies.glob("*/study.json")):
            try:
                result.append(self.load(path.parent.name))
            except (ValueError, OSError, TypeError):
                continue
        return sorted(result, key=lambda s: (s.first_timestamp or "", s.name))

    def load_frames(self, study):
        import pandas as pd
        columns = {
            "hypnogram": ["start_sec", "end_sec", "stage", "original_stage", "confidence"],
            "spikes": ["id", "channel", "start_sec", "end_sec", "peak_sec", "amplitude_uv", "robust_z", "status", "origin"],
            "seizures": ["id", "channel", "start_sec", "end_sec", "score", "status", "origin"],
        }
        directory = self.directory(study) / "runs" / study.active_run
        frames = {}
        for name, fields in columns.items():
            path = directory / f"{name}.csv"
            frames[name] = pd.read_csv(path, keep_default_na=False) if study.active_run and path.exists() else pd.DataFrame(columns=fields)
        path = directory / "analysis.json"
        frames["analysis"] = json.loads(path.read_text(encoding="utf-8")) if study.active_run and path.exists() else {"seizure_status": "not_run", "seizure_message": "Seizure screening has not been run.", "coverage": []}
        return frames

    def commit_analysis(self, study, hypnogram, spikes, seizures, analysis):
        run_id = uuid.uuid4().hex[:12]
        directory = self.directory(study) / "runs" / run_id
        for name, frame in [("hypnogram", hypnogram), ("spikes", spikes), ("seizures", seizures)]:
            write_csv(directory / f"{name}.csv", frame)
            write_csv(directory / f"{name}.original.csv", frame)
        analysis = {**analysis, "created_at": datetime.now(timezone.utc).isoformat(), "settings": asdict(study)}
        write_json(directory / "analysis.json", analysis)
        study.active_run = run_id
        study.settings_changed = False
        study.reviewed_spans = []
        self.save(study)

    def save_reviews(self, study, frames):
        if not study.active_run:
            raise ValueError("Run analysis or load saved results before reviewing.")
        directory = self.directory(study) / "runs" / study.active_run
        for name in ("hypnogram", "spikes", "seizures"):
            write_csv(directory / f"{name}.csv", frames[name])
        self.save(study)
