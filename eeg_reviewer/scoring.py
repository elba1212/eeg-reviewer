"""Coverage-aware one-second SWI; shared by the GUI, reports and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def interval_mask(intervals, length, fully_inside=False):
    mask = np.zeros(length, dtype=bool)
    for row in intervals:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if not np.isfinite([start, end]).all() or end <= start:
            continue
        a = int(np.ceil(start) if fully_inside else np.floor(start))
        b = int(np.floor(end) if fully_inside else np.ceil(end))
        mask[max(0, a):min(length, b)] = True
    return mask


def valid_events(study, frame, accepted_only=False):
    if frame.empty:
        return frame
    active = frame["status"].isin(["accepted"]) if accepted_only else ~frame["status"].isin(["rejected"])
    channels = frame["channel"].isin(study.picks + ["All"])
    return frame.loc[active & channels]


def build_bins(study, frames):
    length = int(np.floor(study.duration))
    stages = np.full(length, "UNKNOWN", dtype=object)
    for row in frames["hypnogram"].itertuples(index=False):
        a, b = max(0, int(np.floor(row.start_sec))), min(length, int(np.ceil(row.end_sec)))
        stages[a:b] = row.stage
    valid = np.isin(stages, ["W", "N1", "N2", "N3", "R"])
    coverage = frames["analysis"].get("coverage", [])
    valid &= interval_mask(coverage, length, fully_inside=True)
    valid &= ~interval_mask(study.bad_spans, length)
    spikes = valid_events(study, frames["spikes"])
    occupied = interval_mask(spikes.to_dict("records"), length)
    reviewed = interval_mask(study.reviewed_spans, length, fully_inside=True) & valid
    unresolved = spikes.loc[spikes["status"].isin(["unreviewed", "uncertain"])]
    reviewed &= ~interval_mask(unresolved.to_dict("records"), length)
    accepted = interval_mask(valid_events(study, frames["spikes"], accepted_only=True).to_dict("records"), length)
    return pd.DataFrame({"start_sec": np.arange(length), "stage": stages, "valid": valid,
                         "nrem": np.isin(stages, ["N2", "N3"]) & valid, "occupied": occupied,
                         "reviewed": reviewed, "accepted": accepted})


def fraction(positive, eligible):
    denominator = int(eligible.sum())
    numerator = int((positive & eligible).sum())
    return {"percent": 100.0 * numerator / denominator if denominator else None,
            "positive_seconds": numerator, "eligible_seconds": denominator}


def summarize(study, frames):
    bins = build_bins(study, frames)
    overall = fraction(bins.occupied, bins.nrem)
    reviewed = fraction(bins.accepted, bins.nrem & bins.reviewed)
    per_stage = {stage: fraction(bins.occupied, bins.valid & (bins.stage == stage)) for stage in ("N2", "N3")}
    segments = []
    for start in range(0, len(bins), 1800):
        part = bins.iloc[start:start + 1800]
        segments.append({"start_sec": start, "end_sec": min(start + 1800, len(bins)), **fraction(part.occupied, part.nrem)})
    durations = {stage: int(((bins.stage == stage) & bins.valid).sum()) for stage in ["W", "N1", "N2", "N3", "R"]}
    channels = []
    events = valid_events(study, frames["spikes"])
    for channel in study.picks:
        selected = events[events.channel.isin([channel, "All"])]
        occupied = interval_mask(selected.to_dict("records"), len(bins))
        values = fraction(occupied, bins.nrem)
        electrode = study.mapping.get(channel, channel)
        channels.append({"channel": channel, "electrode": electrode, "region": region(electrode), **values})
    seizures = frames["seizures"]
    active_seizures = seizures[seizures.status != "rejected"] if not seizures.empty else seizures
    usable_seizures = []
    for event in active_seizures.to_dict("records"):
        start, end = float(event["start_sec"]), float(event["end_sec"])
        if end > study.duration or start < 0 or any(start < span["end_sec"] and end > span["start_sec"] for span in study.bad_spans):
            continue
        event["stage"] = next((row.stage for row in frames["hypnogram"].itertuples(index=False) if row.start_sec <= start < row.end_sec), "UNKNOWN")
        event["duration_sec"] = end - start
        usable_seizures.append(event)
    return {"swi": overall, "reviewed_swi": reviewed, "per_stage": per_stage, "segments": segments,
            "durations": durations, "sleep_seconds": sum(durations[s] for s in ["N1", "N2", "N3", "R"]),
            "nrem_sleep_seconds": sum(durations[s] for s in ["N1", "N2", "N3"]),
            "valid_seconds": int(bins.valid.sum()), "unscored_seconds": study.duration - int(bins.valid.sum()),
            "channels": channels, "seizures": usable_seizures, "settings_changed": study.settings_changed}


def region(electrode):
    from .recordings import canonical_electrode
    electrode = canonical_electrode(electrode)
    if "-" in electrode:
        return "Bipolar derivation"
    if electrode.endswith("z"):
        side = "Midline"
    elif electrode[-1:].isdigit():
        side = "Left" if int(electrode[-1]) % 2 else "Right"
    else:
        return "Unmapped"
    area = "frontal" if electrode.startswith("F") else "temporal" if electrode.startswith("T") or electrode in {"P7", "P8"} else "central" if electrode.startswith("C") else "parietal" if electrode.startswith("P") else "occipital" if electrode.startswith("O") else "unmapped"
    return f"{side} {area}"
