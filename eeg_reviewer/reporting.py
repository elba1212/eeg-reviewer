"""Deterministic, local draft reports and complete tabular exports."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import summarize
from .storage import write_json, write_csv

STAGE_Y = {"W": 4, "R": 3, "N1": 2, "N2": 1, "N3": 0, "UNKNOWN": np.nan}
POSITIONS = {"Fp1": (-.32,.92), "Fp2": (.32,.92), "Fpz": (0,1), "F7": (-.82,.55), "F3": (-.4,.5), "Fz": (0,.55), "F4": (.4,.5), "F8": (.82,.55),
             "T7": (-1,0), "C3": (-.5,0), "Cz": (0,0), "C4": (.5,0), "T8": (1,0), "P7": (-.82,-.55), "P3": (-.4,-.5), "Pz": (0,-.55),
             "P4": (.4,-.5), "P8": (.82,-.55), "O1": (-.32,-.92), "O2": (.32,-.92), "Oz": (0,-1)}


def pct(value):
    return "Not available" if value is None else f"{value:.2f}%"


def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def image_tag(figure, alt):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<p><img width="800" src="data:image/png;base64,{encoded}" alt="{escape(alt)}"></p>'


def figure():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9, 2.5), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def charts(study, frames, metrics):
    from .recordings import canonical_electrode
    hyp = frames["hypnogram"]
    fig = figure()
    ax = fig.subplots()
    for row in hyp.itertuples(index=False):
        ax.plot([row.start_sec / 60, min(study.duration, row.end_sec) / 60], [STAGE_Y.get(row.stage, np.nan)] * 2, color="#16877d", lw=1.6)
    ax.set(yticks=[0,1,2,3,4], yticklabels=["N3","N2","N1","REM","Wake"], xlabel="Minutes from recording start", title="Sleep-stage estimates / current review")
    ax.grid(alpha=.15)
    html = image_tag(fig, "Hypnogram")
    fig = figure()
    ax = fig.subplots()
    ax.bar([s["start_sec"] / 60 for s in metrics["segments"]], [s["percent"] if s["percent"] is not None else np.nan for s in metrics["segments"]],
           width=[(s["end_sec"] - s["start_sec"]) / 60 for s in metrics["segments"]], align="edge", color="#16877d", edgecolor="white")
    ax.set(xlabel="Minutes from recording start", ylabel="SWI (%)", ylim=(0,100), title="N2/N3 SWI estimate in 30-minute intervals")
    html += image_tag(fig, "SWI timeline")
    fig = figure()
    fig.set_size_inches(5,5)
    ax = fig.subplots()
    theta = np.linspace(0, 2 * np.pi, 150)
    ax.plot(np.cos(theta), np.sin(theta), color="#546171")
    ax.plot([-.12,0,.12],[1,1.12,1],color="#546171")
    for channel in metrics["channels"]:
        label = canonical_electrode(channel["electrode"])
        if label not in POSITIONS:
            continue
        x, y = POSITIONS[label]
        value = channel["percent"]
        ax.scatter([x],[y],s=230,c=[0 if value is None else value], cmap="YlOrRd",vmin=0,vmax=100,edgecolors="#546171")
        ax.text(x,y-.12,label,ha="center",fontsize=9)
    ax.set_title("Scalp SWI distribution (nose at top)")
    ax.set_aspect("equal")
    ax.axis("off")
    html += image_tag(fig, "Scalp electrode distribution; no intracranial source estimate")
    return html


def table(frame, max_rows=None):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if frame.empty:
        return "<p>No rows available.</p>"
    return frame.head(max_rows).to_html(index=False, escape=True, na_rep="Not available", float_format=lambda x: f"{x:.3f}") if max_rows else frame.to_html(index=False, escape=True, na_rep="Not available", float_format=lambda x: f"{x:.3f}")


def waveform_excerpts(study, frames, metrics):
    from .recordings import Recording
    recording = Recording(study)
    try:
        candidates = metrics["seizures"][:3]
        if len(candidates) < 3:
            spikes = frames["spikes"]
            if not spikes.empty:
                candidates += spikes[spikes.status != "rejected"].head(3 - len(candidates)).to_dict("records")
        output = ""
        for candidate in candidates:
            start = max(0, candidate["start_sec"] - 3)
            end = min(study.duration, max(start + 10, candidate["end_sec"] + 3), start + 30)
            picks = study.picks[:10]
            if not picks:
                continue
            values = recording.get_uv(picks, start, end)
            fig = figure()
            ax = fig.subplots()
            times = start + np.arange(values.shape[1]) / study.sfreq
            step = max(1, values.shape[1] // 4000)
            for index, (ch, values_ch) in enumerate(zip(picks, values)):
                ax.plot(times[::step], values_ch[::step] + index * 200, lw=.5, color="#253d56")
            ax.axvspan(candidate["start_sec"], candidate["end_sec"], color="#edb66a", alpha=.35)
            ax.set(yticks=np.arange(len(picks)) * 200, yticklabels=[study.mapping.get(ch,ch) for ch in picks], xlabel="Seconds from recording start", title=f"Candidate excerpt at {clock(candidate['start_sec'])} • spacing 200 µV")
            output += image_tag(fig, "Candidate EEG excerpt")
        return output
    finally:
        recording.close()


def report_html(study, frames, include_charts=True):
    metrics = summarize(study, frames)
    analysis = frames["analysis"]
    sw = metrics["swi"]
    reviewed = metrics["reviewed_swi"]
    seizures = metrics["seizures"]
    status = analysis.get("seizure_status", "not_run")
    coverage = sum(s["end_sec"] - s["start_sec"] for s in analysis.get("seizure_coverage", []))
    seizure_text = (f"{len(seizures)} suspected event(s) remain for review." if seizures else "No candidates detected in analyzed intervals.") if status == "completed" else "Seizure screening unavailable / not run. This is not a negative finding."
    from .pipeline import merge_intervals
    candidate_seconds = sum(e["end_sec"]-e["start_sec"] for e in merge_intervals(seizures))
    status_counts = {key:sum(e["status"] == key for e in seizures) for key in ["unreviewed","accepted","uncertain"]}
    stage_table = [{"Stage": "REM" if stage == "R" else "Wake" if stage == "W" else stage, "Duration": clock(seconds), "Seconds": seconds} for stage, seconds in metrics["durations"].items()]
    channels = sorted(metrics["channels"], key=lambda ch: ch["percent"] if ch["percent"] is not None else -1, reverse=True)
    maximum = channels[0] if channels and channels[0]["percent"] is not None else None
    distribution = f"Candidate SWI is maximal at {escape(maximum['electrode'])} ({escape(maximum['region'])}), {pct(maximum['percent'])}." if maximum else "Electrode distribution is unavailable."
    age = "Not supplied" if study.age is None else str(study.age)
    stale = "<p class='notice'>Channel settings changed after this analysis. Reanalysis is required; values below describe the saved run.</p>" if study.settings_changed else ""
    content = f"""<!doctype html><html><head><meta charset="utf-8"><title>EEG sleep and epileptic activity draft report</title>
    <style>body{{font-family:Segoe UI,Arial,sans-serif;color:#20364a;max-width:1000px;margin:32px auto;font-size:11pt}} h1{{font-size:24pt}} h2{{color:#147d75;margin-top:28px}} table{{border-collapse:collapse;width:100%;font-size:9pt;margin:12px 0}}td,th{{border:1px solid #dce3e8;padding:7px;text-align:left}}th{{background:#eef5f5}}img{{max-width:100%;height:auto}}.notice{{padding:14px;background:#fff1d8}}.muted{{color:#5d6b77}} .metric{{font-size:16pt}}</style></head><body>
    <p class="muted">EEG REVIEWER · LOCAL ANALYSIS · DRAFT</p><h1>Sleep and epileptic activity</h1>
    <p class="notice">Personal review draft. Automatic stages and event flags require expert review. Accepted annotations record a user's review decision and do not establish a clinical diagnosis.</p>{stale}
    <h2>Recording and methods</h2><p>Study: {escape(study.name)}<br>Acquisition start: {escape(study.first_timestamp or 'Unknown')}<br>Age at recording: {escape(age)} years<br>Duration: {clock(study.duration)} · Sample rate: {study.sfreq:g} Hz<br>Reference: {escape(study.reference)}<br>Staging EEG: {escape(study.eeg_channel)} ({escape(study.mapping.get(study.eeg_channel,''))})<br>EOG: {escape(study.eog_channel or 'Not used')} · EMG: {escape(study.emg_channel or 'Not used')}<br>Excluded channels: {escape(', '.join(study.excluded) or 'None')}</p>
    <p>Spike-analysis electrodes: {escape(', '.join(study.mapping.get(ch,ch) for ch in study.picks))}. Valid staged and spike-analyzed time: {clock(metrics['valid_seconds'])}. Unscored or excluded time: {clock(metrics['unscored_seconds'])}. Sleep staging is provisional, particularly for naps and pediatric epilepsy. REM and NREM summaries are not a diagnosis.</p>
    <h2>Sleep architecture</h2><p>Total scored sleep: {clock(metrics['sleep_seconds'])}. NREM (N1 + N2 + N3): {clock(metrics['nrem_sleep_seconds'])}.</p>{table(stage_table)}
    <h2>Spike-wave index</h2><p class="metric">Working N2/N3 SWI estimate: {pct(sw['percent'])}</p><p>{sw['positive_seconds']} positive one-second bins / {sw['eligible_seconds']} eligible N2/N3 one-second bins. N1, REM, Wake, unknown stages, gaps, artifacts and unanalyzed time are excluded. Simultaneous electrode detections count once. Rejected events are excluded; unreviewed candidates remain in this estimate.</p>
    <p>Reviewed SWI: {pct(reviewed['percent'])} ({reviewed['positive_seconds']} / {reviewed['eligible_seconds']} seconds explicitly reviewed, excluding unresolved candidates). This reviewed interval score is not a whole-recording score unless coverage is complete.</p>
    {table([{'Stage': k, **v} for k,v in metrics['per_stage'].items()])}{table(metrics['segments'])}
    <h2>Possible seizures</h2><p>{seizure_text} Screening status: {escape(status)}. Analyzed seizure-screening time: {clock(coverage)}.</p><p>{escape(analysis.get('seizure_message',''))}</p>
    <p>Total candidate-covered duration: {candidate_seconds:.2f} seconds. Unreviewed: {status_counts['unreviewed']}; user-accepted: {status_counts['accepted']}; uncertain: {status_counts['uncertain']}. These are candidate counts and durations, not clinically confirmed seizure burden.</p>
    <p>{escape(analysis.get('model_limitations','Pediatric clinical seizure-detection accuracy has not been established for this application.'))}</p>
    {table([{k:e.get(k,'') for k in ['start_sec','end_sec','duration_sec','stage','score','status','origin']} for e in seizures],100)}
    <p class="muted">All event rows, including rejected annotations, are retained in the CSV export. Brief flags remain candidates and are not automatically classified as electrographic seizures.</p>
    <h2>Scalp distribution</h2><p>{distribution} This describes the recorded scalp field and depends on electrode coverage, reference, artifacts, and review status. It does not estimate the intracranial source or establish a seizure-onset zone.</p>{table(channels)}
    <h2>Review notes</h2><p>{escape(study.narrative or 'No additional review notes.').replace(chr(10),'<br>')}</p>
    <h2>Quality and excluded intervals</h2>{table(study.bad_spans)}
    """
    if include_charts:
        content += "<h2>Figures</h2>" + charts(study, frames, metrics) + "<h2>Representative candidate excerpts</h2>" + waveform_excerpts(study, frames, metrics)
    content += f"<h2>Analysis provenance</h2><p>{escape(analysis.get('source','Unknown'))}. Run: {escape(study.active_run)}. Created: {escape(analysis.get('created_at','Unknown'))}.</p><pre>{escape(str(analysis.get('versions',{})))}</pre><p>Seizure model: {escape(analysis.get('model',{}).get('model','Not run'))}. Model checksum: {escape(analysis.get('model',{}).get('sha256','Not available'))}.</p><p class='muted'>Report generated {datetime.now(timezone.utc).isoformat()} · EEG Reviewer · Preliminary personal review</p></body></html>"
    return content


def export_report(store, study, destination):
    if not study.active_run:
        raise ValueError("No analysis is available to export.")
    if study.settings_changed:
        raise ValueError("Run analysis again after changing channel settings before exporting results.")
    frames = store.load_frames(study)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "EEG_report.html"
    report.write_text(report_html(study, frames), encoding="utf-8")
    for name in ["hypnogram", "spikes", "seizures"]:
        write_csv(output / f"{name}.csv", frames[name])
    metrics = summarize(study, frames)
    write_csv(output / "swi_30min.csv", pd.DataFrame(metrics["segments"]))
    write_csv(output / "electrode_distribution.csv", pd.DataFrame(metrics["channels"]))
    write_json(output / "summary.json", metrics)
    write_json(output / "analysis_provenance.json", frames["analysis"])
    return report
