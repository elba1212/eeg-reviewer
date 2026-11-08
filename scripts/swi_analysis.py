"""First-pass spike-wave candidate detection used by the desktop review app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def detect_spike_wave_candidates(
    raw,
    picks: list[str] | None = None,
    threshold_z: float = 6.0,
    refractory_sec: float = 0.25,
    min_abs_uv: float = 75.0,
    max_abs_uv: float = 1500.0,
) -> pd.DataFrame:
    """Find sharp, high-amplitude candidate peaks for expert visual review.

    This detector is not a seizure detector and does not diagnose epilepsy.
    """
    import mne

    work = raw.copy().load_data(verbose="ERROR").filter(1.0, 40.0, picks=picks, verbose="ERROR")
    data_uv = work.get_data(picks=picks) * 1e6
    names = [work.ch_names[i] for i in mne.pick_channels(work.ch_names, picks)] if picks else work.ch_names
    sfreq = float(work.info["sfreq"])
    refractory = max(1, int(round(refractory_sec * sfreq)))
    events = []

    for ch_idx, ch_name in enumerate(names):
        x = data_uv[ch_idx]
        finite = np.isfinite(x)
        if not finite.any():
            continue
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        robust_sigma = 1.4826 * mad if mad > 0 else np.nanstd(x)
        if robust_sigma <= 0 or not np.isfinite(robust_sigma):
            continue

        z = np.abs((x - med) / robust_sigma)
        height = max(threshold_z, min_abs_uv / robust_sigma)
        peaks, props = signal.find_peaks(z, height=height, distance=refractory)

        for peak, peak_height in zip(peaks, props["peak_heights"]):
            amplitude = float(x[peak])
            if abs(amplitude) > max_abs_uv:
                continue
            peak_sec = peak / sfreq
            events.append({
                "channel": ch_name,
                "peak_sec": peak_sec,
                "start_sec": max(0.0, peak_sec - 0.2),
                "end_sec": peak_sec + 0.8,
                "amplitude_uv": amplitude,
                "robust_z": float(peak_height),
            })

    columns = ["channel", "peak_sec", "start_sec", "end_sec", "amplitude_uv", "robust_z"]
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events, columns=columns).sort_values(
        ["peak_sec", "channel"]
    ).reset_index(drop=True)
