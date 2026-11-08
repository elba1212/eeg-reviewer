"""Convert an XLTEK text export into Python-friendly NumPy files.

The XLTEK text export is very large, so this converter streams the file and
writes a memory-mapped ``.npy`` array instead of loading everything into RAM.
The output array is shaped ``(n_channels, n_samples)`` and stores volts, which
matches MNE's convention.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


DEFAULT_SOURCE = Path("input.txt")
DEFAULT_OUTPUT_DIR = Path("converted")
DATA_LINE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\t")
CHANNEL_RE = re.compile(r"^C\d{3}$", re.IGNORECASE)


@dataclass
class XltekTextMetadata:
    source_file: str
    data_file: str
    sampling_rate_hz: float
    source_units: str
    stored_units: str
    scale_to_stored_units: float
    storage_format: str
    missing_value: int | float | None
    missing_samples: int
    n_channels: int
    n_samples: int
    channel_names: list[str]
    first_timestamp: str | None
    last_timestamp: str | None
    first_stamp: int | None
    last_stamp: int | None
    source_start_timestamp: str | None = None
    requested_start_timestamp: str | None = None
    requested_end_timestamp: str | None = None
    start_sample_index: int = 0
    end_sample_index: int | None = None


def _detect_encoding(source_file: Path) -> str:
    with source_file.open("rb") as handle:
        prefix = handle.read(4)
    if prefix.startswith(b"\xff\xfe"):
        return "utf-16"
    if prefix.startswith(b"\xfe\xff"):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _parse_header(source_file: Path) -> tuple[dict[str, str], list[str], list[int], int]:
    header: dict[str, str] = {}
    column_names: list[str] = []
    data_start_line = 0
    encoding = _detect_encoding(source_file)

    with source_file.open("r", encoding=encoding, errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if DATA_LINE_RE.match(line):
                data_start_line = line_number
                break

            stripped = line.strip()
            if not stripped.startswith("%"):
                continue

            content = stripped.lstrip("%").strip()
            if content.startswith("Date.Time"):
                column_names = [part.strip() for part in content.split("\t") if part.strip()]
                continue

            if ":" in content:
                key, value = content.split(":", 1)
                header[key.strip()] = value.strip()
            elif "Sampling Rate" in content:
                header["Sampling Rate"] = content.split("Sampling Rate", 1)[1].strip()
            elif content.startswith("Channels"):
                header["Channels"] = content.split("Channels", 1)[1].strip()
            elif content.startswith("Units"):
                header["Units"] = content.split("Units", 1)[1].strip()

    if not column_names:
        raise ValueError("Could not find the XLTEK text column header.")
    if data_start_line == 0:
        raise ValueError("Could not find the first data line.")

    channel_indices = [i for i, name in enumerate(column_names) if CHANNEL_RE.match(name)]
    channel_names = [column_names[i] for i in channel_indices]
    return header, channel_names, channel_indices, data_start_line


def _sampling_rate(header: dict[str, str]) -> float:
    text = header.get("Sampling Rate", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Could not parse sampling rate from {text!r}")
    return float(match.group(0))


def _source_units(header: dict[str, str]) -> str:
    return header.get("Units", "unknown").strip() or "unknown"


def _scale_factor(source_units: str, target_units: str) -> float:
    normalized = source_units.lower().replace("µ", "u")
    target = target_units.lower().replace("µ", "u")
    to_volts = {
        "v": 1.0,
        "mv": 1e-3,
        "uv": 1e-6,
        "microv": 1e-6,
        "microvolt": 1e-6,
        "microvolts": 1e-6,
    }
    if normalized not in to_volts:
        raise ValueError(f"Unknown source unit {source_units!r}; add the scale explicitly.")
    if target not in to_volts:
        raise ValueError(f"Unknown target unit {target_units!r}; add the scale explicitly.")
    return to_volts[normalized] / to_volts[target]


def _storage_params(storage_format: str, source_units: str) -> tuple[np.dtype, str, float]:
    if storage_format == "float32_volts":
        return np.dtype("float32"), "V", _scale_factor(source_units, "V")
    if storage_format == "int16_microvolts":
        return np.dtype("int16"), "uV", _scale_factor(source_units, "uV")
    raise ValueError(f"Unsupported storage format {storage_format!r}")


def _convert_values(values: np.ndarray, scale: float, dtype: np.dtype) -> np.ndarray:
    scaled = values * scale
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        missing = info.min
        scaled = np.nan_to_num(scaled, nan=missing, posinf=info.max, neginf=info.min + 1)
        scaled = np.clip(np.rint(scaled), info.min + 1, info.max)
        scaled[np.isnan(values)] = missing
    return scaled.astype(dtype, copy=False)


def _parse_channel_values(parts: list[str], channel_indices: list[int]) -> tuple[np.ndarray, int]:
    values = np.empty(len(channel_indices), dtype="float32")
    missing = 0
    for output_idx, part_idx in enumerate(channel_indices):
        try:
            values[output_idx] = float(parts[part_idx])
        except ValueError:
            values[output_idx] = np.nan
            missing += 1
    return values, missing


def _scale_to_volts(source_units: str) -> float:
    return _scale_factor(source_units, "V")


def count_data_lines(source_file: Path) -> int:
    count = 0
    encoding = _detect_encoding(source_file)
    with source_file.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            if DATA_LINE_RE.match(line):
                count += 1
    return count


def _timestamp_from_parts(date_time: str) -> str:
    parsed = datetime.strptime(date_time, "%m/%d/%Y %H:%M:%S")
    return parsed.isoformat(sep=" ")


def _first_data_timestamp_and_stamp(source_file: Path) -> tuple[datetime, int]:
    encoding = _detect_encoding(source_file)
    with source_file.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            if not DATA_LINE_RE.match(line):
                continue
            parts = line.rstrip("\n").split("\t")
            return datetime.strptime(parts[0], "%m/%d/%Y %H:%M:%S"), int(parts[2])
    raise ValueError("Could not find a data line.")


def _parse_window_time(text: str | None) -> datetime | None:
    if text is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(
        f"Could not parse time {text!r}. Use e.g. '2026-05-14 21:00:00'."
    )


def _sample_window(
    source_file: Path,
    sampling_rate: float,
    start_time: str | None,
    end_time: str | None,
    max_samples: int | None,
) -> tuple[datetime, int, int | None]:
    source_start, _ = _first_data_timestamp_and_stamp(source_file)
    requested_start = _parse_window_time(start_time)
    requested_end = _parse_window_time(end_time)

    start_sample = 0
    if requested_start is not None:
        start_sample = max(0, round((requested_start - source_start).total_seconds() * sampling_rate))

    end_sample: int | None = None
    if requested_end is not None:
        end_sample = max(start_sample, round((requested_end - source_start).total_seconds() * sampling_rate))
    if max_samples is not None:
        max_end = start_sample + max_samples
        end_sample = min(end_sample, max_end) if end_sample is not None else max_end

    return source_start, start_sample, end_sample


def convert_txt_export(
    source_file: Path,
    output_dir: Path,
    max_samples: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    output_stem: str | None = None,
    storage_format: str = "float32_volts",
) -> XltekTextMetadata:
    header, channel_names, channel_indices, _ = _parse_header(source_file)
    sampling_rate = _sampling_rate(header)
    source_units = _source_units(header)
    dtype, stored_units, scale = _storage_params(storage_format, source_units)

    source_start, start_sample, end_sample = _sample_window(
        source_file, sampling_rate, start_time, end_time, max_samples
    )

    if end_sample is not None:
        n_samples = end_sample - start_sample
    elif start_sample == 0 and max_samples is None:
        n_samples = count_data_lines(source_file)
    else:
        total_samples = count_data_lines(source_file)
        n_samples = max(0, total_samples - start_sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or source_file.stem
    unit_label = "microvolts" if stored_units == "uV" else "volts"
    data_file = output_dir / f"{stem}_signals_{unit_label}.npy"
    array = open_memmap(
        data_file,
        mode="w+",
        dtype=dtype,
        shape=(len(channel_names), n_samples),
    )

    first_timestamp: str | None = None
    last_timestamp: str | None = None
    first_stamp: int | None = None
    last_stamp: int | None = None
    missing_samples = 0
    input_sample_idx = 0
    output_sample_idx = 0
    encoding = _detect_encoding(source_file)

    with source_file.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            if not DATA_LINE_RE.match(line):
                continue
            if input_sample_idx < start_sample:
                input_sample_idx += 1
                continue
            if end_sample is not None and input_sample_idx >= end_sample:
                break

            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(channel_indices):
                raise ValueError(
                    f"Malformed data row at input sample {input_sample_idx}: {line[:120]!r}"
                )

            if output_sample_idx == 0:
                first_timestamp = _timestamp_from_parts(parts[0])
                first_stamp = int(parts[2])

            values, row_missing = _parse_channel_values(parts, channel_indices)
            missing_samples += row_missing
            array[:, output_sample_idx] = _convert_values(values, scale, dtype)
            last_timestamp = _timestamp_from_parts(parts[0])
            last_stamp = int(parts[2])

            output_sample_idx += 1
            input_sample_idx += 1
            if output_sample_idx >= n_samples:
                break

    array.flush()
    metadata = XltekTextMetadata(
        source_file=str(source_file).replace("\\", "/"),
        data_file=str(data_file).replace("\\", "/"),
        sampling_rate_hz=sampling_rate,
        source_units=source_units,
        stored_units=stored_units,
        scale_to_stored_units=scale,
        storage_format=storage_format,
        missing_value=int(np.iinfo(dtype).min) if np.issubdtype(dtype, np.integer) else None,
        missing_samples=missing_samples,
        n_channels=len(channel_names),
        n_samples=output_sample_idx,
        channel_names=channel_names,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        first_stamp=first_stamp,
        last_stamp=last_stamp,
        source_start_timestamp=source_start.isoformat(sep=" "),
        requested_start_timestamp=start_time,
        requested_end_timestamp=end_time,
        start_sample_index=start_sample,
        end_sample_index=end_sample,
    )

    metadata_file = output_dir / f"{stem}_metadata.json"
    metadata_file.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Convert only the first N samples. Useful for parser testing.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional window start, e.g. '2026-05-14 21:00:00'.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional window end, e.g. '2026-05-15 06:00:00'.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem.",
    )
    parser.add_argument(
        "--storage-format",
        choices=["float32_volts", "int16_microvolts"],
        default="float32_volts",
        help="Storage format for the output array.",
    )
    args = parser.parse_args()

    metadata = convert_txt_export(
        args.source_file,
        args.output_dir,
        args.max_samples,
        args.start_time,
        args.end_time,
        args.output_stem,
        args.storage_format,
    )
    print(f"Converted samples: {metadata.n_samples}")
    print(f"Channels: {metadata.n_channels}")
    print(f"Sampling rate: {metadata.sampling_rate_hz} Hz")
    print(f"Wrote: {metadata.data_file}")


if __name__ == "__main__":
    main()
