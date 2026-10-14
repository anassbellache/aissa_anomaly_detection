"""Data loading and I/O for the ESS SDS cavity waveform dumps under ``data/``.

Flow::

    many parquet files -> metadata-only catalogue -> select cavity / campaign / event / cycles
                       -> load only the selected waveforms -> pair amplitude and phase

Usage::

    import sds_io as sds

    cat = sds.build_catalogue("data")            # metadata only, ~0.2 s
    cat.events                                   # one row per event
    cat.cycles                                   # one row per (event_id, cycle_id)
    cat.issues                                   # data oddities found while building

    dev = "HBL-010RFC:RFS-DIG-101"
    paired = sds.load_paired(cat.select(device=dev))      # every event holding that cavity
    cavity = paired.loc[dev, 15643300300]                 # index (cycle_id, signal)
    cycle = cavity.loc[cavity.index.unique("cycle_id")[0]]
    probe = sds.to_complex(cycle.loc["probe", "amplitude"], cycle.loc["probe", "phase_deg"])
    t_us = sds.time_axis(len(probe))

    for n, block in paired.xs("probe", level="signal").groupby("n_samples"):
        X = sds.stack(block["amplitude"])                 # (rows, n) float32, one block per length

Data model
----------
One parquet file holds the buffer of one system (HBL, MBL, SPOKES, ...) for one *event* (a PM or DD
trigger). One row is one waveform of one PV for one machine *cycle* (14 Hz pulse counter).

Three levels of time, kept apart:

* event  -- ``event_id``, ``event_ts``, ``ctype``, ``campaign``: constant per file.
* cycle  -- ``cycle_ts`` and the beam fields, keyed by ``(event_id, cycle_id)``.
* row    -- ``trig_ts``, ``data_ts`` and the trigger settings, which differ per system.

Rules this module follows:

* Timestamps stay ``datetime64[ns, UTC]`` and are never rounded or made naive. The 2025 files lost
  precision upstream, so **joins and keys use the integer ids only**, never timestamps.
* ``cycle_offset = cycle_id - event_id`` is the exact relative-time axis in cycles (``event_id`` is
  the cycle in which the event fired); ``t_rel_s`` is the same thing measured from the timestamps.
* ``cycle_id`` is never converted to absolute time: the counter origin drifts between campaigns.
* Campaign directory names are labels only (UTC for 2025, CEST for 2026).
* ``Cmp0`` is amplitude and ``Cmp1`` is phase in degrees

Every file is a single row group, so no engine can skip rows; waveforms are streamed in small
batches and only the wanted rows are kept, which bounds memory to roughly the size of the result.
"""

from __future__ import annotations

import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RF_SYSTEMS = ("HBL", "MBL", "SPOKES")
CYCLE_PERIOD_NS = 71_428_568  # 14 Hz machine cycle
SIGNALS = {"0": "probe", "5": "forward", "6": "reflected"}
COMPONENTS = {"0": "amplitude", "1": "phase_deg"}
# Sample spacing from PV_naming.xlsx (two significant figures; unverified for the 5032-sample files).
NOMINAL_DT_US = {"rf": 0.12}
LOCAL_TZ = "Europe/Stockholm"  # Event_reports.xlsx timestamps are naive local time

WAVEFORM_COL = "data_array"
_LIST1_COLS = {
    "array_dimensions.binning": "binning",
    "array_dimensions.offset": "offset",
    "array_dimensions.reverse": "reverse",
}
# Column order differs between files, so everything is read by name.
META_COLS = [
    "pv", "array_size", *_LIST1_COLS,
    "ctype", "event_id", "cycle_id", "event_ts", "cycle_ts", "trig_ts", "data_ts",
    "trig_name", "trig_evr", "trig_code", "trig_user_tag", "trig_delay", "collector",
    "beam_mode", "beam_state", "beam_dest", "beam_present", "beam_len", "beam_current",
    "beam_energy", "target_sector", "target_raster",
]
PAIR_KEY = ["device", "event_id", "cycle_id", "signal"]

_PATH_RE = re.compile(
    r"(?:(?P<archive>[^/]+)/SDS_Data/)?(?P<campaign>\d{8}_\d{4}_\d{8}_\d{4})/(?P<ctype>DD|PM)/(?P<event_id>\d+)/(?P<system>[A-Za-z]+)\.parquet$"
)
_RF_PV_RE = re.compile(
    r"^(?P<section>[A-Za-z]+)-(?P<module>\d+)RFC:RFS-DIG-(?P<cavity>\d)01"
    r":Dwn(?P<dwn>[056])-Cmp(?P<cmp>[01])SDS-(?:PM|DOD)Data$"
)
_OTHER_PV_RE = re.compile(r"^(?P<section>[A-Za-z]+)-(?P<module>\d+)[A-Za-z]*:")
_SDS_SUFFIX_RE = re.compile(r"SDS-(?:PM|DOD)Data$")
_TRIP_RE = re.compile(
    r"→\s*(?P<mps_input>.+?)\s+=\s+(?P<state>\S+)\s+\[(?P<condition>[^\]]*)\]"
    r"\s+@\s*[\d:.]+\s+\((?P<offset_s>[+-]?[\d.]+) s\)"
)


class CatalogueError(ValueError):
    """A file breaks an assumption the catalogue relies on."""


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PVName:
    """A parsed PV name. Cavity PVs have ``kind == "rf"``; anything else is ``"other"``."""

    raw: str
    kind: str
    device: str
    signal: str
    component: str | None = None
    section: str | None = None
    module: int | None = None
    cavity: int | None = None


@lru_cache(maxsize=None)
def parse_pv(pv: str) -> PVName:
    """Parse ``HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData`` and friends.

    The ``SDS-PMData`` / ``SDS-DODData`` suffix is ignored: it does not track the event class
    (2026 DD files use ``PMData``); use the ``ctype`` column for that.
    """
    device, _, tail = pv.rpartition(":")
    m = _RF_PV_RE.match(pv)
    if m:
        return PVName(
            raw=pv, kind="rf", device=device,
            signal=SIGNALS[m["dwn"]], component=COMPONENTS[m["cmp"]],
            section=m["section"].upper(), module=int(m["module"]), cavity=int(m["cavity"]),
        )
    m = _OTHER_PV_RE.match(pv)
    return PVName(
        raw=pv, kind="other", device=device or pv, signal=_SDS_SUFFIX_RE.sub("", tail),
        section=m["section"].upper() if m else None, module=int(m["module"]) if m else None,
    )


@dataclass(frozen=True)
class SourceFile:
    """One parquet file: the buffer of one system for one event."""

    file_id: int
    path: Path
    archive_dir: str
    campaign: str
    ctype: str
    event_id: int
    system: str
    size_bytes: int


@dataclass
class Catalogue:
    """Metadata for every waveform row (no waveform data), plus the files it came from."""

    frame: pd.DataFrame
    files: tuple[SourceFile, ...]
    issues: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        f = self.frame
        return (
            f"Catalogue({len(f)} rows, {f['file_id'].nunique()} files, "
            f"{f['event_id'].nunique()} events, {f['device'].nunique()} devices)"
        )

    def select(self, **criteria: Any) -> Catalogue:
        """Filter on any catalogue column: a scalar means ``==``, an iterable means ``isin``.

        ``cat.select(device="HBL-010RFC:RFS-DIG-101", ctype="DD", signal=["probe", "forward"])``
        """
        mask = np.ones(len(self.frame), dtype=bool)
        for column, value in criteria.items():
            if column not in self.frame.columns:
                raise KeyError(f"{column!r} is not a catalogue column; choose from {list(self.frame.columns)}")
            if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                dtype = self.frame[column].dtype
                if isinstance(dtype, pd.DatetimeTZDtype):
                    # A naive value (e.g. from .to_numpy()) would compare unequal to everything.
                    value = pd.Timestamp(value)
                    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
                elif pd.api.types.is_integer_dtype(dtype) and isinstance(value, (str, bytes)):
                    value = int(value)
                mask &= (self.frame[column] == value).to_numpy(dtype=bool, na_value=False)
            else:
                mask &= self.frame[column].isin(list(value)).to_numpy(dtype=bool)
        return Catalogue(self.frame[mask].reset_index(drop=True), self.files, self.issues)

    def estimate_bytes(self) -> int:
        """Size of the selected waveforms once loaded (float32)."""
        return int(self.frame["n_samples"].sum()) * 4

    @property
    def events(self) -> pd.DataFrame:
        """One row per event."""
        f = self.frame
        g = f.groupby("event_id", sort=False)
        out = g.agg(
            campaign=("campaign", "first"),
            ctype=("ctype", "first"),
            event_ts=("event_ts", "first"),
            n_rows=("pv", "size"),
            n_devices=("device", "nunique"),
            n_cycles=("cycle_id", "nunique"),
            first_cycle_id=("cycle_id", "min"),
            last_cycle_id=("cycle_id", "max"),
            first_cycle_ts=("cycle_ts", "min"),
            last_cycle_ts=("cycle_ts", "max"),
        )
        out["systems"] = g["system"].agg(lambda s: ",".join(sorted(s.unique())))
        out["cycle_stride"] = g["cycle_id"].agg(_stride)
        return out.sort_values("event_ts")

    @property
    def cycles(self) -> pd.DataFrame:
        """One row per ``(event_id, cycle_id)``: cycle time and beam state."""
        f = self.frame
        g = f.groupby(["event_id", "cycle_id"], sort=True)
        out = g[
            ["cycle_ts", "cycle_offset", "t_rel_s", "beam_mode", "beam_state",
             "beam_present", "beam_len", "beam_current"]
        ].first()
        out["systems"] = g["system"].agg(lambda s: ",".join(sorted(s.unique())))
        out["n_rows"] = g.size()
        return out


def _stride(cycle_ids: pd.Series) -> int:
    """Most common step between consecutive stored cycles (0 if there is a single cycle)."""
    steps = np.diff(np.sort(cycle_ids.unique()))
    if len(steps) == 0:
        return 0
    values, counts = np.unique(steps, return_counts=True)
    return int(values[counts.argmax()])


# --------------------------------------------------------------------------------------------------
# Discovery and catalogue
# --------------------------------------------------------------------------------------------------

def discover_files(root: str | Path = "data", systems: tuple[str, ...] | None = RF_SYSTEMS) -> list[SourceFile]:
    """Find the SDS parquet files under ``root``, dropping duplicate extractions.

    Files that do not sit at ``.../<campaign>/<DD|PM>/<event_id>/<SYSTEM>.parquet`` (such as the
    sample file) are ignored. ``systems=None`` keeps every system.
    """
    root = Path(root).resolve()  # absolute paths, so the catalogue survives a later chdir
    found: dict[tuple, list[tuple[str, Path]]] = {}
    for path in sorted(root.rglob("*.parquet")):
        m = _PATH_RE.search(path.as_posix())
        if not m or (systems is not None and m["system"] not in systems):
            continue
        key = (m["campaign"], m["ctype"], int(m["event_id"]), m["system"])
        found.setdefault(key, []).append((m["archive"] or "", path))

    files = []
    for key in sorted(found):
        # A directory still carrying ".tar" in its name is a second extraction of the same archive.
        candidates = sorted(found[key], key=lambda c: (".tar" in c[0], c[0]))
        (archive_dir, path), dropped = candidates[0], candidates[1:]
        if dropped:
            warnings.warn(
                f"{key[3]}.parquet of {key[1]} event {key[2]} found {len(candidates)} times; "
                f"using {archive_dir}, ignoring {[d for d, _ in dropped]}",
                stacklevel=2,
            )
        campaign, ctype, event_id, system = key
        files.append(SourceFile(len(files), path, archive_dir, campaign, ctype, event_id, system,
                                path.stat().st_size))
    return files


def _read_meta(src: SourceFile) -> pd.DataFrame:
    """Read every non-waveform column of one file and check the per-file invariants."""
    with pq.ParquetFile(src.path) as pf:
        missing = [c for c in [*META_COLS, WAVEFORM_COL] if c not in pf.schema_arrow.names]
        if missing:
            raise CatalogueError(f"{src.path}: missing columns {missing}")
        if pf.metadata.num_row_groups != 1:
            # row_in_file is a position within the file; the streaming reader relies on file order.
            raise CatalogueError(f"{src.path}: expected 1 row group, found {pf.metadata.num_row_groups}")
        row_group = pf.metadata.row_group(0)
        n_values = sum(
            row_group.column(i).num_values
            for i in range(row_group.num_columns)
            if row_group.column(i).path_in_schema.split(".")[0] == WAVEFORM_COL
        )
        table = pf.read(columns=META_COLS, use_threads=False).select(META_COLS)

    for name in _LIST1_COLS:
        i = table.schema.get_field_index(name)
        table = table.set_column(i, name, pc.list_element(table.column(name), 0))
    df = table.to_pandas().rename(columns={**_LIST1_COLS, "array_size": "n_samples"})
    df["event_id"] = df["event_id"].astype("int64")  # uint64 mixed with int64 silently becomes float64
    df["cycle_id"] = df["cycle_id"].astype("int64")
    df["n_samples"] = df["n_samples"].astype("int64")

    # The footer counts every stored sample plus one entry per empty list, so this tells for free
    # whether array_size can be trusted as the waveform length.
    expected = int(df["n_samples"].sum() + (df["n_samples"] == 0).sum())
    if expected != n_values:
        raise CatalogueError(
            f"{src.path}: array_size does not match the stored waveform lengths "
            f"({expected} values declared, {n_values} stored)"
        )
    if df["event_id"].nunique() != 1 or df["event_id"].iloc[0] != src.event_id:
        raise CatalogueError(f"{src.path}: event_id column {df['event_id'].unique()} != path id {src.event_id}")
    if df["ctype"].nunique() != 1 or df["ctype"].iloc[0] != src.ctype:
        raise CatalogueError(f"{src.path}: ctype column {df['ctype'].unique()} != path ctype {src.ctype}")
    if df["event_ts"].nunique() != 1:
        raise CatalogueError(f"{src.path}: event_ts is not constant")
    if df.duplicated(["pv", "cycle_id"]).any():
        raise CatalogueError(f"{src.path}: duplicate (pv, cycle_id) rows")
    pairs = df[["cycle_id", "cycle_ts"]].drop_duplicates()
    if pairs["cycle_id"].duplicated().any() or pairs["cycle_ts"].duplicated().any():
        raise CatalogueError(f"{src.path}: cycle_id and cycle_ts are not one-to-one")

    df.insert(0, "file_id", src.file_id)
    df.insert(1, "row_in_file", np.arange(len(df), dtype="int64"))
    df["system"] = src.system
    df["campaign"] = src.campaign
    return df


def build_catalogue(root: str | Path = "data", systems: tuple[str, ...] | None = RF_SYSTEMS,
                    strict: bool = True) -> Catalogue:
    """Crawl ``root`` and build the metadata-only catalogue (one row per PV x cycle x event).

    With ``strict=False`` a file that breaks an invariant is recorded in ``issues`` and skipped
    instead of raising.
    """
    files = discover_files(root, systems)
    if not files:
        raise FileNotFoundError(f"no SDS parquet files for systems={systems} under {Path(root).resolve()}")

    issues: list[dict] = []
    frames = []
    for src in files:
        try:
            frames.append(_read_meta(src))
        except (CatalogueError, pa.ArrowException, OSError) as err:
            if not isinstance(err, CatalogueError):
                err = CatalogueError(f"{src.path}: unreadable parquet file ({err})")
            if strict:
                raise err
            issues.append({"file_id": src.file_id, "check": "file_skipped", "n_rows": 0, "detail": str(err)})
    if not frames:
        raise CatalogueError("every file was skipped; see the recorded reasons: "
                             + "; ".join(i["detail"] for i in issues))
    df = pd.concat(frames, ignore_index=True)

    per_event = df.groupby("event_id")[["campaign", "ctype", "event_ts"]].nunique()
    if (per_event > 1).any(axis=None):
        bad = per_event.index[(per_event > 1).any(axis=1)].tolist()
        raise CatalogueError(f"event_id maps to several campaigns, ctypes or event_ts: {bad}")

    parsed = pd.DataFrame([asdict(parse_pv(pv)) for pv in df["pv"].unique()]).rename(columns={"raw": "pv"})
    parsed[["module", "cavity"]] = parsed[["module", "cavity"]].astype("Int64")
    df = df.merge(parsed, on="pv", how="left", validate="many_to_one")

    df["cycle_offset"] = df["cycle_id"] - df["event_id"]
    df["t_rel_s"] = (df["cycle_ts"] - df["event_ts"]).dt.total_seconds()
    df["trig_lag_s"] = (df["trig_ts"] - df["cycle_ts"]).dt.total_seconds()
    df["trig_ts_ok"] = (df["trig_lag_s"] >= 0) & (df["trig_lag_s"] < CYCLE_PERIOD_NS / 1e9)
    df["is_empty"] = df["n_samples"] == 0

    columns = [
        "event_id", "cycle_id", "pv",
        "file_id", "row_in_file", "system", "campaign", "ctype",
        "section", "module", "device", "cavity", "kind", "signal", "component",
        "event_ts", "cycle_ts", "trig_ts", "data_ts", "cycle_offset", "t_rel_s", "trig_lag_s",
        "n_samples", "binning", "offset", "reverse",
        "trig_delay", "trig_code", "trig_user_tag", "trig_name", "trig_evr", "collector",
        "beam_mode", "beam_state", "beam_dest", "beam_present", "beam_len", "beam_current",
        "beam_energy", "target_sector", "target_raster",
        "is_empty", "trig_ts_ok",
    ]
    df = df[columns].sort_values(["event_id", "system", "pv", "cycle_id"], ignore_index=True)
    issues += _find_issues(df)
    return Catalogue(df, tuple(files), pd.DataFrame(issues, columns=["file_id", "check", "n_rows", "detail"]))


def _find_issues(df: pd.DataFrame) -> list[dict]:
    """Oddities worth knowing about that do not stop the catalogue from being used."""
    issues = []

    def add(file_id, check, n_rows, detail):
        issues.append({"file_id": file_id, "check": check, "n_rows": int(n_rows), "detail": detail})

    for file_id, f in df.groupby("file_id"):
        if f["is_empty"].any():
            add(file_id, "empty_waveform", f["is_empty"].sum(),
                f"cycles {sorted(f.loc[f['is_empty'], 'cycle_id'].unique().tolist())}")
        lengths = sorted(f.loc[~f["is_empty"], "n_samples"].unique().tolist())
        if len(lengths) > 1:
            add(file_id, "mixed_lengths", len(f), f"n_samples {lengths}")
        if not f["trig_ts_ok"].all():
            bad = f[~f["trig_ts_ok"]]
            add(file_id, "trig_ts_suspect", len(bad),
                f"trig_ts outside [cycle_ts, cycle_ts + 1 cycle) for cycles {sorted(bad['cycle_id'].unique().tolist())}")
        per_pv = f.groupby("pv")["cycle_id"].nunique()
        short = per_pv[per_pv < f["cycle_id"].nunique()]
        if len(short):
            devices = sorted(f.loc[f["pv"].isin(short.index), "device"].unique().tolist())
            add(file_id, "incomplete_cycle_grid", short.sum(),
                f"fewer than {f['cycle_id'].nunique()} cycles for {devices}")
        steps = np.unique(np.diff(np.sort(f["cycle_id"].unique())))
        if len(steps) > 1:
            add(file_id, "irregular_cycle_stride", len(f), f"cycle_id steps {steps.tolist()}")
        rf = f[f["kind"] == "rf"]
        n_components = rf.groupby(["device", "signal", "cycle_id"])["component"].nunique()
        if (n_components < 2).any():
            lone = n_components[n_components < 2]
            add(file_id, "missing_component", len(lone),
                f"amplitude or phase missing for {sorted(lone.index.get_level_values('device').unique().tolist())}")
        signals = rf.groupby("device")["signal"].nunique()
        if (signals < len(SIGNALS)).any():
            add(file_id, "missing_signal", (signals < len(SIGNALS)).sum(),
                f"not all of {list(SIGNALS.values())} for {sorted(signals[signals < len(SIGNALS)].index.tolist())}")

    cycle_ts = df.groupby(["event_id", "cycle_id"])["cycle_ts"].nunique()
    if (cycle_ts > 1).any():
        add(-1, "cycle_ts_disagrees_across_files", (cycle_ts > 1).sum(),
            "the same (event_id, cycle_id) has different cycle_ts in different system files")
    n_events = df.groupby(["pv", "cycle_id"])["event_id"].nunique()
    if (n_events > 1).any():
        add(-1, "cycle_in_several_events", (n_events > 1).sum(),
            "the same (pv, cycle_id) was dumped by more than one event; drop duplicates before splitting data")
    return issues


# --------------------------------------------------------------------------------------------------
# Waveform loading
# --------------------------------------------------------------------------------------------------

def _read_rows(path: Path, positions: np.ndarray) -> list[np.ndarray]:
    """Return the waveforms at the given ascending row positions as owned float32 arrays.

    The file is one row group, so the column is decoded from the start; reading in small batches
    with a bounded buffer keeps memory near the size of the result instead of the whole column.
    """
    out: list[np.ndarray] = [None] * len(positions)  # type: ignore[list-item]
    with pq.ParquetFile(path, pre_buffer=False, buffer_size=1 << 20) as pf:
        start = 0
        for batch in pf.iter_batches(batch_size=64, columns=[WAVEFORM_COL], use_threads=False):
            stop = start + batch.num_rows
            lo, hi = np.searchsorted(positions, [start, stop])
            if hi > lo:
                col = batch.column(0)
                if col.null_count:
                    raise ValueError(f"{path}: null waveform in rows {start}..{stop}")
                offsets = col.offsets.to_numpy()
                values = col.values.to_numpy(zero_copy_only=False)
                for k in range(lo, hi):
                    i = positions[k] - start
                    out[k] = values[offsets[i]:offsets[i + 1]].astype(np.float32, copy=True)
            start = stop
            if hi == len(positions):
                break
    return out


def load_waveforms(sel: Catalogue, skip_empty: bool = True, max_bytes: int = 2 << 30,
                   max_workers: int = 8) -> pd.DataFrame:
    """Load the waveforms of the selected rows: the catalogue frame plus a ``data`` column.

    Works for any PV (no pairing). ``data`` holds one float32 array per row.
    """
    frame = sel.frame
    if skip_empty and frame["is_empty"].any():
        warnings.warn(f"skipping {int(frame['is_empty'].sum())} empty waveforms", stacklevel=2)
        frame = frame[~frame["is_empty"]]
    frame = frame.reset_index(drop=True)

    n_bytes = int(frame["n_samples"].sum()) * 4
    if n_bytes > max_bytes:
        raise ValueError(
            f"selection needs {n_bytes / 2**30:.2f} GiB of waveforms, above max_bytes="
            f"{max_bytes / 2**30:.2f} GiB; narrow the selection, use iter_paired, or raise max_bytes"
        )

    paths = {f.file_id: f.path for f in sel.files}
    jobs = []
    for file_id, group in frame.groupby("file_id"):
        group = group.sort_values("row_in_file")
        jobs.append((paths[file_id], group["row_in_file"].to_numpy(), group.index.to_numpy()))

    data = np.empty(len(frame), dtype=object)
    if jobs:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
            results = pool.map(lambda job: _read_rows(job[0], job[1]), jobs)
            for (path, _, index), arrays in zip(jobs, results):
                for i, array in zip(index, arrays):
                    if array is None:
                        raise ValueError(f"{path}: row for catalogue entry {i} was not found")
                    data[i] = array

    lengths = np.fromiter((len(a) for a in data), dtype="int64", count=len(data))
    if not np.array_equal(lengths, frame["n_samples"].to_numpy()):
        raise ValueError("loaded waveform lengths do not match the catalogue n_samples")
    frame = frame.copy()
    frame["data"] = pd.Series(data, index=frame.index, dtype=object)
    return frame


def load_paired(sel: Catalogue, strict: bool = True, skip_empty: bool = True, max_bytes: int = 2 << 30,
                max_workers: int = 8) -> pd.DataFrame:
    """Load cavity waveforms with amplitude and phase side by side.

    Returns a frame indexed by ``(device, event_id, cycle_id, signal)`` with columns ``amplitude``
    and ``phase_deg`` (one float32 array each) followed by cycle metadata. ``strict=True`` raises
    when a signal lacks one component or the two were not acquired with the same settings;
    ``strict=False`` drops those signals with a warning.
    """
    frame = sel.frame
    if (frame["kind"] != "rf").any():
        raise ValueError("load_paired only handles cavity PVs; use load_waveforms for other devices")
    if skip_empty and frame["is_empty"].any():
        warnings.warn(f"skipping {int(frame['is_empty'].sum())} empty waveforms", stacklevel=2)
        frame = frame[~frame["is_empty"]]
    frame = frame.drop_duplicates(["file_id", "row_in_file"])  # the same row selected twice

    groups = frame.groupby(PAIR_KEY, sort=False)
    stats = groups[["component", "n_samples", "trig_ts", "trig_delay"]].nunique(dropna=False)
    stats["n_rows"] = groups.size()
    unpaired = stats["component"] != 2
    repeated = ~unpaired & (stats["n_rows"] != 2)
    misaligned = ~unpaired & ~repeated & (stats[["n_samples", "trig_ts", "trig_delay"]] > 1).any(axis=1)
    bad = unpaired | repeated | misaligned
    if bad.any():
        message = (
            f"{int(unpaired.sum())} signals without both amplitude and phase, {int(repeated.sum())} with "
            f"several rows per component, {int(misaligned.sum())} with differing n_samples / trig_ts / trig_delay; first: {stats.index[bad][:5].tolist()}"
        )
        if strict:
            raise ValueError(message + " (select each component exactly once, or pass strict=False to drop them)")
        warnings.warn("dropping " + message, stacklevel=2)
        keep = pd.MultiIndex.from_frame(frame[PAIR_KEY]).isin(stats.index[~bad])
        frame = frame[keep]

    long = load_waveforms(Catalogue(frame.reset_index(drop=True), sel.files, sel.issues),
                          skip_empty=False, max_bytes=max_bytes, max_workers=max_workers)
    amplitude = long[long["component"] == "amplitude"].set_index(PAIR_KEY)
    phase = long[long["component"] == "phase_deg"].set_index(PAIR_KEY)

    meta = ["n_samples", "campaign", "ctype", "system", "section", "module", "cavity",
            "event_ts", "cycle_ts", "cycle_offset", "t_rel_s",
            "beam_mode", "beam_state", "beam_present", "beam_len", "beam_current", "trig_ts_ok"]
    paired = amplitude[["data", *meta]].rename(columns={"data": "amplitude"})
    paired.insert(1, "phase_deg", phase["data"].reindex(paired.index))
    paired["amp_is_zero"] = [not a.any() for a in paired["amplitude"]]
    return paired.sort_index()


def iter_paired(sel: Catalogue, by: str = "event_id", **kwargs: Any) -> Iterator[tuple[Any, pd.DataFrame]]:
    """Yield ``(value, load_paired(...))`` one ``by`` group at a time, to sweep large selections."""
    for value, group in sel.frame.groupby(by, sort=True):
        yield value, load_paired(Catalogue(group.reset_index(drop=True), sel.files, sel.issues), **kwargs)


def stack(arrays: pd.Series) -> np.ndarray:
    """Stack equal-length waveforms into a 2-D float32 array (rows, samples).

    Raises on mixed lengths instead of padding or truncating: group by ``n_samples`` first.
    """
    arrays = list(arrays)
    lengths = sorted({len(a) for a in arrays})
    if len(lengths) > 1:
        raise ValueError(f"cannot stack waveforms of different lengths {lengths}; group by n_samples first")
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(arrays).astype(np.float32, copy=False)


def to_complex(amplitude: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Complex waveform from amplitude and phase in degrees."""
    return amplitude * np.exp(1j * np.deg2rad(phase_deg))


def time_axis(n_samples: int, dt_us: float = NOMINAL_DT_US["rf"]) -> np.ndarray:
    """Sample times in microseconds from the start of the waveform, at the nominal spacing."""
    return np.arange(n_samples) * dt_us


# --------------------------------------------------------------------------------------------------
# Side tables: challenge sample and event reports
# --------------------------------------------------------------------------------------------------

def load_sample(path: str | Path = "data/sample/ess_llrf_sample.parquet") -> pd.DataFrame:
    """The challenge sample in its original row order (``holdout_ids.json`` indexes it by position).

    Its ``I`` / ``Q`` columns are really amplitude and phase in degrees and are renamed accordingly.
    It is a separate, resampled (256-point) dataset and is not meant to be joined to the catalogue.
    """
    df = pd.read_parquet(path).rename(columns={"I": "amplitude", "Q": "phase_deg"})
    df[["event_id", "cycle_id"]] = df[["event_id", "cycle_id"]].astype("int64")
    return df


def load_split(path: str | Path = "holdout_ids.json") -> tuple[np.ndarray, np.ndarray]:
    """``(train_idx, holdout_idx)`` as row positions into ``load_sample()`` (use with ``.iloc``)."""
    split = json.loads(Path(path).read_text())
    return np.asarray(split["train_idx"], dtype="int64"), np.asarray(split["holdout_idx"], dtype="int64")


def _local_to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="ISO8601").dt.as_unit("ns").dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")


def load_event_reports(path: str | Path = "Event_reports.xlsx") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the PM event / trip-report matches: ``(events, trips)``.

    ``events`` is indexed by int64 ``event_id``; ``trips`` has one row per tripped MPS input.
    The sheet's timestamps are naive local time and are converted to UTC here. Join to the
    catalogue on ``event_id`` only. MPS input labels (``MBL-050 S1``) are kept verbatim.
    """
    try:
        raw = pd.read_excel(path, sheet_name="Event Matches", dtype={"event_id": str})
    except ImportError as err:
        raise ImportError("reading Event_reports.xlsx needs openpyxl: uv pip install openpyxl") from err

    events = pd.DataFrame({
        "event_id": raw["event_id"].astype("int64"),
        "event_ts": _local_to_utc(raw["event_ts"]),
        "report_time": _local_to_utc(raw["report_time"]),
        "match_status": raw["match_status"],
        "match_delta_seconds": raw["match_delta_seconds"],
        "n_trips": 0,
    }).set_index("event_id")

    rows = []
    matched = raw.dropna(subset=["trip_description"])
    for event_id, stamps, lines in zip(matched["event_id"].astype("int64"), matched["trip_timestamps"],
                                       matched["trip_description"]):
        stamps, lines = str(stamps).splitlines(), str(lines).splitlines()
        if len(stamps) != len(lines):
            raise ValueError(f"event {event_id}: {len(stamps)} trip timestamps for {len(lines)} trip lines")
        for stamp, line in zip(stamps, lines):
            m = _TRIP_RE.search(line)
            if not m:
                raise ValueError(f"event {event_id}: cannot parse trip line {line!r}")
            rows.append({"event_id": event_id, "mps_input": m["mps_input"], "state": m["state"],
                         "condition": m["condition"], "trip_ts": stamp, "offset_s": float(m["offset_s"])})
    trips = pd.DataFrame(rows, columns=["event_id", "mps_input", "state", "condition", "trip_ts", "offset_s"])
    trips["trip_ts"] = _local_to_utc(trips["trip_ts"])
    events["n_trips"] = trips.groupby("event_id").size().reindex(events.index, fill_value=0)
    return events, trips
