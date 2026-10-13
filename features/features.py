"""
compute_features.py
Compute kickoff-notebook-style features for every waveform in the ESS LLRF
parquet files (SPOKES.parquet, MBL.parquet, HBL.parquet).

One waveform = one PV (signal) on one cycle, e.g.
    MBL-010RFC:RFS-DIG-101:Dwn0-Cmp1SDS-PMData   on cycle 3

Components:  Cmp0 = amplitude,  Cmp1 = phase   (renamed "Amp" and "Phase" in the output)

Phase needs two extra steps before computing features:
  1. unwrap   - phase jumps from +180 to -180 deg are artefacts of wrapping, not real changes
  2. mask     - outside the RF pulse the amplitude is ~0 and the phase is pure noise,
                so phase features use only samples where the amplitude is above
                PHASE_MASK_FRACTION x (its maximum)

Outputs (written next to the script):
  features_long.csv  one row per waveform
  features_wide.csv  one row per (cavity, cycle), columns like Dwn0_Phase_std

Run:  python compute_features.py
Needs: numpy (>= 1.21), pandas, pyarrow
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ============================ SETTINGS ============================
FILES = ["SPOKES.parquet", "MBL.parquet", "HBL.parquet"]
BATCH_SIZE = 64              # waveforms read at a time (keeps memory low)
COMPONENT_NAMES = {"Cmp0": "Amp", "Cmp1": "Phase"}
PHASE_UNIT = "deg"           # "deg" or "rad" - check the values: degrees span about -180..180
PHASE_MASK_FRACTION = 0.1    # use phase only where Amp > 10% of max(Amp); set 0 to disable
# =================================================================

PV_PATTERN = (
    r"^(?P<section>[A-Za-z]+)-(?P<cm>\d+)RFC:RFS-DIG-(?P<cav>\d+):"
    r"(?P<chan>Dwn\d+)-(?P<comp>Cmp\d)"
)
META_COLS = ["pv", "cycle_id", "cycle_ts", "array_size"]
FEATURE_NAMES = ["n_samples", "mean", "std", "min", "max", "ptp", "rms", "skew", "kurt",
                 "dwdt_std", "dwdt_max", "fft_peak_bin", "fft_peak_freq", "fft_peak_mag"]


# ---------------------------------------------------------------- features
def waveform_features(w):
    """Same descriptors as the kickoff notebook (section 4), for ONE waveform."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    if n < 4:                                  # too short to say anything (e.g. empty mask)
        return {k: (n if k == "n_samples" else np.nan) for k in FEATURE_NAMES}
    mean = w.mean()
    std = w.std()
    c = w - mean
    s = std + 1e-12
    dwdt = np.diff(w)
    spec = np.abs(np.fft.rfft(c))
    spec[0] = 0.0                              # ignore DC (already captured by `mean`)
    peak = int(spec.argmax())
    return {
        "n_samples": n,
        "mean": mean,
        "std": std,
        "min": w.min(),
        "max": w.max(),
        "ptp": w.max() - w.min(),
        "rms": np.sqrt(np.mean(w ** 2)),
        "skew": np.mean(c ** 3) / s ** 3,
        "kurt": np.mean(c ** 4) / s ** 4 - 3.0,
        "dwdt_std": dwdt.std(),
        "dwdt_max": np.abs(dwdt).max(),
        "fft_peak_bin": peak,
        "fft_peak_freq": peak / n,             # in units of the (unknown) sampling rate
        "fft_peak_mag": spec[peak],
    }


def unwrap_phase(phase):
    """Remove the artificial +-180 deg (or +-pi) jumps caused by phase wrapping."""
    period = 360.0 if PHASE_UNIT == "deg" else 2 * np.pi
    return np.unwrap(np.asarray(phase, dtype=np.float64), period=period)


def rf_on_mask(amp):
    """True where the RF is on, i.e. the amplitude is above a fraction of its maximum."""
    amp = np.asarray(amp, dtype=np.float64)
    if PHASE_MASK_FRACTION <= 0 or amp.max() <= 0:
        return np.ones(len(amp), dtype=bool)
    return amp > PHASE_MASK_FRACTION * amp.max()


# ---------------------------------------------------------------- reading
def iter_waveforms(file):
    """Yield (metadata dict, numpy waveform) one at a time, batch by batch,
    so a 146 MB file never has to be fully in memory as Python objects."""
    pf = pq.ParquetFile(file)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=META_COLS + ["data_array"]):
        meta = batch.select(META_COLS).to_pandas()
        arrays = batch.column("data_array")
        for i in range(batch.num_rows):
            w = arrays[i].values.to_numpy(zero_copy_only=False)
            yield meta.iloc[i].to_dict(), w


# ---------------------------------------------------------------- main work
def amp_phase_rows(ident, amp, phase):
    """Feature rows for one amplitude/phase pair (same channel, cavity, cycle)."""
    rows = []
    if amp is not None:
        f = waveform_features(amp)
        f["rf_on_fraction"] = rf_on_mask(amp).mean()   # how much of the window the RF is on
        rows.append({**ident, "comp": "Amp", **f})
    if phase is not None:
        ph = np.asarray(phase, dtype=np.float64)
        if amp is not None and len(amp) == len(ph):
            ph = ph[rf_on_mask(amp)]                    # 1) keep only the RF-on part
        ph = unwrap_phase(ph)                           # 2) remove the +-180 jumps
        if len(ph):                                     # 3) shift by whole turns so the
            period = 360.0 if PHASE_UNIT == "deg" else 2 * np.pi    # first value is in
            ph = ph - period * np.round(ph[0] / period)              # the usual range
        rows.append({**ident, "comp": "Phase", **waveform_features(ph)})
    return rows


def features_for_file(file):
    rows = []
    pending = {}   # holds the amplitude or phase waiting for its partner
    skipped = 0
    for meta, w in iter_waveforms(file):
        parts = pd.Series([meta["pv"]]).str.extract(PV_PATTERN).iloc[0]
        if parts.isna().any() or parts["comp"] not in COMPONENT_NAMES:
            skipped += 1
            continue
        name = COMPONENT_NAMES[parts["comp"]]
        ident = {k: parts[k] for k in ["section", "cm", "cav", "chan"]}
        ident.update(cycle_id=meta["cycle_id"], cycle_ts=meta["cycle_ts"])
        key = (parts["cm"], parts["cav"], parts["chan"], meta["cycle_id"])

        other = pending.pop(key, None)
        if other is None:
            pending[key] = (name, w, ident)          # wait for the partner
            continue
        sigs = {name: w, other[0]: other[1]}
        rows += amp_phase_rows(ident, sigs.get("Amp"), sigs.get("Phase"))

    # anything still waiting never found its partner: compute it on its own
    for name, w, ident in pending.values():
        rows += amp_phase_rows(ident, w if name == "Amp" else None,
                               w if name == "Phase" else None)
    if pending:
        print(f"  {file}: {len(pending)} signals had no amplitude/phase partner "
              f"(phase computed without the RF-on mask)")
    if skipped:
        print(f"  {file}: skipped {skipped} waveforms whose PV name didn't match the pattern")
    return pd.DataFrame(rows)


def make_wide(long_df):
    """One row per (section, cryomodule, cavity, cycle); one column per
    channel x component x feature, e.g. 'Dwn0_Phase_std'."""
    id_cols = ["section", "cm", "cav", "cycle_id"]
    feat_cols = [c for c in long_df.columns
                 if c not in id_cols + ["chan", "comp", "cycle_ts", "n_samples"]]
    # (default dropna=True: only cavity/cycle combinations that really exist, and
    #  no all-empty columns such as a Phase rf_on_fraction)
    wide = long_df.pivot_table(index=id_cols, columns=["chan", "comp"],
                               values=feat_cols, aggfunc="first")
    wide.columns = [f"{chan}_{comp}_{feat}" for feat, chan, comp in wide.columns]
    wide = wide.dropna(axis=0, how="all")          # safety: no fully empty rows
    wide = wide[sorted(wide.columns)].reset_index()

    # number cycles 0, 1, 2, ... in time order (same as plot_waveforms.py)
    order = (long_df[["cycle_id", "cycle_ts"]].drop_duplicates()
             .sort_values("cycle_ts").reset_index(drop=True))
    order["cycle"] = order.index
    return order[["cycle_id", "cycle"]].merge(wide, on="cycle_id").sort_values(
        ["section", "cm", "cav", "cycle"]).reset_index(drop=True)


if __name__ == "__main__":
    all_long = []
    for f in FILES:
        print(f"Processing {f} ...")
        df = features_for_file(f)
        print(f"  {len(df)} waveforms -> features")
        all_long.append(df)

    long_df = pd.concat(all_long, ignore_index=True)
    long_df.to_csv("features_long.csv", index=False)

    wide_df = make_wide(long_df)
    wide_df.to_csv("features_wide.csv", index=False)

    print(f"\nfeatures_long.csv : {long_df.shape[0]} rows (one per waveform)")
    print(f"features_wide.csv : {wide_df.shape[0]} rows (one per cavity x cycle), "
          f"{wide_df.shape[1]} columns")
    print(wide_df.groupby("section").size().rename("rows per section").to_string())