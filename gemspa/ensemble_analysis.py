#!/usr/bin/env python3
# gemspa/ensemble_analysis.py

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill

# NEW: import RAW and FILTERED ensemble step-size helpers
from .step_size_analysis import (
    run_condition_step_kde,
    run_condition_step_kde_filtered,
)

# ------------------------------
# CSV loading / normalization
# ------------------------------
def _sanitize_csv_load(path):
    """Load a trajectory CSV and normalize headers (TrackMate-friendly)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {}
    if "trajectory" in df.columns:
        alias["trajectory"] = "track_id"
    if "position_x" in df.columns and "x" not in df.columns:
        alias["position_x"] = "x"
    if "position_y" in df.columns and "y" not in df.columns:
        alias["position_y"] = "y"
    if "spot_frame" in df.columns and "frame" not in df.columns:
        alias["spot_frame"] = "frame"
    if alias:
        df = df.rename(columns=alias)
    for req in ("track_id", "frame", "x", "y"):
        if req not in df.columns:
            raise KeyError(f"{os.path.basename(path)} missing required column '{req}'")
    return df


def _condition_from_stem(stem):
    """'Traj_DMSO_001' or 'DMSO_001' → 'DMSO'."""
    rep = stem.replace("Traj_", "") if stem.startswith("Traj_") else stem
    return re.sub(r"_[0-9]+$", "", rep)


def _collect_condition_files(work_dir, condition):
    """Raw CSVs for a condition (top-level and one-level replicate subfolders)."""
    pats = [
        os.path.join(work_dir, f"Traj_{condition}_*.csv"),
        os.path.join(work_dir, f"{condition}_*.csv"),
        os.path.join(work_dir, f"{condition}_*", f"Traj_{condition}_*.csv"),
        os.path.join(work_dir, f"{condition}_*", f"{condition}_*.csv"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    return sorted(set(files))


# ------------------------------
# MSD computation / plotting
# ------------------------------
def _compute_track_msd(px_xy_df, micron_per_px, tlag_cutoff):
    """Return MSD array for a single track (pixels → microns)."""
    coords = px_xy_df[["x", "y"]].to_numpy() * micron_per_px
    L = min(tlag_cutoff, max(1, coords.shape[0] - 1))
    if L <= 0:
        return None
    msd = np.empty(L, dtype=float)
    for lag in range(1, L + 1):
        d = coords[lag:] - coords[:-lag]
        msd[lag - 1] = np.nan if d.size == 0 else np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)
    return msd


def _fit_lin_and_loglog(tau, msd):
    """Fit D from linear MSD and α from log–log MSD."""
    m = np.isfinite(msd) & (msd > 0) & (tau > 0)
    D_lin, alpha = 0.0, 0.0
    if m.sum() >= 2:
        slope, _ = np.polyfit(tau[m], msd[m], 1)
        D_lin = max(0.0, slope / 4.0)
        a, _b = np.polyfit(np.log10(tau[m]), np.log10(msd[m]), 1)
        alpha = float(a)
    return float(D_lin), float(alpha)


def _plot_linear(tau, msd, D_est, title, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(tau, msd, "o", ms=4)
    ax.set_xlabel("τ (s)")
    ax.set_ylabel("MSD (μm²)")
    t = fill(f"{title}\nD = {D_est:.3g} μm²/s", width=70)
    ax.set_title(t, fontsize=10, pad=6)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_loglog(tau, msd, alpha_est, title, out):
    m = (tau > 0) & (msd > 0)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(np.log10(tau[m]), msd[m], "o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("log10 τ (s)")
    ax.set_ylabel("MSD (μm²), log scale")
    t = fill(f"{title}\nα = {alpha_est:.4f}", width=70)
    ax.set_title(t, fontsize=10, pad=6)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ensemble_msd_for_condition(
    work_dir,
    condition,
    micron_per_px,
    time_step,
    tlag_cutoff,
    min_track_len,
    only_track_ids=None,
):
    """
    Compute ensemble-averaged MSD for a condition (optionally restricting to selected track IDs).
    Returns (tau, ens_msd, (D, alpha)) or None if no usable tracks.
    """
    csvs = _collect_condition_files(work_dir, condition)
    if not csvs:
        return None

    msds = []
    for f in csvs:
        try:
            df = _sanitize_csv_load(f)
        except Exception:
            continue

        if only_track_ids is not None:
            stem = os.path.splitext(os.path.basename(f))[0]
            keep = only_track_ids.get(stem, None)
            if keep is not None and len(keep) > 0:
                # normalize to strings to avoid dtype mismatches
                df = df.copy()
                df["track_id"] = df["track_id"].astype(str)
                df = df[df["track_id"].isin(keep)]

        for _, g in df.groupby("track_id"):
            if len(g) < min_track_len:
                continue
            msd = _compute_track_msd(g.sort_values("frame"), micron_per_px, tlag_cutoff)
            if msd is not None and np.isfinite(msd).any():
                msds.append(msd)

    if not msds:
        return None

    L = min(len(v) for v in msds)
    if L < 1:
        return None

    msd_mat = np.vstack([v[:L] for v in msds])
    ens_msd = np.nanmean(msd_mat, axis=0)
    tau = np.arange(1, L + 1, dtype=float) * time_step
    D, alpha = _fit_lin_and_loglog(tau, ens_msd)
    return tau, ens_msd, (D, alpha)


def _load_filtered_track_ids(work_dir, condition, filt):
    """
    From each replicate's msd_results.csv, collect track IDs that pass D/α filters.
    Returns a dict keyed by BOTH "<cond>_<rep>" and "Traj_<cond>_<rep>" for robust matching.
    Track IDs are normalized to strings to avoid dtype mismatches.
    """
    d = {}
    rep_dirs = sorted(glob.glob(os.path.join(work_dir, f"{condition}_*"))) + sorted(
        glob.glob(os.path.join(work_dir, f"Traj_{condition}_*"))
    )
    for rd in rep_dirs:
        csvp = os.path.join(rd, "msd_results.csv")
        if not os.path.exists(csvp):
            continue
        df = pd.read_csv(csvp)
        if not {"D_fit", "alpha_fit", "track_id"}.issubset(df.columns):
            continue
        m = (df["D_fit"].between(filt["D_min"], filt["D_max"])) & (
            df["alpha_fit"].between(filt["alpha_min"], filt["alpha_max"])
        )
        # Normalize IDs to strings
        keep = set(df.loc[m, "track_id"].astype(str).tolist())
        stem = os.path.basename(rd)  # e.g., "DMSO_001"
        d[stem] = keep
        d["Traj_" + stem] = keep  # e.g., "Traj_DMSO_001"
        print(f"[ensemble] filtered keep IDs for {stem}: {len(keep)}")
    return d


# ------------------------------
# Public API
# ------------------------------
def run_ensemble(
    work_dir,
    filter_D_min=0.001,
    filter_D_max=2.0,
    filter_alpha_min=0.0,
    filter_alpha_max=2.0,
    time_step=0.010,
    micron_per_px=0.11,
    tlag_cutoff=10,
    min_track_len=11,
):
    """
    Build grouped tables and ensemble MSD plots per condition.

    Outputs:
      - grouped_raw/msd_results.csv
      - grouped_filtered/msd_results.csv
      - grouped_raw/ensemble_msd_vs_tau_<condition>.png
      - grouped_raw/ensemble_msd_vs_tau_loglog_<condition>.png
      - grouped_filtered/ensemble_msd_vs_tau_<condition>.png
      - grouped_filtered/ensemble_msd_vs_tau_loglog_<condition>.png
      - grouped_raw/step_kde/step_kde_<condition>_(ensemble).png
      - grouped_filtered/step_kde/step_kde_<condition>_(filtered_ensemble).png
    """
    # Identify conditions from replicate folders already created by the per-file step
    rep_dirs = sorted([d for d in glob.glob(os.path.join(work_dir, "*")) if os.path.isdir(d)])
    conditions = {}
    for rd in rep_dirs:
        stem = os.path.basename(rd)
        cond = _condition_from_stem(stem)
        conditions.setdefault(cond, []).append(rd)

    filt = dict(
        D_min=filter_D_min,
        D_max=filter_D_max,
        alpha_min=filter_alpha_min,
        alpha_max=filter_alpha_max,
    )

    grouped_raw_dir = os.path.join(work_dir, "grouped_raw")
    grouped_filt_dir = os.path.join(work_dir, "grouped_filtered")
    os.makedirs(grouped_raw_dir, exist_ok=True)
    os.makedirs(grouped_filt_dir, exist_ok=True)

    all_raw_rows = []
    all_filt_rows = []

    for cond, _reps in conditions.items():
        # Gather replicate msd_results.csv for grouped tables
        rep_dirs_c = sorted(glob.glob(os.path.join(work_dir, f"{cond}_*"))) + sorted(
            glob.glob(os.path.join(work_dir, f"Traj_{cond}_*"))
        )
        for rd in rep_dirs_c:
            p = os.path.join(rd, "msd_results.csv")
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            if {"D_fit", "alpha_fit"}.issubset(df.columns):
                df["condition"] = cond
                all_raw_rows.append(df)
                m = (df["D_fit"].between(filt["D_min"], filt["D_max"])) & (
                    df["alpha_fit"].between(filt["alpha_min"], filt["alpha_max"])
                )
                all_filt_rows.append(df.loc[m].copy())

        # -------- RAW ensemble MSD + RAW ensemble step-size KDE --------
        res = _ensemble_msd_for_condition(
            work_dir,
            cond,
            micron_per_px,
            time_step,
            tlag_cutoff,
            min_track_len,
            only_track_ids=None,
        )
        if res is not None:
            tau, msd, (D, alpha) = res
            _plot_linear(
                tau,
                msd,
                D,
                f"ens-avg MSD (2d) — {cond}",
                os.path.join(grouped_raw_dir, f"ensemble_msd_vs_tau_{cond}.png"),
            )
            _plot_loglog(
                tau,
                msd,
                alpha,
                f"ens-avg log-log MSD (2d) — {cond}",
                os.path.join(grouped_raw_dir, f"ensemble_msd_vs_tau_loglog_{cond}.png"),
            )
        else:
            print(f"[ensemble] No usable tracks for RAW ensemble of {cond}; skipping MSD plots.")

        raw_step_dir = os.path.join(grouped_raw_dir, "step_kde")
        run_condition_step_kde(work_dir, cond, raw_step_dir)

        # -------- FILTERED ensemble MSD + FILTERED ensemble step-size KDE --------
        keep_ids = _load_filtered_track_ids(work_dir, cond, filt)
        res_f = _ensemble_msd_for_condition(
            work_dir,
            cond,
            micron_per_px,
            time_step,
            tlag_cutoff,
            min_track_len,
            only_track_ids=keep_ids,
        )
        if res_f is not None:
            tau_f, msd_f, (D_f, alpha_f) = res_f
            _plot_linear(
                tau_f,
                msd_f,
                D_f,
                f"ens-avg MSD (2d) — {cond} (filtered)",
                os.path.join(grouped_filt_dir, f"ensemble_msd_vs_tau_{cond}.png"),
            )
            _plot_loglog(
                tau_f,
                msd_f,
                alpha_f,
                f"ens-avg log-log MSD (2d) — {cond} (filtered)",
                os.path.join(grouped_filt_dir, f"ensemble_msd_vs_tau_loglog_{cond}.png"),
            )
        else:
            print(f"[ensemble] No usable tracks for FILTERED ensemble of {cond}; skipping MSD plots.")

        filt_step_dir = os.path.join(grouped_filt_dir, "step_kde")
        run_condition_step_kde_filtered(
            work_dir,
            cond,
            keep_ids_map=keep_ids,
            out_dir=filt_step_dir,
            micron_per_px=micron_per_px,
            tlag_cutoff=tlag_cutoff,
            min_track_len=max(2, min_track_len),
        )

    # Write grouped tables once per run (all conditions stacked)
    if all_raw_rows:
        gr = pd.concat(all_raw_rows, ignore_index=True)
        gr.to_csv(os.path.join(grouped_raw_dir, "msd_results.csv"), index=False)
    if all_filt_rows:
        gf = pd.concat(all_filt_rows, ignore_index=True)
        gf.to_csv(os.path.join(grouped_filt_dir, "msd_results.csv"), index=False)
