#!/usr/bin/env python3
# gemspa/step_size_analysis.py

import os
import glob
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.stats import ks_2samp

plt.rcParams.setdefault("font.size", 12)

STEP_FILE_NAME = "all_data_step_sizes.txt"


# ------------------------------
# Loading / reshaping helpers
# ------------------------------
def _is_wide_format(df: pd.DataFrame) -> bool:
    core = {"group", "tlag"}
    extra = [c for c in df.columns if c not in core]
    if len(extra) < 2:
        return False
    # consider it wide if at least half the extra columns look numeric-like
    numericish = 0
    for c in extra:
        try:
            float(str(c).replace("_", "").replace(",", "").replace("−", "-"))
            numericish += 1
        except Exception:
            continue
    return numericish >= max(2, int(0.5 * len(extra)))


def load_step_data(path: str) -> pd.DataFrame:
    """
    Load step-size data from TSV/CSV and return **long-form** [group, tlag, step_size].

    Accepts either:
      - long format with columns: group, tlag, step_size (plus common aliases)
      - wide format with columns: group, tlag, and many numeric columns per row
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Basic renames (aliases)
    rename = {}
    if "track_group" in df.columns and "group" not in df.columns:
        rename["track_group"] = "group"
    if "lag" in df.columns and "tlag" not in df.columns:
        rename["lag"] = "tlag"
    if "stepsize" in df.columns and "step_size" not in df.columns:
        rename["stepsize"] = "step_size"
    df = df.rename(columns=rename)

    # Wide → long if needed
    if _is_wide_format(df):
        core = ["group", "tlag"]
        value_cols = [c for c in df.columns if c not in core]
        df = df.melt(
            id_vars=core, value_vars=value_cols, var_name="idx", value_name="step_size"
        ).drop(columns=["idx"])

    # Keep only the needed columns (infer step col if ambiguous)
    needed = {"group", "tlag", "step_size"}
    if not needed.issubset(df.columns):
        candidates = [c for c in df.columns if c not in ("group", "tlag")]
        if not candidates:
            raise ValueError(
                f"No step-size column found in {path}. Columns={list(df.columns)}"
            )
        df = df.rename(columns={candidates[0]: "step_size"})

    # Clean types
    df = df.dropna(subset=["group", "tlag"])
    df["tlag"] = pd.to_numeric(df["tlag"], errors="coerce")
    df["step_size"] = pd.to_numeric(df["step_size"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["tlag", "step_size"])

    # Ranges
    df = df[df["step_size"] >= 0]
    df["tlag"] = df["tlag"].astype(int)
    return df


# ------------------------------
# Analytics / plotting
# ------------------------------
def calc_alpha2(obs: np.ndarray) -> float:
    """Non-Gaussian parameter α₂ = ⟨r⁴⟩ / (3⟨r²⟩²) – 1."""
    obs = np.asarray(obs, dtype=float)
    obs = obs[np.isfinite(obs)]
    if obs.size == 0:
        return np.nan
    m2 = np.mean(obs ** 2)
    if m2 == 0:
        return np.nan
    return float(np.mean(obs ** 4) / (3.0 * m2 ** 2) - 1.0)


def plot_step_kde(df: pd.DataFrame, results_dir: str) -> None:
    """
    For each group and each τ (tlag), plot a KDE of step_size.
    Y-axis is log-scaled density. Lines are color-coded by τ (color-blind-safe).
    """
    os.makedirs(results_dir, exist_ok=True)

    for group, gdf in df.groupby("group"):
        fig, ax = plt.subplots(figsize=(10, 7))

        taus = sorted(gdf["tlag"].unique())
        palette = sns.color_palette("colorblind", n_colors=len(taus))
        colors = dict(zip(taus, palette))

        plotted = False
        alpha2_vals = {}
        for t, sub in gdf.groupby("tlag"):
            vals = sub["step_size"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size < 3 or np.allclose(vals, vals[0]):
                continue

            a2 = calc_alpha2(vals)
            if np.isfinite(a2):
                alpha2_vals[int(t)] = a2

            # Slightly smoother than default to avoid tiny valleys on log-y
            sns.kdeplot(
                x=vals,
                fill=False,
                bw_method="silverman",
                bw_adjust=1.2,
                warn_singular=False,
                common_norm=False,
                color=colors[int(t)],
                label=f"τ={int(t)}",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"[step_size] No KDE layers for group {group!r}; skipping.")
            continue

        ax.set_title(f"{group}")
        ax.set_xlabel("Step Size (μm)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatter(base=10))
        ax.set_ylabel("Density (log₁₀)")
        ax.legend(loc="upper right", fontsize=9)

        # inset α₂ annotation
        if alpha2_vals:
            lines = [f"τ={k}: α₂={v:.2f}" for k, v in sorted(alpha2_vals.items())]
            ax.text(
                0.98,
                0.02,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
                fontsize=9,
            )

        fig.tight_layout()
        out = os.path.join(
            results_dir, f"step_kde_{str(group).replace(' ', '_')}.png"
        )
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[step_size] wrote {out}")


def ks_comparison(df: pd.DataFrame, group_a: str, group_b: str, results_dir: str) -> None:
    """KS tests per τ between two groups; save volcano-like plot (−log10 p vs τ)."""
    os.makedirs(results_dir, exist_ok=True)
    taus = sorted(set(df["tlag"].tolist()))
    pvals = []
    for t in taus:
        a = (
            df[(df["group"] == group_a) & (df["tlag"] == t)]["step_size"]
            .dropna()
            .to_numpy()
        )
        b = (
            df[(df["group"] == group_b) & (df["tlag"] == t)]["step_size"]
            .dropna()
            .to_numpy()
        )
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 3 or b.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
            pvals.append(np.nan)
            continue
        pvals.append(ks_2samp(a, b).pvalue)

    if not pvals or all([not np.isfinite(x) for x in pvals]):
        print("[step_size] KS comparison skipped (insufficient data).")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    y = [
        -np.log10(p) if (p is not None and np.isfinite(p) and p > 0) else np.nan
        for p in pvals
    ]
    ax.plot(taus, y, "o-")
    ax.set_xlabel("τ (frames)")
    ax.set_ylabel("−log10 p (KS)")
    ax.set_title(f"KS by τ: {group_a} vs {group_b}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(
        results_dir,
        f"ks_volcano_{str(group_a)}_vs_{str(group_b)}.png".replace(" ", "_"),
    )
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[step_size] wrote {out}")


# ------------------------------
# RAW & FILTERED condition/ensemble step-size KDEs
# ------------------------------
def _iter_replicate_step_files(work_dir: str, condition: str):
    """Yield all replicate step-size files for a condition (robust to naming)."""
    pats = [
        os.path.join(work_dir, f"{condition}_*", STEP_FILE_NAME),
        os.path.join(work_dir, f"Traj_{condition}_*", STEP_FILE_NAME),
    ]
    seen = set()
    for p in pats:
        for f in glob.glob(p):
            if os.path.isfile(f) and f not in seen:
                seen.add(f)
                yield f


def _collect_condition_csvs(work_dir: str, condition: str):
    """Top-level + one-level replicate raw trajectory CSVs for a condition."""
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


def _normalize_headers_inplace(df: pd.DataFrame, fpath: str) -> bool:
    """
    Case-insensitive, robust header normalization for raw trajectory CSVs.
    Returns True if canonical columns are present after renaming; False otherwise.
    """
    cols = {c.lower(): c for c in df.columns}

    def first_present(*opts):
        for o in opts:
            if o in cols:
                return cols[o]  # original-cased column
        return None

    src_track = first_present("track_id", "trajectory", "track id", "trackid", "track")
    src_x     = first_present("x", "position_x", "pos_x", "x (px)", "x_px")
    src_y     = first_present("y", "position_y", "pos_y", "y (px)", "y_px")
    src_frame = first_present("frame", "spot_frame", "t", "time", "frame_id")

    rename_map = {}
    if src_track and src_track != "track_id":
        rename_map[src_track] = "track_id"
    if src_x and src_x != "x":
        rename_map[src_x] = "x"
    if src_y and src_y != "y":
        rename_map[src_y] = "y"
    if src_frame and src_frame != "frame":
        rename_map[src_frame] = "frame"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required = {"track_id", "x", "y", "frame"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        print(
            f"[step_size] {os.path.basename(fpath)} missing columns after normalize: "
            f"{sorted(missing)}; skipping"
        )
        return False
    return True


def _compute_track_step_sizes(track_df: pd.DataFrame, micron_per_px: float, tlag_cutoff: int):
    """Compute step sizes for lags 1..tlag_cutoff for a single track."""
    g = track_df.sort_values("frame")
    xy = g[["x", "y"]].to_numpy(dtype=float) * float(micron_per_px)
    n = xy.shape[0]
    L = min(tlag_cutoff, max(1, n - 1))
    out = []
    for lag in range(1, L + 1):
        d = xy[lag:] - xy[:-lag]
        if d.size == 0:
            continue
        steps = np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2))
        out.append((lag, steps))
    return out  # list of (tlag, np.array)


def run_condition_step_kde(work_dir: str, condition: str, out_dir: str,
                           micron_per_px: float = 0.11, tlag_cutoff: int = 10,
                           min_track_len: int = 2) -> bool:
    """
    RAW ensemble KDE:
      1) Try aggregating per-replicate 'all_data_step_sizes.txt' files.
      2) If none found (or empty), fall back to recomputing from raw trajectory CSVs
         using the same robust header normalization as the filtered path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- First try: per-replicate step-size files
    frames = []
    for step_file in _iter_replicate_step_files(work_dir, condition):
        try:
            df = load_step_data(step_file)  # handles wide/long
            if not df.empty:
                df = df.copy()
                df["group"] = f"{condition} (ensemble)"
                frames.append(df[["group", "tlag", "step_size"]])
        except Exception as e:
            print(f"[step_size] skip {step_file}: {e}")

    # If we have data, plot and return
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        if not all_df.empty:
            plot_step_kde(all_df, out_dir)
            return True

    # ---- Fallback: build RAW ensemble directly from raw CSVs
    print(f"[step_size] RAW ensemble fallback: recomputing from raw CSVs for {condition!r}")
    csvs = _collect_condition_csvs(work_dir, condition)
    if not csvs:
        print(f"[step_size] raw fallback: no raw CSVs for {condition!r}")
        return False

    rows = []
    total_pts = 0
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=None, engine="python")
        except Exception:
            continue

        if not _normalize_headers_inplace(df, f):
            continue

        # compute steps for ALL tracks (no filtering)
        for tid, g in df.groupby("track_id"):
            if len(g) < min_track_len:
                continue
            for lag, steps in _compute_track_step_sizes(g, micron_per_px, tlag_cutoff):
                rows.append((f"{condition} (ensemble)", int(lag), steps))
                total_pts += steps.size

    if not rows:
        print(f"[step_size] raw fallback: no usable steps for {condition!r}")
        return False

    group_col, tlag_col, step_col = [], [], []
    for grp, lag, arr in rows:
        if arr.size == 0:
            continue
        group_col.extend([grp] * arr.size)
        tlag_col.extend([lag] * arr.size)
        step_col.extend(arr.tolist())

    df_long = pd.DataFrame({"group": group_col, "tlag": tlag_col, "step_size": step_col})
    plot_step_kde(df_long, out_dir)
    print(f"[step_size] RAW ensemble fallback wrote {total_pts} points for {condition!r}")
    return True


def run_condition_step_kde_filtered(
    work_dir: str,
    condition: str,
    keep_ids_map: dict,
    out_dir: str,
    micron_per_px: float = 0.11,
    tlag_cutoff: int = 10,
    min_track_len: int = 2,
) -> bool:
    """
    Recompute step sizes using only tracks that pass D/α filters (keep_ids_map),
    and write a single KDE per condition.

    keep_ids_map: dict mapping replicate stems to sets of track IDs to keep.
                  Keys may look like "V1_001" and/or "Traj_V1_001".
    """
    os.makedirs(out_dir, exist_ok=True)
    csvs = _collect_condition_csvs(work_dir, condition)
    if not csvs:
        print(f"[step_size] filtered ensemble: no raw CSVs for {condition!r}")
        return False

    def _lookup_keep_ids(stem: str):
        """Robust key lookup: try several variants and then match by replicate suffix."""
        s0 = stem
        s1 = stem[5:] if stem.startswith("Traj_") else stem   # drop Traj_
        # direct matches
        keep = keep_ids_map.get(s0) or keep_ids_map.get(s1) or keep_ids_map.get("Traj_" + s1)
        if keep:
            return keep
        # fallback: match by replicate suffix (e.g., _001)
        m = re.search(r"_(\d+)$", s1)
        if m:
            suf = m.group(0)  # includes underscore
            for k, v in keep_ids_map.items():
                if k.endswith(suf):
                    return v
        return None

    rows = []
    total_kept_rows = 0
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=None, engine="python")
        except Exception:
            continue

        if not _normalize_headers_inplace(df, f):
            continue

        stem = os.path.splitext(os.path.basename(f))[0]
        keep_ids = _lookup_keep_ids(stem)
        if not keep_ids:
            # nothing to include from this file for filtered ensemble
            continue

        # normalize both sides to strings for safe comparison
        keep_ids = {str(x) for x in keep_ids}
        df = df.copy()
        df["track_id"] = df["track_id"].astype(str)

        before = len(df)
        df = df[df["track_id"].isin(keep_ids)]
        after = len(df)
        print(f"[step_size] {stem}: kept {after}/{before} rows after filter")

        for tid, g in df.groupby("track_id"):
            if len(g) < min_track_len:
                continue
            for lag, steps in _compute_track_step_sizes(g, micron_per_px, tlag_cutoff):
                rows.append((f"{condition} (filtered ensemble)", int(lag), steps))
                total_kept_rows += steps.size

    if not rows:
        print(f"[step_size] no filtered step sizes for {condition!r}")
        return False

    # Build long-form DataFrame
    group_col, tlag_col, step_col = [], [], []
    for grp, lag, arr in rows:
        if arr.size == 0:
            continue
        group_col.extend([grp] * arr.size)
        tlag_col.extend([lag] * arr.size)
        step_col.extend(arr.tolist())

    df_long = pd.DataFrame({"group": group_col, "tlag": tlag_col, "step_size": step_col})
    plot_step_kde(df_long, out_dir)
    print(f"[step_size] filtered ensemble steps written for {condition!r}: {total_kept_rows} points")
    return True


# ------------------------------
# Public entry (per-replicate)
# ------------------------------
def run_step_size_analysis_if_requested(results_dir: str) -> None:
    """If {results_dir}/all_data_step_sizes.txt exists, render KDE(s) and KS plot."""
    step_file = os.path.join(results_dir, STEP_FILE_NAME)
    if not os.path.isfile(step_file):
        print(f"[step_size] {STEP_FILE_NAME} not found in {results_dir}; skipping.")
        return

    try:
        df = load_step_data(step_file)
        print(f"[step_size] loaded {len(df)} rows from {step_file}")
        plot_step_kde(df, results_dir)
        groups = list(df["group"].dropna().unique())
        if len(groups) >= 2:
            ks_comparison(df, groups[0], groups[1], results_dir)
    except Exception as e:
        print(f"Step-size analysis failed: {e}")
