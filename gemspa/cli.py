#!/usr/bin/env python3
# gemspa/cli.py

import argparse
import glob
import os
import re
from functools import partial
from multiprocessing import Pool, cpu_count

# Absolute imports from the installed package
from gemspa.trajectory_analysis import trajectory_analysis
from gemspa.step_size_analysis import run_step_size_analysis_if_requested
from gemspa.ensemble_analysis import run_ensemble
from gemspa.compare_conditions import compare_conditions


def _process_replicate(args, csv_file):
    base = os.path.basename(csv_file)
    name, _ = os.path.splitext(base)

    # Replicate name & condition parsing (works with and without 'Traj_' prefix)
    replicate = name.replace('Traj_', '') if name.startswith('Traj_') else name
    condition = re.sub(r'_[0-9]+$', '', replicate)

    result_dir = os.path.join(args.work_dir, replicate)
    os.makedirs(result_dir, exist_ok=True)

    # threads per replicate
    tpr = args.threads_per_rep if args.threads_per_rep is not None else max(
        1, cpu_count() // max(1, args.n_jobs)
    )

    ta = trajectory_analysis(
        data_file=csv_file,
        results_dir=result_dir,
        condition=condition,
        time_step=args.time_step,
        micron_per_px=args.micron_per_px,
        ts_resolution=args.ts_resolution,
        min_track_len_linfit=args.min_track_len,
        tlag_cutoff_linfit=args.tlag_cutoff,
        make_rainbow_tracks=args.rainbow_tracks,
        img_file_prefix=args.img_prefix,
        rainbow_min_D=args.rainbow_min_D,
        rainbow_max_D=args.rainbow_max_D,
        rainbow_colormap=args.rainbow_colormap,
        rainbow_scale=args.rainbow_scale,
        rainbow_dpi=args.rainbow_dpi,
        n_jobs=args.n_jobs,
        threads_per_rep=tpr,
    )

    ta.write_params_to_log_file()
    ta.calculate_msd_and_diffusion()

    if args.step_size_analysis:
        ta.export_step_sizes()
        run_step_size_analysis_if_requested(result_dir)


def main():
    parser = argparse.ArgumentParser(
        description="GEMspa Single-Particle Tracking Analysis CLI"
    )
    parser.add_argument(
        "-d", "--work-dir", required=True, help="Directory with CSV trajectory files"
    )
    parser.add_argument(
        "--csv-pattern",
        default="Traj_*.csv",
        help="Glob for input CSVs (default: Traj_*.csv). "
             "Examples: '*Spots in tracks*.csv' for TrackMate.",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=cpu_count(),
        help="Parallel processes (across replicates)",
    )
    parser.add_argument(
        "--threads-per-rep",
        type=int,
        default=None,
        help="Threads per replicate (default=cores / n_jobs)",
    )

    # Core SPT / MSD params
    parser.add_argument("--time-step", type=float, default=0.010)
    parser.add_argument("--micron-per-px", type=float, default=0.11)
    parser.add_argument("--ts-resolution", type=float, default=0.005)
    parser.add_argument("--min-track-len", type=int, default=11)
    parser.add_argument("--tlag-cutoff", type=int, default=10)

    # Rainbow overlays
    parser.add_argument("--rainbow-tracks", action="store_true")
    parser.add_argument("--img-prefix", default="MAX_")
    parser.add_argument("--rainbow-min-D", type=float, default=0.0)
    parser.add_argument("--rainbow-max-D", type=float, default=2.0)
    parser.add_argument("--rainbow-colormap", default="viridis")
    parser.add_argument("--rainbow-scale", type=float, default=1.0)
    parser.add_argument("--rainbow-dpi", type=int, default=200)

    # Ensemble filtering & compare
    parser.add_argument("--filter-D-min", type=float, default=0.001)
    parser.add_argument("--filter-D-max", type=float, default=2.0)
    parser.add_argument("--filter-alpha-min", type=float, default=0.0)
    parser.add_argument("--filter-alpha-max", type=float, default=2.0)

    # Step-size analysis
    parser.add_argument(
        "--step-size-analysis",
        action="store_true",
        help="Also run the step-size KDE & KS analysis",
    )

    args = parser.parse_args()

    # Discover CSVs (skip empties)
    all_csv = glob.glob(os.path.join(args.work_dir, args.csv_pattern))
    csvs = [f for f in all_csv if os.path.getsize(f) > 0]
    if not csvs:
        parser.exit(
            message=f"No files matching {args.csv_pattern!r} in {args.work_dir}\n"
        )

    # Per-replicate processing
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as pool:
            pool.map(partial(_process_replicate, args), csvs)
    else:
        for f in csvs:
            _process_replicate(args, f)

    # Grouping + ensemble plots (use same params as per-file analysis)
    run_ensemble(
        args.work_dir,
        filter_D_min=args.filter_D_min,
        filter_D_max=args.filter_D_max,
        filter_alpha_min=args.filter_alpha_min,
        filter_alpha_max=args.filter_alpha_max,
        time_step=args.time_step,
        micron_per_px=args.micron_per_px,
        tlag_cutoff=args.tlag_cutoff,
        min_track_len=args.min_track_len,
    )

    compare_conditions(
        args.work_dir,
        filter_D_min=args.filter_D_min,
        filter_D_max=args.filter_D_max,
        filter_alpha_min=args.filter_alpha_min,
        filter_alpha_max=args.filter_alpha_max,
    )


if __name__ == "__main__":
    main()
