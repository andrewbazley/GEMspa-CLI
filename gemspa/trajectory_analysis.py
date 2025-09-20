#!/usr/bin/env python3
# gemspa/trajectory_analysis.py

import os
import re
import glob
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
from numba import njit, prange

from .msd_diffusion import msd_diffusion
from .rainbow_tracks import draw_rainbow_tracks

class trajectory_analysis:
    """
    MSD, diffusion, rainbow overlay, step-size export, and (NEW)
    per-file ensemble MSD vs tau plots (linear & log-log).
    """

    def __init__(
        self,
        data_file,
        results_dir='.',
        condition=None,
        time_step=0.010,
        micron_per_px=0.11,
        ts_resolution=0.005,
        min_track_len_linfit=11,
        tlag_cutoff_linfit=10,
        make_rainbow_tracks=False,
        img_file_prefix='MAX_',
        rainbow_min_D=0.0,
        rainbow_max_D=2.0,
        rainbow_colormap='viridis',
        rainbow_scale=1.0,
        rainbow_dpi=200,
        n_jobs=1,
        threads_per_rep=None,
        log_file=None
    ):
        self.n_jobs = n_jobs
        self.threads_per_rep = threads_per_rep or max(1, cpu_count() // max(1, n_jobs))

        # core parameters
        self.data_file            = data_file
        self.results_dir          = results_dir
        base = os.path.splitext(os.path.basename(self.data_file))[0]
        self.condition            = condition or re.sub(r'_[0-9]+$', '', base)
        self.time_step            = time_step
        self.micron_per_px        = micron_per_px
        self.ts_resolution        = ts_resolution
        self.min_track_len_linfit = min_track_len_linfit
        self.tlag_cutoff_linfit   = tlag_cutoff_linfit
        self.make_rainbow_tracks  = make_rainbow_tracks
        self.img_prefix           = img_file_prefix
        self.rainbow_min_D        = rainbow_min_D
        self.rainbow_max_D        = rainbow_max_D
        self.rainbow_colormap     = rainbow_colormap
        self.rainbow_scale        = rainbow_scale
        self.rainbow_dpi          = rainbow_dpi
        self.rainbow_line_width   = 0.1

        os.makedirs(self.results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logn = log_file or f"{base}_{ts}.log"
        self.log = open(os.path.join(self.results_dir, logn), 'w')

        # ---- load & sanitize input CSV (TrackMate-friendly) ----
        df = pd.read_csv(data_file, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]
        alias_map = {}
        if 'trajectory'   in df.columns: alias_map['trajectory']   = 'track_id'
        if 'position_x'   in df.columns and 'x'     not in df.columns: alias_map['position_x'] = 'x'
        if 'position_y'   in df.columns and 'y'     not in df.columns: alias_map['position_y'] = 'y'
        if 'spot_frame'   in df.columns and 'frame' not in df.columns: alias_map['spot_frame'] = 'frame'
        if alias_map:
            df = df.rename(columns=alias_map)
        for req in ('track_id','frame','x','y'):
            if req not in df.columns:
                raise KeyError(f"Input CSV missing required column '{req}'. Found columns: {list(df.columns)}")

        self.raw_df = df.copy()
        self.raw_df['condition'] = self.condition

        self.msd_processor = msd_diffusion(save_dir=self.results_dir)

    @staticmethod
    @njit(parallel=True)
    def _compute_msd(coords, max_lag):
        n = coords.shape[0]
        out = np.zeros(max_lag)
        for lag in prange(1, max_lag+1):
            d = coords[lag:] - coords[:n-lag]
            out[lag-1] = np.mean(d[:,0]**2 + d[:,1]**2) if d.size else 0.0
        return out

    def _one_track(self, grp):
        coords = grp[['x','y']].to_numpy() * self.micron_per_px
        max_lag = min(self.tlag_cutoff_linfit, max(1, coords.shape[0]-1))
        msd_v = self._compute_msd(coords, max_lag)
        return msd_v

    def _fit_D_alpha_from_loglog(self, tau, msd):
        # log10(MSD) = log10(4D) + alpha * log10(tau)
        m = (tau>0) & (msd>0)
        if m.sum() < 2:
            return 0.0, 0.0
        x = np.log10(tau[m]); y = np.log10(msd[m])
        a, b = np.polyfit(x, y, 1)  # slope, intercept
        alpha = a
        D = (10.0**b)/4.0
        return float(D), float(alpha)

    def calculate_msd_and_diffusion(self):
        """
        NEW: computes per-file ensemble MSD arrays and saves linear & log-log plots.
        Also saves per-track D/alpha distributions as before.
        """
        # collect per-track MSDs
        tracks = [
            grp for _, grp in self.raw_df.groupby('track_id')
            if len(grp) >= self.min_track_len_linfit
        ]
        if len(tracks) == 0:
            raise RuntimeError("No tracks pass min length filter.")

        with parallel_backend('threading'):
            msd_list = Parallel(n_jobs=self.threads_per_rep)(
                delayed(self._one_track)(grp) for grp in tracks
            )

        # choose a common max lag across tracks
        L = min(len(v) for v in msd_list)
        msd_mat = np.vstack([v[:L] for v in msd_list])  # shape: (n_tracks, L)
        tau = np.arange(1, L+1, dtype=float) * self.time_step
        ens_msd = np.nanmean(msd_mat, axis=0)

        # fit D from linear MSD (2D: MSD = 4 D t) and alpha/D from log-log
        # linear slope by least squares on (tau, ens_msd)
        m_lin = (ens_msd>0)
        if m_lin.sum() >= 2:
            slope, intercept = np.polyfit(tau[m_lin], ens_msd[m_lin], 1)
            D_lin = max(0.0, slope/4.0)
        else:
            D_lin = 0.0
        D_ll, alpha_ll = self._fit_D_alpha_from_loglog(tau, ens_msd)

        # ----- SAVE: ensemble MSD plots for this file -----
        self._plot_ens_msd_linear(tau, ens_msd, D_lin, out=os.path.join(self.results_dir, 'msd_vs_tau.png'))
        self._plot_ens_msd_loglog(tau, ens_msd, alpha_ll, out=os.path.join(self.results_dir, 'msd_vs_tau_loglog.png'))

        # ----- per-track fit results for hist/scatter (unchanged outputs) -----
        # Reuse track MSDs to fit each track with your existing fitter
        results = []
        for grp, msd_v in zip(tracks, msd_list):
            # fit with your msd_diffusion helper (keeps behavior consistent with existing pipeline)
            res = self.msd_processor.fit_msd(msd_v, self.time_step)  # returns (D, alpha, r2)
            results.append((grp['track_id'].iloc[0],) + res)

        ids, D_vals, alpha_vals, r2_vals = zip(*results)
        self.results_df = pd.DataFrame({
            'track_id':  ids,
            'condition': self.condition,
            'D_fit':     D_vals,
            'alpha_fit': alpha_vals,
            'r2_fit':    r2_vals
        })
        self.results_df.to_csv(os.path.join(self.results_dir, 'msd_results.csv'), index=False)

        self.make_plot()
        self.make_scatter()

        if self.make_rainbow_tracks:
            base = os.path.splitext(os.path.basename(self.data_file))[0]
            rep  = base.replace('Traj_', '') if base.startswith('Traj_') else base
            cond = self.condition
            rep_num = rep.rsplit('_',1)[-1]
            patterns = [
                f"{self.img_prefix}{cond}_{rep_num}.tif",
                f"{self.img_prefix}{cond}.tif",
                f"{self.img_prefix}{cond}*.tif",
            ]
            matches = []
            for pat in patterns:
                matches += glob.glob(os.path.join(os.path.dirname(self.data_file), pat))
            matches = sorted(set(matches))
            if matches:
                draw_rainbow_tracks(
                    image_path=matches[0],
                    raw_df=self.raw_df,
                    results_df=self.results_df,
                    output_path=os.path.join(self.results_dir, 'rainbow_tracks.png'),
                    min_D=self.rainbow_min_D,
                    max_D=self.rainbow_max_D,
                    colormap=self.rainbow_colormap,
                    scale=self.rainbow_scale,
                    dpi=self.rainbow_dpi,
                    line_width=self.rainbow_line_width
                )
            else:
                print(f"[rainbow] no background image for {rep} "
                      f"(looked for {patterns} in {os.path.dirname(self.data_file)})")
                      

    def _plot_ens_msd_linear(self, tau, msd, D_est, out):
        fig, ax = plt.subplots(figsize=(10.0,5.8))
        ax.plot(tau, msd, 'o', ms=3)
        ax.set_xlabel('τ (s)', fontsize=9)
        ax.set_ylabel('MSD (μm²)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title('ens-avg MSD (2d)\nD = {:.3g} μm²/s'.format(D_est), fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)

    def _plot_ens_msd_loglog(self, tau, msd, alpha_est, out):
        m = (tau>0) & (msd>0)
        fig, ax = plt.subplots(figsize=(10.0,5.8))
        ax.plot(np.log10(tau[m]), msd[m], 'o', ms=3)
        ax.set_yscale('log')
        ax.set_xlabel('log10 τ (s)', fontsize=9)
        ax.set_ylabel('MSD (μm²), log scale', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title('ens-avg log-log MSD (2d)\nα = {:.4f}'.format(alpha_est), fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)

    def export_step_sizes(self, max_tlag=None):
        df = self.raw_df[['track_id','frame','x','y']].copy()
        df['x'] *= self.micron_per_px
        df['y'] *= self.micron_per_px
        arr = df.sort_values(['track_id','frame']).to_numpy()
        self.msd_processor.set_track_data(arr)
        if max_tlag is not None:
            self.msd_processor.max_tlag_step_size = max_tlag
        self.msd_processor.step_sizes_and_angles()
        ss = self.msd_processor.save_step_sizes(file_name='all_data_step_sizes.txt')
        ss = ss.rename(columns={'t': 'tlag'})
        ss.insert(1, 'group', self.condition)
        out = os.path.join(self.results_dir, 'all_data_step_sizes.txt')
        ss.to_csv(out, sep='\t', index=False)

    def make_plot(self):
        fig, ax = plt.subplots(figsize=(8,5))
        d = self.results_df['D_fit']
        dpos = d[d > 0]
        if len(dpos) == 0:
            bins = 30
        else:
            lo, hi = dpos.min(), d.max()
            if lo <= 0 or not np.isfinite(lo):
                bins = 30
            else:
                bins = np.logspace(np.log10(lo), np.log10(hi if hi>lo else lo*1.1), 30)
                ax.set_xscale('log')
        ax.hist(d, bins=bins, edgecolor='black')
        ax.set_xlabel('D_fit (μm²/s)' + (' (log scale)' if len(dpos) else ''))
        ax.set_title(f"D_fit Distribution ({self.condition})")
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, 'D_fit_distribution.png'))
        plt.close(fig)

    def make_scatter(self):
        fig, ax = plt.subplots(figsize=(8,5))
        d = self.results_df['D_fit'].replace({0: np.nan})
        ax.scatter(np.log10(d), self.results_df['alpha_fit'], alpha=0.6)
        ax.set_xlabel('log10(D_fit)')
        ax.set_ylabel('alpha_fit')
        ax.set_title(f"alpha vs log D ({self.condition})")
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, 'alpha_vs_logD.png'))
        plt.close(fig)

    def write_params_to_log_file(self):
        params = {
            'condition':             self.condition,
            'time_step':             self.time_step,
            'micron_per_px':         self.micron_per_px,
            'min_track_len_linfit':  self.min_track_len_linfit,
            'tlag_cutoff_linfit':    self.tlag_cutoff_linfit
        }
        pd.Series(params).to_csv(os.path.join(self.results_dir,'params_log.csv'), header=False)
        for k, v in params.items():
            self.log.write(f"{k}: {v}\n")
        self.log.flush()
