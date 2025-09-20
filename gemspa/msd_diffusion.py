#!/usr/bin/env python3
"""
msd_diffusion.py

Optimized MSD and diffusion analysis routines with Numba JIT and parallelism.
Also includes step-size & angle export helpers.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from joblib import Parallel, delayed
import numba
from numba import njit, prange

@njit(parallel=True)
def _msd2d_jit(x, y, max_lag):
    """Compute MSD for each lag up to max_lag."""
    n = x.shape[0]
    msd = np.zeros(max_lag)
    for lag in prange(1, max_lag + 1):
        total = 0.0
        count = 0
        for i in range(n - lag):
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            total += dx*dx + dy*dy
            count += 1
        msd[lag - 1] = total / count if count > 0 else 0.0
    return msd

@njit(parallel=True)
def _compute_step_sizes_jit(xs, ys):
    """Compute step sizes between consecutive points."""
    n = xs.shape[0]
    steps = np.zeros(n - 1)
    for i in prange(n - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        steps[i] = np.sqrt(dx*dx + dy*dy)
    return steps

class msd_diffusion:
    """
    MSD and diffusion coefficient analysis for trajectory data.
    Now also supports step-size and angle export.
    """
    def __init__(self, save_dir='.'):
        self.save_dir = save_dir
        self.time_step = 0.010
        self.micron_per_px = 0.11
        self.initial_guess_D = 0.2
        self.initial_guess_alpha = 1.0

        # For step-size export
        self.tracks = None
        self.track_lengths = None
        self.max_tlag_step_size = 5
        self.min_track_len_step_size = 3

    def fit_msd(self, msd_vals, time_step=None):
        """Fit MSD to power-law: MSD = 4*D*t^alpha."""
        t = np.arange(1, len(msd_vals) + 1) * (time_step or self.time_step)
        def model(t, D, alpha):
            return 4 * D * np.power(t, alpha)
        try:
            popt, _ = curve_fit(
                model, t, msd_vals,
                p0=[self.initial_guess_D, self.initial_guess_alpha],
                bounds=(0, np.inf)
            )
            D_fit, alpha_fit = popt
        except Exception:
            D_fit, _, alpha_fit = self.fit_msd_linear(msd_vals, time_step)
            return D_fit, alpha_fit, 0.0
        fit_vals = model(t, D_fit, alpha_fit)
        residuals = msd_vals - fit_vals
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_vals - np.mean(msd_vals))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        return D_fit, alpha_fit, r2

    def fit_msd_linear(self, msd_vals, time_step=None):
        """Linear MSD fit: MSD = 4*D*t."""
        t = np.arange(1, len(msd_vals) + 1) * (time_step or self.time_step)
        def lin_fn(t, D):
            return 4 * D * t
        popt, _ = curve_fit(lin_fn, t, msd_vals, p0=[self.initial_guess_D])
        D_fit = popt[0]
        fit_vals = lin_fn(t, D_fit)
        residuals = msd_vals - fit_vals
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_vals - np.mean(msd_vals))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        return D_fit, 1.0, r2

    # === New step-size export helpers ===
    def set_track_data(self, arr):
        """Store raw track array [[id,frame,x,y],…] and compute lengths."""
        self.tracks = arr
        ids, counts = np.unique(arr[:,0], return_counts=True)
        self.track_lengths = np.vstack((ids, counts)).T

    def step_sizes_and_angles(self):
        """Compute step sizes and angles for export."""
        valid = self.track_lengths[self.track_lengths[:,1] >= self.min_track_len_step_size]
        ids = valid[:,0].astype(int)
        if ids.size == 0:
            self.step_sizes = np.empty((0,0))
            self.angles = np.empty((0,0))
            return

        lengths = valid[:,1].astype(int)
        total_steps_t1 = int(np.sum(lengths - 1))
        M = self.max_tlag_step_size
        self.step_sizes = np.full((M, total_steps_t1), np.nan)
        self.deltaX = np.full((M, total_steps_t1), np.nan)
        self.deltaY = np.full((M, total_steps_t1), np.nan)
        total_angles = int(np.sum(lengths - 2))
        self.angles = np.full((M, total_angles), np.nan)

        pos_steps = np.zeros(M, dtype=int)
        pos_angles = np.zeros(M, dtype=int)

        for tid, length in zip(ids, lengths):
            track = self.tracks[self.tracks[:,0] == tid]
            x = track[:,2]
            y = track[:,3]
            max_shifts = min(M, len(x) - 1)
            max_angles = min(M, (len(x) - 1) // 2)
            for lag in range(1, max_shifts + 1):
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                steps = np.sqrt(dx*dx + dy*dy)
                count = steps.size
                i = lag - 1
                self.step_sizes[i, pos_steps[i]:pos_steps[i]+count] = steps
                self.deltaX[i, pos_steps[i]:pos_steps[i]+count] = dx
                self.deltaY[i, pos_steps[i]:pos_steps[i]+count] = dy
                pos_steps[i] += count
                if lag <= max_angles:
                    angles = []
                    for j in range(0, len(dx) - lag, lag):
                        v1 = np.array([dx[j], dy[j]])
                        v2 = np.array([dx[j+lag], dy[j+lag]])
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 == 0 or norm2 == 0:
                            continue
                        cosang = np.dot(v1, v2) / (norm1 * norm2)
                        cosang = np.clip(cosang, -1, 1)
                        angles.append(np.degrees(np.arccos(cosang)))
                    ang_arr = np.array(angles)
                    cnta = ang_arr.size
                    self.angles[lag-1, pos_angles[lag-1]:pos_angles[lag-1]+cnta] = ang_arr
                    pos_angles[lag-1] += cnta

    def save_step_sizes(self, file_name='step_sizes.txt'):
        """Write the step_sizes array to a tab-delimited file and return a DataFrame."""
        M, N = self.step_sizes.shape
        data = {'t': np.arange(1, M+1)}
        for col in range(N):
            data[str(col)] = self.step_sizes[:,col]
        df = pd.DataFrame(data)
        path = os.path.join(self.save_dir, file_name)
        df.to_csv(path, sep='\t', index=False)
        return df

