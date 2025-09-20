#!/usr/bin/env python3
"""
compare_conditions.py

Generates comparative plots across experimental conditions using the single
grouped_filtered/msd_results.csv written by ensemble_analysis.run_ensemble():

- Log-scale x-axis, density-normalized, overlaid histograms of filtered D_fit
  with mean-lines and a KS-test asterisk.
- Linear-scale histogram of alpha_fit.
- Boxplot of replicate median D_fit with jittered points and a KS-test asterisk.
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


def _p_to_asterisks(p):
    if p < 1e-4:   return "****"
    if p < 1e-3:   return "***"
    if p < 1e-2:   return "**"
    if p < 5e-2:   return "*"
    return "n.s."


def compare_conditions(root_dir,
                       filter_D_min=0.0, filter_D_max=float('inf'),
                       filter_alpha_min=0.0, filter_alpha_max=float('inf')):
    # ---- Load grouped_filtered data once and split by condition ----
    path = os.path.join(root_dir, 'grouped_filtered', 'msd_results.csv')
    if not os.path.isfile(path):
        print(f"[compare] grouped_filtered/msd_results.csv not found in {root_dir}; skipping.")
        return
    gfil = pd.read_csv(path)
    req = {'condition', 'D_fit', 'alpha_fit'}
    if not req.issubset(gfil.columns):
        print(f"[compare] Required columns missing: {req - set(gfil.columns)}; skipping.")
        return

    # Apply numeric filter guardrails (optional)
    gfil = gfil.replace([np.inf, -np.inf], np.nan).dropna(subset=['D_fit', 'alpha_fit'])
    if filter_D_min > 0:
        gfil = gfil[gfil['D_fit'] >= filter_D_min]
    if np.isfinite(filter_D_max):
        gfil = gfil[gfil['D_fit'] <= filter_D_max]
    gfil = gfil[gfil['alpha_fit'].between(filter_alpha_min, filter_alpha_max)]

    cond_map = {c: df[['D_fit', 'alpha_fit']].copy() for c, df in gfil.groupby('condition')}
    conds = list(cond_map.keys())
    if len(conds) < 2:
        print("[compare] Need at least 2 conditions; skipping.")
        return

    # Output folder
    comp_dir = os.path.join(root_dir, 'comparison')
    os.makedirs(comp_dir, exist_ok=True)

    # Palette
    colors = sns.color_palette(n_colors=len(conds))

    # -------------------------------
    # D_fit histogram (log x-axis)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # data-driven bins
    allD = pd.concat([cond_map[c]['D_fit'] for c in conds]).replace([np.inf, -np.inf], np.nan).dropna()
    allD = allD[allD > 0]
    if allD.empty:
        print("[compare] No positive D_fit values; skipping D histogram.")
    else:
        lo = max(allD.min(), max(filter_D_min, 1e-6))
        hi = min(allD.max(), filter_D_max if np.isfinite(filter_D_max) else allD.max())
        if hi <= lo:
            hi = lo * 1.1
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

        means = {}
        for c, col in zip(conds, colors):
            df = cond_map[c]
            ax.hist(df['D_fit'], bins=bins, density=True, alpha=0.5, color=col, label=c)
            mval = df['D_fit'].mean()
            means[c] = mval
            darker = tuple(max(0, x * 0.7) for x in col)
            ax.axvline(mval, color=darker, linewidth=2)

        ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.set_xlabel('D_fit (μm²/s), log scale')
        ax.set_ylabel('Density')
        ax.set_title('Ensemble Filtered D_fit Distributions')
        ax.legend()

        # KS-test (first two conditions)
        d1 = cond_map[conds[0]]['D_fit']
        d2 = cond_map[conds[1]]['D_fit']
        p = ks_2samp(d1, d2).pvalue
        stars = _p_to_asterisks(p)
        y_max = ax.get_ylim()[1]
        y_base = y_max * 0.90
        y_tip  = y_base * 1.02
        x1 = means[conds[0]]
        x2 = means[conds[1]]
        ax.plot([x1, x1, x2, x2], [y_base, y_tip, y_tip, y_base], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y_tip * 1.01, stars, ha='center', va='bottom')

        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, 'ensemble_filtered_D_histograms.png'))
        plt.close(fig)

    # -------------------------------
    # alpha_fit histogram (linear)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    means = {}
    for c, col in zip(conds, colors):
        df = cond_map[c]
        ax.hist(df['alpha_fit'], bins=50, density=True, alpha=0.5, color=col, label=c)
        mval = df['alpha_fit'].mean()
        means[c] = mval
        darker = tuple(max(0, x * 0.7) for x in col)
        ax.axvline(mval, color=darker, linewidth=2)

    ax.set_xlabel('alpha_fit')
    ax.set_ylabel('Density')
    ax.set_title('Ensemble Filtered α Distributions')
    ax.legend()

    # KS-test (first two conditions)
    a1 = cond_map[conds[0]]['alpha_fit']
    a2 = cond_map[conds[1]]['alpha_fit']
    p = ks_2samp(a1, a2).pvalue
    stars = _p_to_asterisks(p)
    y_max = ax.get_ylim()[1]
    y_base = y_max * 0.90
    y_tip  = y_base * 1.02
    x1 = means[conds[0]]
    x2 = means[conds[1]]
    ax.plot([x1, x1, x2, x2], [y_base, y_tip, y_tip, y_base], lw=1.5, color='black')
    ax.text((x1 + x2) / 2, y_tip * 1.01, stars, ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'ensemble_filtered_alpha_histograms.png'))
    plt.close(fig)

    # -------------------------------
    # Boxplot of replicate median D_fit (hue fix)
    # -------------------------------
    med_records = []
    rep_rx = re.compile(r'(.+)_\d+$')
    for sub in os.listdir(root_dir):
        m = rep_rx.match(sub)
        if not m:
            continue
        cond = m.group(1)
        p_rep = os.path.join(root_dir, sub, 'msd_results.csv')
        if not os.path.isfile(p_rep):
            continue
        dfr = pd.read_csv(p_rep, usecols=['D_fit'])
        dfr = dfr.replace([np.inf, -np.inf], np.nan).dropna()
        if filter_D_min > 0:
            dfr = dfr[dfr['D_fit'] >= filter_D_min]
        if np.isfinite(filter_D_max):
            dfr = dfr[dfr['D_fit'] <= filter_D_max]
        med_records.append({'condition': cond, 'median_D': dfr['D_fit'].median()})

    med_df = pd.DataFrame(med_records)
    if med_df.empty:
        print("[compare] No replicate median data for boxplot; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    order = sorted(med_df['condition'].unique())
    pal = dict(zip(order, sns.color_palette(n_colors=len(order))))

    # ➜ Fix: pass hue='condition' to avoid seaborn 0.14 warning
    sns.boxplot(x='condition', y='median_D', data=med_df, order=order, ax=ax,
                showfliers=False, palette=pal, hue='condition', dodge=False)
    # remove redundant legend created by hue
    leg = ax.get_legend()
    if leg: leg.remove()

    sns.stripplot(x='condition', y='median_D', data=med_df, order=order,
                  color='black', size=6, jitter=True, ax=ax)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Median D_fit (μm²/s)')
    ax.set_title('Replicate Median D_fit by Condition')

    # KS-test on pooled D (first two conditions)
    d1 = cond_map[conds[0]]['D_fit']
    d2 = cond_map[conds[1]]['D_fit']
    p = ks_2samp(d1, d2).pvalue
    stars = _p_to_asterisks(p)
    y_max = med_df['median_D'].max()
    y_base = y_max * 1.05
    y_tip  = y_base * 1.02
    ax.plot([0, 0, 1, 1], [y_base, y_tip, y_tip, y_base], lw=1.5, color='black')
    ax.text(0.5, y_tip * 1.01, stars, ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'replicate_median_D_boxplot.png'))
    plt.close(fig)
