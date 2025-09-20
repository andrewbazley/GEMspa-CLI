#!/usr/bin/env python3
# gemspa/rainbow_tracks.py

import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import io, draw

def draw_rainbow_tracks(
    image_path,
    raw_df,
    results_df,
    output_path,
    id_col='track_id',
    x_col='x',
    y_col='y',
    D_col='D_fit',
    min_D=0.0,
    max_D=2.0,
    line_width=0.01,
    colormap='viridis',
    scale=4.0,
    dpi=1000
):
    """
    Overlay tracks on the background image, color-coded by diffusion coefficient,
    zoomed in by `scale` and clamped to [min_D, max_D] for the LUT.
    """
    # 1) Load image and build RGB canvas
    img = io.imread(image_path)
    if img.ndim == 2:
        canvas = np.stack([img]*3, axis=-1)
    else:
        canvas = img.copy()
    h, w = canvas.shape[:2]

    # 2) Normalize & colormap
    D_vals = np.clip(results_df[D_col].astype(float), min_D, max_D)
    norm = plt.Normalize(vmin=min_D, vmax=max_D)
    cmap = cm.get_cmap(colormap)

    # 3) Draw tracks
    raw_df[id_col]   = raw_df[id_col].astype(str)
    results_df[id_col] = results_df[id_col].astype(str)

    for _, row in results_df.iterrows():
        tid   = str(row[id_col])
        D_val = np.clip(float(row[D_col]), min_D, max_D)
        color = cmap(norm(D_val))[:3]
        track = raw_df[raw_df[id_col] == tid]
        coords = track[[x_col, y_col]].to_numpy()
        for i in range(len(coords)-1):
            y0, x0 = coords[i,1], coords[i,0]
            y1, x1 = coords[i+1,1], coords[i+1,0]
            rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
            canvas[rr, cc] = (np.array(color)*255).astype(np.uint8)

    # 4) Create a big figure via subplots
    fig_w = (w / 100.0) * scale
    fig_h = (h / 100.0) * scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # 5) Display and remove margins
    ax.axis('off')
    ax.imshow(canvas, interpolation='nearest')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 6) Save high-res, tightly cropped
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
