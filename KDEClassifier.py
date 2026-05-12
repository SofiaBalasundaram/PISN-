#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:28:18 2026

@author: sofiabalasundaram

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

os.chdir('/Users/sofiabalasundaram/Desktop/Bachelor Thesis')

# load the three datasets
features      = pd.read_parquet("classified_SLSNe_dataset_alerts.parquet")
pisn_features = pd.read_csv("simulated_features.csv")
ibb_features  = pd.read_csv("SN2018ibb_RainbowFit_Features.csv")

# simplify the labels to just SLSN or other
features['Label'] = features.apply(
    lambda x: "SLSN" if "SLSN" in x['label'] else "other", axis=1
)
pisn_features['Label'] = 'PISN simulation'
ibb_features['Label']  = 'SN 2018ibb'

# the 5 features from RainbowFit that describe each light curve
fnames = ['rise_time', 'fall_time', 'Tmin', 'Tmax', 't_color']

# log scale the features so large ranges don't dominate
def log_abs(df, cols):
    out = df[cols].copy()
    out = out.apply(lambda s: np.log10(s.abs()))
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

pisn_X  = log_abs(pisn_features, fnames)
slsn_X  = log_abs(features[features['Label'] == 'SLSN'],  fnames)
other_X = log_abs(features[features['Label'] == 'other'], fnames)
ibb_X   = log_abs(ibb_features, fnames)

# keep track of objectIds so we can retrieve candidates later
other_ids = features[features['Label'] == 'other'].loc[other_X.index, 'objectId'] \
    if 'objectId' in features.columns else None
slsn_ids  = features[features['Label'] == 'SLSN'].loc[slsn_X.index, 'objectId'] \
    if 'objectId' in features.columns else None

# standardise everything using the PISN distribution as reference
# KDE is distance based so all features need to be on the same scale
scaler = StandardScaler()
scaler.fit(pisn_X)

pisn_Xs  = scaler.transform(pisn_X)
slsn_Xs  = scaler.transform(slsn_X)
other_Xs = scaler.transform(other_X)
ibb_Xs   = scaler.transform(ibb_X)

# find the best bandwidth using cross validation
# bandwidth controls how smooth the density estimate is
print("Searching for optimal bandwidth …")
bandwidths = np.logspace(-1, 1, 20)
grid = GridSearchCV(
    KernelDensity(kernel='gaussian'),
    {'bandwidth': bandwidths},
    cv=3,
    n_jobs=-1
)
grid.fit(pisn_Xs)
best_bw = grid.best_params_['bandwidth']
print(f"Best bandwidth: {best_bw:.3f}")

# fit the KDE on the simulated PISN points only
kde = KernelDensity(kernel='gaussian', bandwidth=best_bw)
kde.fit(pisn_Xs)

# score every population — how PISN-like is each object?
log_dens_pisn  = kde.score_samples(pisn_Xs)
log_dens_slsn  = kde.score_samples(slsn_Xs)
log_dens_other = kde.score_samples(other_Xs)
log_dens_ibb   = kde.score_samples(ibb_Xs)

# set the threshold at the 5th percentile of PISN scores
# so 95% of simulated PISNe are above it
PERCENTILE = 5
threshold = np.percentile(log_dens_pisn, PERCENTILE)
print(f"KDE threshold (p{PERCENTILE} of PISN log-density): {threshold:.3f}")

# flag anything above the threshold as a PISN candidate
pisn_flags  = log_dens_pisn  >= threshold
slsn_flags  = log_dens_slsn  >= threshold
other_flags = log_dens_other >= threshold
ibb_flags   = log_dens_ibb   >= threshold

print(f"\nPISN above threshold  : {pisn_flags.sum()}/{len(pisn_flags)} ({100*pisn_flags.mean():.1f} %)")
print(f"SLSN above threshold  : {slsn_flags.sum()}/{len(slsn_flags)} ({100*slsn_flags.mean():.1f} %)")
print(f"Other above threshold : {other_flags.sum()}/{len(other_flags)} ({100*other_flags.mean():.1f} %)")
print(f"SN 2018ibb            : {'ABOVE' if ibb_flags.any() else 'BELOW'} threshold")

# save all scores to a csv
slsn_scores_df = pd.DataFrame({
    'objectId': slsn_ids.values if slsn_ids is not None else slsn_X.index,
    'log_density': log_dens_slsn,
    'is_PISN_candidate': slsn_flags,
    'Label': 'SLSN'
})

other_scores_df = pd.DataFrame({
    'objectId': other_ids.values if other_ids is not None else other_X.index,
    'log_density': log_dens_other,
    'is_PISN_candidate': other_flags,
    'Label': 'other'
})

all_scores = pd.concat([slsn_scores_df, other_scores_df], ignore_index=True)
all_scores.to_csv("kde_scores.csv", index=False)
print("\nAll scores saved to kde_scores.csv")

# print the top candidates
print("\n--- Top 10 SLSN candidates (highest KDE score) ---")
top_slsn = slsn_scores_df.sort_values('log_density', ascending=False).head(10)
print(top_slsn[['objectId', 'log_density']].to_string(index=False))

print("\n--- Top 10 Other candidates (highest KDE score) ---")
top_other = other_scores_df.sort_values('log_density', ascending=False).head(10)
print(top_other[['objectId', 'log_density']].to_string(index=False))

# plot the KDE frontier on top of the feature space plots
# since the KDE is 5D we project it down to 2D using Monte Carlo sampling
print("\nGenerating 2D feature plots with KDE frontier …")

pair_list = [(fnames[i], fnames[i+1]) for i in range(len(fnames)-1)]

for f1, f2 in pair_list:
    i1 = fnames.index(f1)
    i2 = fnames.index(f2)

    x_all = np.concatenate([pisn_Xs[:, i1], slsn_Xs[:, i1],
                             other_Xs[:, i1], ibb_Xs[:, i1]])
    y_all = np.concatenate([pisn_Xs[:, i2], slsn_Xs[:, i2],
                             other_Xs[:, i2], ibb_Xs[:, i2]])

    margin = 0.5
    x_grid = np.linspace(x_all.min() - margin, x_all.max() + margin, 80)
    y_grid = np.linspace(y_all.min() - margin, y_all.max() + margin, 80)
    xx, yy = np.meshgrid(x_grid, y_grid)

    other_dim_idx = [j for j in range(pisn_Xs.shape[1]) if j not in (i1, i2)]

    # randomly sample the other 3 dimensions from the PISN distribution
    np.random.seed(42)
    n_mc = 200
    mc_idx = np.random.choice(len(pisn_Xs), size=n_mc, replace=True)
    mc_other = pisn_Xs[mc_idx][:, other_dim_idx]

    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    N_grid = len(grid_pts)

    gp_rep = np.repeat(grid_pts, n_mc, axis=0)
    mc_rep = np.tile(mc_other, (N_grid, 1))

    full_pts = np.zeros((N_grid * n_mc, len(fnames)))
    full_pts[:, i1] = gp_rep[:, 0]
    full_pts[:, i2] = gp_rep[:, 1]
    for k, j in enumerate(other_dim_idx):
        full_pts[:, j] = mc_rep[:, k]

    log_d = kde.score_samples(full_pts)
    log_d_grid = log_d.reshape(N_grid, n_mc)
    log_d_marginal = np.log(np.mean(np.exp(log_d_grid - log_d_grid.max(axis=1, keepdims=True)), axis=1)) \
                     + log_d_grid.max(axis=1)
    zz = log_d_marginal.reshape(xx.shape)

    mu = scaler.mean_
    sigma = scaler.scale_

    def unscale(val, idx):
        return val * sigma[idx] + mu[idx]

    xx_data = unscale(xx, i1)
    yy_data = unscale(yy, i2)

    fig, ax = plt.subplots(figsize=(8, 6))

    cfill = ax.contourf(xx_data, yy_data, zz, levels=20,
                        cmap='YlOrRd', alpha=0.4)
    plt.colorbar(cfill, ax=ax, label='log density (marginalised)')

    # dashed line is the PISN frontier
    ax.contour(xx_data, yy_data, zz, levels=[threshold],
               colors='black', linewidths=1.5, linestyles='--')

    o_df = log_abs(features[features['Label'] == 'other'], fnames)
    s_df = log_abs(features[features['Label'] == 'SLSN'],  fnames)
    p_df = log_abs(pisn_features, fnames)
    i_df = log_abs(ibb_features,  fnames)

    ax.scatter(o_df[f1], o_df[f2], color='#d3d3d3', s=2,  alpha=0.5, label='other')
    ax.scatter(s_df[f1], s_df[f2], color='#15284f', s=4,  alpha=0.8, label='SLSN')
    ax.scatter(p_df[f1], p_df[f2], color='#2ecc71', s=2,  alpha=0.6, label='PISN simulation')
    ax.scatter(i_df[f1], i_df[f2], color='gold', s=300, marker='*',
               zorder=10, label='SN 2018ibb', edgecolors='black', linewidths=0.5)

    ax.set_xlabel(f"log10({f1})", fontsize=13)
    ax.set_ylabel(f"log10({f2})", fontsize=13)
    ax.set_title(f"{f1} vs {f2}  —  dashed line = KDE threshold (p{PERCENTILE})", fontsize=12)
    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(f"kde_{f1}_{f2}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved kde_{f1}_{f2}.png")

print("\nDone.")

# Find top KDE score of candidates 
print("\n--- Top 10 PISN candidates ---")
scores = pd.read_csv("kde_scores.csv")
candidates = scores[scores['is_PISN_candidate'] == True]
unique_candidates = candidates.drop_duplicates('objectId').sort_values('log_density', ascending=False)
print(unique_candidates[['objectId', 'log_density', 'Label']].head(10).to_string(index=False))