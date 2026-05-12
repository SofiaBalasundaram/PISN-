#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:57:17 2026

@author: sofiabalasundaram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/sofiabalasundaram/Desktop/Bachelor Thesis')

# load the three datasets
features      = pd.read_parquet("classified_SLSNe_dataset_alerts.parquet")
pisn_features = pd.read_csv("simulated_features.csv")
ibb_features  = pd.read_csv("SN2018ibb_RainbowFit_Features.csv")

# the 5 RainbowFit features I'm using to compare populations
fnames = ['rise_time', 'fall_time', 'Tmin', 'Tmax', 't_color']

# simplify labels to just SLSN or other
features['Label'] = features.apply(
    lambda x: "SLSN" if "SLSN" in x['label'] else "other", axis=1
)
features = features.sort_values(by='Label')

pisn_features['Label'] = 'PISN simulation'
ibb_features['Label']  = 'SN 2018ibb'

# plot each consecutive pair of features against each other
# plotting other first so SLSN and PISN points appear on top
for i in range(len(fnames) - 1):
    f1 = fnames[i]
    f2 = fnames[i + 1]

    other = features[features['Label'] == 'other'][[f1, f2]].copy().dropna()
    slsn  = features[features['Label'] == 'SLSN'][[f1, f2]].copy().dropna()
    pisn  = pisn_features[[f1, f2]].copy().dropna()
    ibb   = ibb_features[[f1, f2]].copy().dropna()

    # log scale so the huge range of values is easier to see
    for df in [other, slsn, pisn, ibb]:
        df[f1] = np.log10(df[f1].abs())
        df[f2] = np.log10(df[f2].abs())

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(other[f1], other[f2], color='#d3d3d3', s=2, alpha=0.5, label='other')
    ax.scatter(slsn[f1],  slsn[f2],  color='#15284f', s=4, alpha=0.8, label='SLSN')
    ax.scatter(pisn[f1],  pisn[f2],  color='#2ecc71', s=2, alpha=0.6, label='PISN simulation')

    # SN 2018ibb plotted as a star 
    ax.scatter(ibb[f1], ibb[f2], color='gold', s=300, marker='*',
               zorder=10, label='SN 2018ibb', edgecolors='black', linewidths=0.5)

    ax.set_title(f"{f1} vs {f2}", fontsize=14)
    ax.set_xlabel(f"log10({f1})", fontsize=14)
    ax.set_ylabel(f"log10({f2})", fontsize=14)
    ax.legend(fontsize=10, markerscale=1)
    plt.tight_layout()
    plt.savefig(f"{f1}_{f2}.png", dpi=300, bbox_inches='tight')
    plt.show()