#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:46:38 2026
@author: sofiabalasundaram

"""

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/Users/sofiabalasundaram/Desktop/Bachelor Thesis')

# these are the top 10 objects from the KDE scoring, ordered by log density
# (less negative = deeper inside the PISN region = more promising)
candidates = [
    ('ZTF21abrpizd', -4.643810, 'other'),
    ('ZTF24aayircu', -4.711866, 'other'),
    ('ZTF23aaisxic', -4.754224, 'other'),
    ('ZTF22abizuah', -4.822046, 'SLSN'),
    ('ZTF23aaczpkm', -4.990425, 'other'),
    ('ZTF21aciyirg', -5.231158, 'other'),
    ('ZTF22aadesjc', -5.239871, 'SLSN'),
    ('ZTF21aazgkjf', -5.253063, 'SLSN'),
    ('ZTF21aanuxvd', -5.272175, 'SLSN'),
    ('ZTF24aasteui', -5.482432, 'other'),
]

filter_colors = {1: 'green', 2: 'red'}
filter_labels = {1: 'g band', 2: 'r band'}

def fetch_lightcurve(objectId):
    # fetch light curve data from the Fink portal API
    r = requests.post(
        'https://api.ztf.fink-portal.org/api/v1/objects',
        json={'objectId': objectId, 'withupperlim': 'False', 'output-format': 'json'}
    )
    if r.status_code == 200:
        df = pd.read_json(io.BytesIO(r.content))
        return df
    else:
        print(f"Failed to fetch {objectId}: status {r.status_code}")
        return None

# plot all 10 in a grid
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, (objectId, score, label) in enumerate(candidates):
    ax = axes[idx]
    print(f"Fetching {objectId} ...")

    df = fetch_lightcurve(objectId)

    if df is None or df.empty:
        ax.text(0.5, 0.5, f"No data for {objectId}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{objectId}\n({label}, score: {score:.2f})", fontsize=9)
        continue

    # drop upper limits — they don't have a magpsf value
    valid = df[df['i:magpsf'].notna()].copy()

    for fid in [1, 2]:
        mask = valid['i:fid'] == fid
        if mask.sum() == 0:
            continue
        sub = valid[mask].sort_values('i:jd')
        ax.errorbar(
            sub['i:jd'] - sub['i:jd'].min(),
            sub['i:magpsf'],
            yerr=sub['i:sigmapsf'],
            fmt='o', markersize=3,
            color=filter_colors[fid],
            label=filter_labels[fid],
            alpha=0.8
        )

    ax.invert_yaxis()
    ax.set_title(f"{objectId}  |  {label}  |  score: {score:.3f}", fontsize=9)
    ax.set_xlabel("Days since first observation", fontsize=8)
    ax.set_ylabel("Magnitude", fontsize=8)
    ax.legend(fontsize=7)

plt.suptitle("Top 10 PISN Candidates — Light Curves from Fink", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("top10_pisn_candidates_lightcurves.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved top10_pisn_candidates_lightcurves.png")

# plot the top candidate on its own — cleaner for the thesis
print("\nPlotting top candidate ZTF22aadesjc ...")
df_top = fetch_lightcurve('ZTF22aadesjc')
valid_top = df_top[df_top['i:magpsf'].notna()].copy()

fig, ax = plt.subplots(figsize=(10, 6))
for fid in [1, 2]:
    sub = valid_top[valid_top['i:fid'] == fid].sort_values('i:jd')
    if sub.empty:
        continue
    ax.errorbar(
        sub['i:jd'] - sub['i:jd'].min(),
        sub['i:magpsf'],
        yerr=sub['i:sigmapsf'],
        fmt='o', markersize=4,
        color=filter_colors[fid],
        label=filter_labels[fid],
        alpha=0.8
    )

ax.invert_yaxis()
ax.set_xlabel("Days since first observation", fontsize=13)
ax.set_ylabel("Magnitude", fontsize=13)
ax.set_title("ZTF22aadesjc — Top PISN Candidate", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("ZTF22aadesjc_lightcurve.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved ZTF22aadesjc_lightcurve.png")

# after looking at the light curves manually, some candidates are clearly not
# good — noisy, too sparse, or no data at all. removing those here and saving
# a clean ranked list of the ones actually worth following up
excluded = {
    'ZTF24aayircu': 'no data in Fink',
    'ZTF21abrpizd': 'noisy and irregular, likely active galaxy',
    'ZTF23aaisxic': 'too sparse, insufficient data',
    'ZTF23aaczpkm': 'too sparse, insufficient data',
    'ZTF21aanuxvd': 'only ~60 days of coverage, too short to confirm',
}

promising = []
for objectId, score, label in candidates:
    if objectId in excluded:
        continue
    promising.append({'objectId': objectId, 'log_density': score, 'label': label})

df_promising = pd.DataFrame(promising)
df_promising = df_promising.sort_values('log_density', ascending=False).reset_index(drop=True)
df_promising.index += 1

print("\nMost promising PISN candidates:")
print(df_promising.to_string())

df_promising.to_csv("promising_pisn_candidates.csv", index_label='rank')
print("\nSaved promising_pisn_candidates.csv")