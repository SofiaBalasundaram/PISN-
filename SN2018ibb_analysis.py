#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:30:28 2026
@author: sofiabalasundaram

Analysis of SN 2018ibb — the reference PISN candidate.
Loads the photometric data, computes absolute magnitudes, fits RainbowFit,
and saves the extracted features for use in the simulation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from light_curve.light_curve_py import RainbowFit
from astropy.constants import c, h, k_B
from astropy import units as u
from fink_utils.photometry.conversion import mag2fluxcal_snana
from astropy.cosmology import Planck18 as cosmo

# redshift of SN 2018ibb
z_0 = 0.166

# effective wavelengths for g and r bands in angstrom
band_wave_aa = {'g': 4770.0, 'r': 6231.0}

# load the photometric data
df = pd.read_csv("2018ibb_Photometric_Data.csv")
data = df.copy()

# split by filter for computing absolute magnitudes separately
zr_data = df[df['filter'] == 'zr'].copy()
zg_data = df[df['filter'] == 'zg'].copy()

def distance_modulus(z):
    # using Planck18 cosmology — more accurate than the low-redshift approximation
    d_l_pc = cosmo.luminosity_distance(z).to(u.pc).value
    return 5 * np.log10(d_l_pc / 10)

def calc_abs_mag_DM(apparent_mag, z):
    DM = distance_modulus(z)
    return apparent_mag - DM + 2.5 * np.log10(1 + z)

# compute absolute magnitudes
zg_data['abs_mag'] = calc_abs_mag_DM(zg_data['mag'], z_0)
zr_data['abs_mag'] = calc_abs_mag_DM(zr_data['mag'], z_0)

# find peak brightness — minimum magnitude = brightest point
idx_zg = zg_data['abs_mag'].idxmin()
idx_zr = zr_data['abs_mag'].idxmin()

min_mag_zg = zg_data.loc[idx_zg, 'abs_mag']
t_max_zg   = zg_data.loc[idx_zg, 'mjd']

min_mag_zr = zr_data.loc[idx_zr, 'abs_mag']
t_max_zr   = zr_data.loc[idx_zr, 'mjd']

print("Peak absolute magnitude (g-band):", min_mag_zg)
print("Time of peak (g-band):", t_max_zg)
print("Peak absolute magnitude (r-band):", min_mag_zr)
print("Time of peak (r-band):", t_max_zr)

# convert to rest-frame time — removes time dilation from cosmic expansion
zg_data['t_rest'] = (zg_data['mjd'] - t_max_zg) / (1 + z_0)
zr_data['t_rest'] = (zr_data['mjd'] - t_max_zr) / (1 + z_0)

zg_data['abs_mag_err'] = zg_data['magerr']
zr_data['abs_mag_err'] = zr_data['magerr']

# plot the r-band light curve
plt.figure(figsize=(8, 5))
plt.plot(zr_data['t_rest'], zr_data['abs_mag'], color='red', label='zr')
plt.errorbar(zr_data['t_rest'], zr_data['abs_mag'],
             yerr=zr_data['abs_mag_err'], fmt='none', ecolor='black', alpha=0.9)
plt.gca().invert_yaxis()
plt.xlabel('Days Since Maximum')
plt.ylabel('Absolute Magnitude')
plt.title('Light Curve (R-band)')
plt.legend()
plt.show()

# plot the g-band light curve
plt.figure(figsize=(8, 5))
plt.plot(zg_data['t_rest'], zg_data['abs_mag'], color='green', label='zg')
plt.errorbar(zg_data['t_rest'], zg_data['abs_mag'],
             yerr=zg_data['abs_mag_err'], fmt='none', ecolor='black')
plt.gca().invert_yaxis()
plt.xlabel('Days Since Maximum')
plt.ylabel('Absolute Magnitude')
plt.title('Light Curve (G-band)')
plt.legend()
plt.show()

# prepare data for RainbowFit
data['band'] = data['filter'].map({'zg': 'g', 'zr': 'r'})
flux, flux_err = mag2fluxcal_snana(data['mag'].values, data['magerr'].values)
t    = data['mjd'].values
band = data['band'].values

# fit RainbowFit to the real 2018ibb light curve
rainbow_model = RainbowFit.from_angstrom(
    band_wave_aa, with_baseline=False, temperature='sigmoid', bolometric='bazin'
)
values = rainbow_model(t, flux, sigma=flux_err, band=band)

# print the fitted parameters
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:
        print(f"{name:15}: {val:.2e}")
    else:
        print(f"{name:15}: {val:.2f}")

# unpack for use in later scripts
reference_time = values[0]
amplitude      = values[1]
rise_time      = values[2]
fall_time      = values[3]
Tmin           = values[4]
Tmax           = values[5]
t_color        = values[6]

# save features to CSV — this gets loaded in KDEClassifier.py
feature_dict = {}
for name, val in zip(rainbow_model.names, values):
    feature_dict[name] = f"{val:.2e}" if abs(val) < 1e-2 else round(val, 2)

pd.DataFrame([feature_dict]).to_csv("SN2018ibb_RainbowFit_Features.csv", index=False)
print("Features saved to SN2018ibb_RainbowFit_Features.csv")

# evaluate the model on a dense time grid for smooth plotting
smooth_time   = np.linspace(58350, 58750, 100)
smooth_flux_g = rainbow_model.model(smooth_time, np.repeat('g', len(smooth_time)), *values[:-1])
smooth_flux_r = rainbow_model.model(smooth_time, np.repeat('r', len(smooth_time)), *values[:-1])

# use the smooth model peak as t0 — avoids any noisy spike in the raw data
peak_idx  = np.argmax(smooth_flux_g)
t0_model  = smooth_time[peak_idx]

# plot the RainbowFit on top of the data
plt.figure(figsize=(8, 5))
mask_g = band == 'g'
mask_r = band == 'r'
plt.errorbar(t[mask_g], flux[mask_g], yerr=flux_err[mask_g],
             fmt='o', color='darkgreen', label='g-band data', alpha=0.7)
plt.errorbar(t[mask_r], flux[mask_r], yerr=flux_err[mask_r],
             fmt='o', color='orange', label='r-band data', alpha=0.7)
plt.plot(smooth_time, smooth_flux_g, '-', color='green', label='g-band fit')
plt.plot(smooth_time, smooth_flux_r, '-', color='red', label='r-band fit')
plt.xlabel("MJD")
plt.ylabel("Flux (relative)")
plt.title("SN 2018ibb RainbowFit")
plt.legend()
plt.savefig("SN2018ibb_RainbowFit.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()