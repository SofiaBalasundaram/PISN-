#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:30:28 2026
@author: sofiabalasundaram

Takes the SN 2018ibb light curve from SN2018ibb_analysis.py and simulates
18,750 PISN-like light curves by varying redshift, temperature, stretch,
sampling and extinction. Then fits RainbowFit to each one to extract features.
The output is simulated_features.csv which feeds into KDEClassifier.py.

Run SN2018ibb_analysis.py first.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
import time
from astropy.constants import c, h, k_B
from astropy import units as u
from dust_extinction.parameter_averages import F99
from astropy.cosmology import Planck18 as cosmo
from light_curve.light_curve_py import RainbowFit

start = time.time()

# variables needed from SN2018ibb_analysis.py:
# z_0, t, flux, flux_err, band, band_wave_aa
# Tmin, Tmax, reference_time, t_color, t0_model, rainbow_model
# smooth_time, smooth_flux_g, smooth_flux_r

smooth_band          = np.array(['g'] * len(smooth_time) + ['r'] * len(smooth_time))
smooth_flux_combined = np.concatenate([smooth_flux_g, smooth_flux_r])
smooth_t_combined    = np.concatenate([smooth_time, smooth_time])
smooth_flux_err_combined = np.zeros(len(smooth_flux_combined))
t_ref = np.min(smooth_t_combined[smooth_band == 'g'])

# -------------------------------------------------------------------------#
# physical functions for transforming the light curve
# -------------------------------------------------------------------------#

def compute_temperature(t, Tmin, Tmax, t0, t_color):
    # sigmoid temperature evolution from RainbowFit
    delta_T = Tmax - Tmin
    return Tmin + delta_T / (1 + np.exp((t - t0) / t_color))

def planck(wavelength_aa, T):
    # Planck spectral radiance — wavelength in angstrom, temperature in kelvin
    lam = wavelength_aa * 1e-10
    exponent = (h.value * c.value) / (lam * k_B.value * T)
    return (2 * h.value * c.value**2 / lam**5) / (np.exp(exponent) - 1)

def redshift_temperature(T, z_old, z_new):
    # higher redshift = lower observed temperature
    return T * ((1 + z_old) / (1 + z_new))

def apply_snr_noise(flux, flux_err, snr_scale):
    # add gaussian noise scaled by SNR to mimic measurement uncertainty
    new_flux_err = flux_err / snr_scale
    noise = np.random.normal(0, new_flux_err)
    return flux + noise, new_flux_err

def redshift_snr(z_old, z_new):
    # SNR drops as 1/distance as the object gets further away
    d_old = cosmo.luminosity_distance(z_old).value
    d_new = cosmo.luminosity_distance(z_new).value
    return d_old / d_new

def apply_redshift(t, flux, flux_err, band, z_old, z_new,
                   Tmin, Tmax, t0, t_color, band_wave_aa):
    # shift to a new redshift — applies temperature correction,
    # Planck scaling, SNR noise and time dilation
    T     = compute_temperature(t, Tmin, Tmax, t0, t_color)
    T_red = redshift_temperature(T, z_old, z_new)

    wav_array = np.array([band_wave_aa[b] for b in band])
    ratio     = planck(wav_array, T_red) / planck(wav_array, T)

    new_flux     = flux * ratio
    new_flux_err = flux_err * ratio

    snr_scale = redshift_snr(z_old, z_new)
    new_flux, new_flux_err = apply_snr_noise(new_flux, new_flux_err, snr_scale)

    new_t = t * (1 + z_new) / (1 + z_old)
    return new_t, new_flux, new_flux_err

def apply_temperature(t, flux, flux_err, band, temp_factor,
                      Tmin, Tmax, t0, t_color, band_wave_aa):
    # scale temperature by a factor — > 1 is hotter, < 1 is cooler
    T     = compute_temperature(t, Tmin, Tmax, t0, t_color)
    T_new = T * temp_factor

    wav_array  = np.array([band_wave_aa[b] for b in band])
    ratio      = planck(wav_array, T_new) / planck(wav_array, T)
    return flux * ratio, flux_err * ratio

def apply_stretch(t, t0, rise, fall):
    # stretch rise and fall separately relative to peak
    t_stretch = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < t0:
            t_stretch[i] = t0 + (t[i] - t0) * rise
        else:
            t_stretch[i] = t0 + (t[i] - t0) * fall
    return t_stretch

def apply_sampling(t, flux, flux_err, band, sampling_fraction):
    # randomly remove data points to mimic real survey cadence
    mask = np.random.rand(len(t)) < sampling_fraction
    return t[mask], flux[mask], flux_err[mask], band[mask]

def compute_milky_way_extinction(ebv, lambda_angstrom, Rv=3.1):
    # F99 dust extinction curve at a given wavelength
    lambda_eff = lambda_angstrom * u.AA
    ext = F99(Rv=Rv)
    R_lambda = ext(lambda_eff) * Rv
    return R_lambda * ebv

def apply_extinction(flux, flux_err, band, ebv, band_wave_aa):
    # more dust = more extinction = lower observed flux
    new_flux     = np.zeros(len(flux))
    new_flux_err = np.zeros(len(flux_err))
    for i in range(len(flux)):
        wav               = band_wave_aa[band[i]]
        A_lambda          = compute_milky_way_extinction(ebv, wav)
        extinction_factor = 10**(-0.4 * A_lambda)
        new_flux[i]       = flux[i] * extinction_factor
        new_flux_err[i]   = flux_err[i] * extinction_factor
    return new_flux, new_flux_err

# -------------------------------------------------------------------------#
# effect of redshift
# -------------------------------------------------------------------------#

z_values = [0.1, 0.166, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, z in zip(axes, z_values):
    new_t, new_flux, _ = apply_redshift(
        smooth_t_combined, smooth_flux_combined, smooth_flux_err_combined,
        smooth_band, z_0, z, Tmin, Tmax, reference_time, t_color, band_wave_aa
    )
    mask_g = smooth_band == 'g'
    sort_g = np.argsort(new_t[mask_g])
    ax.plot(new_t[mask_g][sort_g] - np.min(new_t[mask_g][sort_g]),
            new_flux[mask_g][sort_g], linewidth=2, color='green', label='g-band fit')

    mask_r = smooth_band == 'r'
    sort_r = np.argsort(new_t[mask_r])
    ax.plot(new_t[mask_r][sort_r] - np.min(new_t[mask_r][sort_r]),
            new_flux[mask_r][sort_r], linewidth=2, color='red', label='r-band fit')

    t_data, new_flux_data, new_flux_err_data = apply_redshift(
        t, flux, flux_err, band, z_0, z,
        Tmin, Tmax, reference_time, t_color, band_wave_aa
    )
    t_data = t_data - np.min(t_data)
    ax.errorbar(t_data[band == 'g'], new_flux_data[band == 'g'],
                yerr=new_flux_err_data[band == 'g'],
                fmt='o', color='darkgreen', markersize=3, alpha=0.6, label='g-band data')
    ax.errorbar(t_data[band == 'r'], new_flux_data[band == 'r'],
                yerr=new_flux_err_data[band == 'r'],
                fmt='o', color='darkred', markersize=3, alpha=0.6, label='r-band data')
    ax.set_title(f"z = {z}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Redshift on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("redshift_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------------#
# effect of temperature
# -------------------------------------------------------------------------#

temp_factors = [0.8, 1.0, 1.2]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, temp in zip(axes, temp_factors):
    f_g   = smooth_flux_combined[smooth_band == 'g']
    f_r   = smooth_flux_combined[smooth_band == 'r']
    err_g = smooth_flux_err_combined[smooth_band == 'g']
    err_r = smooth_flux_err_combined[smooth_band == 'r']

    new_f_g, _ = apply_temperature(smooth_t_combined[smooth_band == 'g'],
                                   f_g, err_g, smooth_band[smooth_band == 'g'],
                                   temp, Tmin, Tmax, reference_time, t_color, band_wave_aa)
    new_f_r, _ = apply_temperature(smooth_t_combined[smooth_band == 'r'],
                                   f_r, err_r, smooth_band[smooth_band == 'r'],
                                   temp, Tmin, Tmax, reference_time, t_color, band_wave_aa)

    # normalise so panels are comparable
    peak_flux = max(new_f_r.max(), new_f_g.max())
    new_f_g  /= peak_flux
    new_f_r  /= peak_flux

    ax.plot(smooth_t_combined[smooth_band == 'g'] - t_ref, new_f_g, linewidth=2, color='green', label='g-band')
    ax.plot(smooth_t_combined[smooth_band == 'r'] - t_ref, new_f_r, linewidth=2, color='red',   label='r-band')
    ax.set_title(f"Temperature factor = {temp}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Temperature on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("temperature_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------------#
# effect of stretch
# -------------------------------------------------------------------------#

rise_factors = [0.5, 1.0, 1.5]
fall_factors = [0.5, 1.0, 1.5]
fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey=True)

for row, fall in enumerate(fall_factors):
    for col, rise in enumerate(rise_factors):
        ax = axes[row][col]
        t_g = apply_stretch(smooth_t_combined[smooth_band == 'g'], t0_model, rise, fall) - t_ref
        t_r = apply_stretch(smooth_t_combined[smooth_band == 'r'], t0_model, rise, fall) - t_ref
        ax.plot(t_g, smooth_flux_combined[smooth_band == 'g'], linewidth=2, color='green', label='g-band fit')
        ax.plot(t_r, smooth_flux_combined[smooth_band == 'r'], linewidth=2, color='red',   label='r-band fit')

        t_data = apply_stretch(t, t0_model, rise, fall) - t_ref
        ax.errorbar(t_data[band == 'g'], flux[band == 'g'], yerr=flux_err[band == 'g'],
                    fmt='o', color='darkgreen', markersize=3, alpha=0.6, label='g-band data')
        ax.errorbar(t_data[band == 'r'], flux[band == 'r'], yerr=flux_err[band == 'r'],
                    fmt='o', color='darkred',   markersize=3, alpha=0.6, label='r-band data')
        ax.set_title(f"Rise = {rise}, Fall = {fall}")
        ax.set_xlabel("Days since peak")
        if col == 0:
            ax.set_ylabel("Flux")
        ax.legend(fontsize=7)

plt.suptitle("Effect of Stretch on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("stretch_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------------#
# effect of sampling
# -------------------------------------------------------------------------#

sampling_fractions = [0.3, 0.6, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, fraction in zip(axes, sampling_fractions):
    t_s, flux_s, flux_err_s, band_s = apply_sampling(t, flux, flux_err, band, fraction)
    ax.errorbar(t_s[band_s == 'g'], flux_s[band_s == 'g'], yerr=flux_err_s[band_s == 'g'],
                fmt='o', color='darkgreen', markersize=3, alpha=0.6, label='g-band')
    ax.errorbar(t_s[band_s == 'r'], flux_s[band_s == 'r'], yerr=flux_err_s[band_s == 'r'],
                fmt='o', color='darkred',   markersize=3, alpha=0.6, label='r-band')
    ax.set_title(f"Sampling fraction = {fraction}")
    ax.set_xlabel("MJD")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Sampling on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("sampling_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------------#
# effect of galactic extinction
# -------------------------------------------------------------------------#

ebv_values = [0.0, 0.1, 0.3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, ebv in zip(axes, ebv_values):
    f_g   = smooth_flux_combined[smooth_band == 'g']
    f_r   = smooth_flux_combined[smooth_band == 'r']
    err_g = smooth_flux_err_combined[smooth_band == 'g']
    err_r = smooth_flux_err_combined[smooth_band == 'r']

    new_f_g, _ = apply_extinction(f_g, err_g, smooth_band[smooth_band == 'g'], ebv, band_wave_aa)
    new_f_r, _ = apply_extinction(f_r, err_r, smooth_band[smooth_band == 'r'], ebv, band_wave_aa)

    ax.plot(smooth_t_combined[smooth_band == 'g'] - t_ref, new_f_g, linewidth=2, color='green', label='g-band')
    ax.plot(smooth_t_combined[smooth_band == 'r'] - t_ref, new_f_r, linewidth=2, color='red',   label='r-band')
    ax.set_title(f"E(B-V) = {ebv}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Galactic Extinction on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("extinction_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------------#
# generate all 18,750 simulated light curves
# -------------------------------------------------------------------------#

redshift_grid   = np.linspace(0.1, 0.8, 10)
rise_grid       = np.linspace(0.8, 1.2, 5)
fall_grid       = np.linspace(0.8, 1.2, 5)
temp_grid       = np.linspace(0.8, 1.2, 5)
sampling_grid   = np.linspace(0.3, 1.0, 5)
extinction_grid = np.linspace(0.0, 0.3, 3)

param_grid = list(itertools.product(
    redshift_grid, rise_grid, fall_grid, temp_grid, extinction_grid, sampling_grid
))
print(f"Total light curves to generate: {len(param_grid)}")

all_lightcurves = []

for params in param_grid:
    z, rise, fall, temp, ebv, sampling = params

    new_t, new_flux, new_flux_err = apply_redshift(
        t, flux, flux_err, band, z_0, z, 20,
        Tmin, Tmax, reference_time, t_color, band_wave_aa
    )
    new_t                                    = apply_stretch(new_t, t0_model, rise, fall)
    new_flux, new_flux_err                   = apply_temperature(
        new_t, new_flux, new_flux_err, band, temp,
        Tmin, Tmax, reference_time, t_color, band_wave_aa
    )
    new_flux, new_flux_err                   = apply_extinction(new_flux, new_flux_err, band, ebv, band_wave_aa)
    new_t, new_flux, new_flux_err, new_band  = apply_sampling(new_t, new_flux, new_flux_err, band, sampling)

    all_lightcurves.append({
        'z': z, 'rise': rise, 'fall': fall, 'temp': temp,
        'ebv': ebv, 'sampling': sampling,
        't': new_t, 'flux': new_flux, 'flux_err': new_flux_err, 'band': new_band
    })

print(f"Generated {len(all_lightcurves)} light curves!")

with open('simulated_lightcurves.pkl', 'wb') as f:
    pickle.dump(all_lightcurves, f)
print("Raw light curves saved to simulated_lightcurves.pkl")

# -------------------------------------------------------------------------#
# fit RainbowFit to each simulated light curve to extract features
# -------------------------------------------------------------------------#

all_features = []

for i, lc in enumerate(all_lightcurves):
    try:
        values_sim = rainbow_model(lc['t'], lc['flux'], sigma=lc['flux_err'], band=lc['band'])

        feature_dict = {
            'z': lc['z'], 'rise': lc['rise'], 'fall': lc['fall'],
            'temp': lc['temp'], 'ebv': lc['ebv'], 'sampling': lc['sampling']
        }
        for name, val in zip(rainbow_model.names, values_sim):
            feature_dict[name] = val

        all_features.append(feature_dict)

        # save a checkpoint every 100 light curves in case it crashes
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_lightcurves)}")
            pd.DataFrame(all_features).to_csv("simulated_features.csv", index=False)

    except Exception as e:
        print(f"Light curve {i} failed: {e}")
        continue

print(f"Successfully extracted features for {len(all_features)} light curves!")

pd.DataFrame(all_features).to_csv("simulated_features.csv", index=False)
print("Simulated features saved to simulated_features.csv")

end = time.time()
print(f"Total run time: {end - start:.1f} seconds")