#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:30:28 2026

@author: sofiabalasundaram
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
from dust_extinction.parameter_averages import F99 
from astropy.cosmology import Planck18 as cosmo
import pickle
import itertools
import time 


# ---------------------------- Constants -----------------------------------#
start = time.time()
c_kms = c.to('km/s').value       # converts from m/s to km/s
H0 = 67.8                        # Hubble constant [km/s/Mpc]
z_0 = 0.166                      # Redshift of SN 2018ibb

# ---------------------------- Load Data -----------------------------------#

# Read photometric data (magnitudes, errors, filters, time)
df = pd.read_csv("2018ibb_Photometric_Data.csv")

# Create working copy
data = df.copy()

# Separate data by filter for independent analysis
zr_data = df[df['filter'] == 'zr'].copy()
zg_data = df[df['filter'] == 'zg'].copy()

# --------------------- Absolute Magnitude ---------------------------------#

def distance_modulus(z):
    """
    Compute distance modulus using astropy Planck18 cosmology.
    More accurate than low-redshift approximation for higher redshifts.
    """
    d_l_pc = cosmo.luminosity_distance(z).to(u.pc).value
    return 5 * np.log10(d_l_pc / 10)

def calc_abs_mag_DM(apparent_mag, z):
    DM = distance_modulus(z)
    return apparent_mag - DM + 2.5 * np.log10(1 + z)

# Compute absolute magnitudes for both filters
zg_data['abs_mag'] = calc_abs_mag_DM(zg_data['mag'], z_0)
zr_data['abs_mag'] = calc_abs_mag_DM(zr_data['mag'], z_0)

# --------------------- Peak Brightness ------------------------------------#

# Find brightest point (minimum magnitude)
idx_zg = zg_data['abs_mag'].idxmin()
idx_zr = zr_data['abs_mag'].idxmin()

# Extract peak magnitude and corresponding time (MJD)
min_mag_zg = zg_data.loc[idx_zg, 'abs_mag']
t_max_zg = zg_data.loc[idx_zg, 'mjd']

min_mag_zr = zr_data.loc[idx_zr, 'abs_mag']
t_max_zr = zr_data.loc[idx_zr, 'mjd']

print("Maximum brightness for G-band:", min_mag_zg)
print("Time at maximum brightness for G-band:", t_max_zg)

print("Maximum brightness for R-band:", min_mag_zr)
print("Time at maximum brightness for R-band:", t_max_zr)

# --------------------- Rest-Frame Time ------------------------------------#

# Convert observation time to rest-frame time:
# removes time dilation due to cosmic expansion
zg_data['t_rest'] = (zg_data['mjd'] - t_max_zg) / (1 + z_0)
zr_data['t_rest'] = (zr_data['mjd'] - t_max_zr) / (1 + z_0)

# --------------------- Plot Light Curves ----------------------------------#

# Use magnitude errors directly
zg_data['abs_mag_err'] = zg_data['magerr']
zr_data['abs_mag_err'] = zr_data['magerr']

# R-band light curve
plt.figure(figsize=(8, 5))
plt.plot(zr_data['t_rest'], zr_data['abs_mag'], color='red', label='zr')
plt.errorbar(zr_data['t_rest'], zr_data['abs_mag'],
             yerr=zr_data['abs_mag_err'],
             fmt='none', ecolor='black', alpha=0.9)
plt.gca().invert_yaxis()  # smaller mag = brighter
plt.xlabel('Days Since Maximum')
plt.ylabel('Absolute Magnitude')
plt.title('Light Curve (R-band)')
plt.legend()
plt.show()

# G-band light curve
plt.figure(figsize=(8, 5))
plt.plot(zg_data['t_rest'], zg_data['abs_mag'], color='green', label='zg')
plt.errorbar(zg_data['t_rest'], zg_data['abs_mag'],
             yerr=zg_data['abs_mag_err'],
             fmt='none', ecolor='black')
plt.gca().invert_yaxis()
plt.xlabel('Days Since Maximum')
plt.ylabel('Absolute Magnitude')
plt.title('Light Curve (G-band)')
plt.legend()
plt.show()

# --------------------- Prepare Data for RainbowFit ------------------------#

# Map filters to simplified band labels
data['band'] = data['filter'].map({'zg': 'g', 'zr': 'r'})

# Assign effective wavelengths (Ångstrom)
band_wave_aa = {'g': 4770.0, 'r': 6231.0}

flux, flux_err =mag2fluxcal_snana(data['mag'].values, data['magerr'].values)

# Extract time and band arrays
t = data['mjd'].values
band = data['band'].values

# --------------------- Fit Rainbow Model ----------------------------------#

# Initialize model
rainbow_model = RainbowFit.from_angstrom(band_wave_aa,with_baseline=False, temperature='sigmoid', bolometric='bazin')

# Fit model to data
values = rainbow_model(t, flux, sigma=flux_err, band=band)

# Print fitted parameters
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:
        print(f"{name:15}: {val:.2e}")
    else:
        print(f"{name:15}: {val:.2f}")

reference_time = values[0]
amplitude      = values[1]
rise_time      = values[2]
fall_time      = values[3]
Tmin           = values[4]
Tmax           = values[5]
t_color        = values[6]

# --------------------- Save Extracted Features ----------------------------#

# Format parameters for CSV output
feature_dict = {}
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:
        feature_dict[name] = f"{val:.2e}"
    else:
        feature_dict[name] = round(val, 2)

# Save features
df_features = pd.DataFrame([feature_dict])
df_features.to_csv("SN2018ibb_RainbowFit_Features.csv", index=False)

# --------------------- Smooth Model for Plotting --------------------------#

# Evaluate model on dense time grid for smooth visualization
smooth_time = np.linspace(58350, 58750, 100)

# Generate model curves for each band separately
smooth_flux_g = rainbow_model.model(smooth_time,np.repeat('g', len(smooth_time)), * values[:-1])
smooth_flux_r = rainbow_model.model (smooth_time, np.repeat('r', len(smooth_time)), * values[:-1])

# Find peak of smooth model to use as t0 for stretch (avoids noisy data spike)
peak_idx = np.argmax(smooth_flux_g)
t0_model = smooth_time[peak_idx]

# --------------------- Plot Fit -------------------------------------------#

plt.figure(figsize=(8,5))

# Plot data points
mask_g = band == 'g'
plt.errorbar(t[mask_g], flux[mask_g], yerr=flux_err[mask_g], fmt='o', color='darkgreen', label='g-band data', alpha=0.7)

mask_r = band == 'r'
plt.errorbar(t[mask_r], flux[mask_r], yerr=flux_err[mask_r], fmt='o', color='orange', label='r-band data', alpha=0.7)

# Plot smooth fitted model
plt.plot(smooth_time, smooth_flux_g, '-', color='green', label='g-band fit')
plt.plot(smooth_time, smooth_flux_r, '-', color='red', label='r-band fit')

plt.xlabel("MJD")
plt.ylabel("Flux (relative)")
plt.title("SN 2018ibb RainbowFit")
plt.legend()

# Save and show figure
plt.savefig("SN2018ibb_RainbowFit.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ----------------------- Redshift and Temperature -----------------------------#

def compute_temperature(t, Tmin, Tmax, t0, t_color):
    """
    Computes the blackbody temperature at each t using the sigmoid model 
    from RainbowFit
    """
    delta_T = Tmax - Tmin
    return Tmin + delta_T /(1 + np.exp((t-t0)/t_color))

def planck(wavelength_aa, T):
    """
    Computes Planck spectral radiance at a given wavelength (in ångström) 
    and temperture (in kelvin)

    """
    lam = (wavelength_aa * 1e-10)       # Converting Å to m 
    exponent = (h.value * c.value) / (lam * k_B.value * T)
    return (2 * h.value * c.value ** 2 / lam ** 5) / (np.exp(exponent)-1)

def redshift_temperature(T, z_old, z_new):  
    """
    Corrects the temperature when moving from old redhsift to new. 
    Higher redshift -> lower observed temp
    """
    return T * ((1 + z_old)/ (1 + z_new))

def apply_snr_noise(flux, snr_scale):
    """
    Add Gaussian observational noise to simulate measurement uncertainty.
    Noise amplitude scales with flux and inverse SNR.
    """
    noise_sigma = flux / snr_scale
    noise = np.random.normal(0, noise_sigma)
    return flux + noise, noise_sigma

def redshift_snr(snr, z_old, z_new):
    """
    Scales SNR when moving from old redshift to new 
    higher redshift -> "noisier" observations 
    """
    return snr * ((1 + z_old) / (1 + z_new)) ** 2 

def apply_redshift(t, flux, flux_err, band, z_old, z_new, snr, Tmin, Tmax, t0, t_color, band_wave_aa):    
    """
    Simulates lightcurve at the new redshift with temperture correction,
    Planck ratio and SNR scaling and time dialation to every point.
    """
    # Temperature at each time
    T = compute_temperature(t, Tmin, Tmax, t0, t_color)
    
    # Redshifted tempertaure 
    T_red = redshift_temperature(T, z_old, z_new)
    
    # Planck ratio for each data point 
    wav_array = np.array([band_wave_aa[b] for b in band])
    B_original = planck(wav_array, T)
    B_redshifted = planck(wav_array, T_red)
    ratio = B_redshifted / B_original
    
    # Scale flux by Planck ratio and error using SNR scaling
    new_flux = flux * ratio 
    new_flux_err = flux_err * ratio
    
    # Scale SNR based on redshift and add noise
    snr_new = redshift_snr(snr, z_old, z_new)
    new_flux, new_flux_err = apply_snr_noise(new_flux, snr_new)
    
    # Time dilation  
    new_t = t * (1 + z_new) / (1 + z_old)
    
    return new_t, new_flux, new_flux_err


def apply_temperature(t, flux, flux_err, band, temp_factor,
                      Tmin, Tmax, t0, t_color, band_wave_aa):
    """
    Apply a temperature correction to the light curve without time stretching.
    temp_factor > 1 means hotter, temp_factor < 1 means cooler.
    """
    # Temperature at each time point
    T = compute_temperature(t, Tmin, Tmax, t0, t_color)

    # Apply temperature factor
    T_new = T * temp_factor

    # Planck ratio for each point
    wav_array = np.array([band_wave_aa[b] for b in band])
    B_original = planck(wav_array, T)
    B_redshifted = planck(wav_array, T_new)
    ratio = B_redshifted / B_original

    # Scale flux
    new_flux = flux * ratio
    new_flux_err = flux_err * ratio

    return new_flux, new_flux_err

# --------------------- Redshift Comparison Plot ---------------------------#

z_values = [0.1, 0.166, 1.0]

smooth_band = np.array(['g'] * len(smooth_time) + ['r'] * len(smooth_time))
smooth_flux_combined = np.concatenate([smooth_flux_g, smooth_flux_r])
smooth_t_combined = np.concatenate([smooth_time, smooth_time])
smooth_flux_err_combined = np.zeros(len(smooth_flux_combined))

# Common reference point for time shifting
t_ref = np.min(smooth_t_combined[smooth_band == 'g']) 

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, z in zip(axes, z_values):

    # Apply redshift simulation to smooth model curves. SNR = 20 (baseline)
    new_t, new_flux, _ = apply_redshift(
        smooth_t_combined, smooth_flux_combined, smooth_flux_err_combined,
        smooth_band, z_0, z, 9999,
        Tmin, Tmax, reference_time, t_color, band_wave_aa
    )

    # Plot smooth model curves
    mask_g = smooth_band == 'g'
    sort_g = np.argsort(new_t[mask_g])
    t_g = new_t[mask_g][sort_g] - np.min(new_t[mask_g][sort_g])
    f_g = new_flux[mask_g][sort_g]
    ax.plot(t_g, f_g, linewidth=2, color='green', label='g-band fit')

    mask_r = smooth_band == 'r'
    sort_r = np.argsort(new_t[mask_r])
    t_r = new_t[mask_r][sort_r] - np.min(new_t[mask_r][sort_r])
    f_r = new_flux[mask_r][sort_r]
    ax.plot(t_r, f_r, linewidth=2, color='red', label='r-band fit')

    # Apply time dilation to observed data points and plot them
    t_data, new_flux_data, new_flux_err_data = apply_redshift(
        t, flux, flux_err, band, z_0, z, 20,
        Tmin, Tmax, reference_time, t_color, band_wave_aa
    )
    t_data = t_data - np.min(t_data)

    mask_g_data = band == 'g'
    mask_r_data = band == 'r'

    ax.errorbar(t_data[mask_g_data], new_flux_data[mask_g_data],
                yerr=new_flux_err_data[mask_g_data],
                fmt='o', color='darkgreen', markersize=3, alpha=0.6, label='g-band data')
    ax.errorbar(t_data[mask_r_data], new_flux_data[mask_r_data],
                yerr=new_flux_err_data[mask_r_data],
                fmt='o', color='darkred', markersize=3, alpha=0.6, label='r-band data')

    ax.set_title(f"z = {z}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Redshift on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("redshift_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------- Temperature Comparison Plot ------------------------#

temp_factors = [0.8, 1.0, 1.2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, temp in zip(axes, temp_factors):

    f_g = smooth_flux_combined[smooth_band == 'g']
    f_r = smooth_flux_combined[smooth_band == 'r']
    err_g = smooth_flux_err_combined[smooth_band == 'g']
    err_r = smooth_flux_err_combined[smooth_band == 'r']

    new_f_g, new_err_g = apply_temperature(smooth_t_combined[smooth_band == 'g'],
                                            f_g, err_g,
                                            smooth_band[smooth_band == 'g'],
                                            temp, Tmin, Tmax,
                                            reference_time, t_color, band_wave_aa)
    new_f_r, new_err_r = apply_temperature(smooth_t_combined[smooth_band == 'r'],
                                            f_r, err_r,
                                            smooth_band[smooth_band == 'r'],
                                            temp, Tmin, Tmax,
                                            reference_time, t_color, band_wave_aa)

    t_g = smooth_t_combined[smooth_band == 'g'] - t_ref
    t_r = smooth_t_combined[smooth_band == 'r'] - t_ref

    ax.plot(t_g, new_f_g, linewidth=2, color='green', label='g-band')
    ax.plot(t_r, new_f_r, linewidth=2, color='red', label='r-band')

    ax.set_title(f"Temperature factor = {temp}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Temperature on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("temperature_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------- Light Curve Stretch ----------------------------------#

def apply_stretch(t, t0, rise, fall):
    """
    Stretches the rise and decay of the lightcurve seperately relative to t0 
    """
    t_stretch = np.zeros(len(t))
    for i in range(len(t)):
        if t[i]<t0 :
            t_stretch[i] = t0 + (t[i] - t0) * rise 
        else: 
            t_stretch[i] = t0 + (t[i] - t0) * fall   
        
    return t_stretch 

# --------------------- Stretch Comparison Plot ----------------------------#

rise_factors = [0.5, 1.0, 1.5]
fall_factors = [0.5, 1.0, 1.5]


fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey=True)

for row, fall in enumerate(fall_factors):
    for col, rise in enumerate(rise_factors):
        ax = axes[row][col]

        # Smooth model curves
        t_g = apply_stretch(smooth_t_combined[smooth_band == 'g'], t0_model, rise, fall)
        t_g = t_g - t_ref
        f_g = smooth_flux_combined[smooth_band == 'g']
        ax.plot(t_g, f_g, linewidth=2, color='green', label='g-band fit')

        t_r = apply_stretch(smooth_t_combined[smooth_band == 'r'], t0_model, rise, fall)
        t_r = t_r - t_ref
        f_r = smooth_flux_combined[smooth_band == 'r']
        ax.plot(t_r, f_r, linewidth=2, color='red', label='r-band fit')

        # Data points — shifted by same reference
        t_data = apply_stretch(t, t0_model, rise, fall)
        t_data = t_data - t_ref

        ax.errorbar(t_data[band == 'g'], flux[band == 'g'],
                    yerr=flux_err[band == 'g'],
                    fmt='o', color='darkgreen', markersize=3, alpha=0.6, label='g-band data')
        ax.errorbar(t_data[band == 'r'], flux[band == 'r'],
                    yerr=flux_err[band == 'r'],
                    fmt='o', color='darkred', markersize=3, alpha=0.6, label='r-band data')

        ax.set_title(f"Rise = {rise}, Fall = {fall}")
        ax.set_xlabel("Days since peak")
        if col == 0:
            ax.set_ylabel("Flux")
        ax.legend(fontsize=7)

plt.suptitle("Effect of Stretch on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("stretch_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------- Survey Sampling ------------------------------------#

def apply_sampling(t, flux, flux_err, band, sampling_fraction):
    """
    Randomly removing data points to mimic real life data
    """
    mask = np.random.rand(len(t)) < sampling_fraction
    return t[mask], flux[mask], flux_err[mask], band[mask]

# --------------------- Sampling Comparison Plot ---------------------------#

sampling_fractions = [0.3, 0.6, 1.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, fraction in zip(axes, sampling_fractions):

    t_g = smooth_t_combined[smooth_band == 'g'] - t_ref
    t_r = smooth_t_combined[smooth_band == 'r'] - t_ref
    f_g = smooth_flux_combined[smooth_band == 'g']
    f_r = smooth_flux_combined[smooth_band == 'r']
    band_g = smooth_band[smooth_band == 'g']
    band_r = smooth_band[smooth_band == 'r']
    
    err_g = smooth_flux_err_combined[smooth_band == 'g']
    err_r = smooth_flux_err_combined[smooth_band == 'r']
    
    t_g_s, f_g_s, err_g_s, _ = apply_sampling(t_g, f_g, err_g, band_g, fraction)
    t_r_s, f_r_s, err_r_s, _ = apply_sampling(t_r, f_r, err_r, band_r, fraction)

    ax.errorbar(t_g_s, f_g_s, fmt='o', color='darkgreen',
                markersize=3, alpha=0.6, label='g-band')
    ax.errorbar(t_r_s, f_r_s, fmt='o', color='darkred',
                markersize=3, alpha=0.6, label='r-band')

    ax.set_title(f"Sampling fraction = {fraction}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Sampling on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("sampling_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------- Galactic Extinction --------------------------------#

def compute_milky_way_extinction(ebv, lambda_angstrom, Rv=3.1):
    """
    Compute Milky Way extinction at a given wavelength.
    
    ebv              : E(B-V) dust extinction value
    lambda_angstrom  : effective wavelength of filter in Angstrom
    Rv               : extinction curve shape parameter (3.1 is standard)
    """
    lambda_eff = lambda_angstrom * u.AA
    ext = F99(Rv=Rv)
    R_lambda = ext(lambda_eff) * Rv
    A_lambda = R_lambda * ebv
    return A_lambda

def apply_extinction(flux, flux_err, band, ebv, band_wave_aa):
    """
    Apply Milky Way extinction to flux in each band.
    Higher ebv means more dust, more extinction, lower observed flux.
    """
    new_flux = np.zeros(len(flux))
    new_flux_err = np.zeros(len(flux_err))

    for i in range(len(flux)):
        wav = band_wave_aa[band[i]]
        A_lambda = compute_milky_way_extinction(ebv, wav)
        extinction_factor = 10**(-0.4 * A_lambda)
        new_flux[i] = flux[i] * extinction_factor
        new_flux_err[i] = flux_err[i] * extinction_factor

    return new_flux, new_flux_err

# --------------------- Extinction Comparison Plot -------------------------#

ebv_values = [0.0, 0.1, 0.3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, ebv in zip(axes, ebv_values):

    # Apply extinction to smooth model curves
    f_g = smooth_flux_combined[smooth_band == 'g']
    f_r = smooth_flux_combined[smooth_band == 'r']
    err_g = smooth_flux_err_combined[smooth_band == 'g']
    err_r = smooth_flux_err_combined[smooth_band == 'r']

    new_f_g, new_err_g = apply_extinction(f_g, err_g, smooth_band[smooth_band == 'g'], ebv, band_wave_aa)
    new_f_r, new_err_r = apply_extinction(f_r, err_r, smooth_band[smooth_band == 'r'], ebv, band_wave_aa)

    t_g = smooth_t_combined[smooth_band == 'g'] - t_ref
    t_r = smooth_t_combined[smooth_band == 'r'] - t_ref

    ax.plot(t_g, new_f_g, linewidth=2, color='green', label='g-band')
    ax.plot(t_r, new_f_r, linewidth=2, color='red', label='r-band')

    ax.set_title(f"E(B-V) = {ebv}")
    ax.set_xlabel("Days since peak")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Flux")
plt.suptitle("Effect of Galactic Extinction on Light Curve", y=1.02)
plt.tight_layout()
plt.savefig("extinction_lightcurve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------- Parameter Grid ------------------------------------#


# Define parameter ranges
redshift_grid    = np.linspace(0.1, 0.8, 10)
rise_grid        = np.linspace(0.8, 1.2, 5)
fall_grid        = np.linspace(0.8, 1.2, 5)
temp_grid        = np.linspace(0.8, 1.2, 5)
sampling_grid = np.linspace(0.3, 1.0, 5)
extinction_grid  = np.linspace(0.0, 0.3, 3)

# Generate all combinations
param_grid = list(itertools.product(redshift_grid, rise_grid, fall_grid, temp_grid, extinction_grid, sampling_grid))

print(f"Total number of light curves to generate: {len(param_grid)}")

# --------------------- Generate Light Curves -----------------------------#

all_lightcurves = []

for params in param_grid:
    z, rise, fall, temp, ebv, sampling = params

    # Step 1: Apply redshift
    new_t, new_flux, new_flux_err = apply_redshift(t, flux, flux_err, band, z_0, z, 20, Tmin, Tmax, reference_time, t_color, band_wave_aa)

    # Step 2: Apply stretch
    new_t = apply_stretch(new_t, t0_model, rise, fall)

    # Step 3: Apply temperature correction
    new_flux, new_flux_err = apply_temperature(new_t, new_flux, new_flux_err, band, temp, Tmin, Tmax, reference_time, t_color, band_wave_aa)

    # Step 4: Apply extinction
    new_flux, new_flux_err = apply_extinction(new_flux, new_flux_err, band, ebv, band_wave_aa)

    # Step 5: Apply sampling
    new_t, new_flux, new_flux_err, new_band = apply_sampling(new_t, new_flux, new_flux_err, band, sampling)

    # Store result
    all_lightcurves.append({'z': z, 'rise': rise, 'fall': fall,'temp': temp, 'ebv': ebv, 'sampling': sampling, 't': new_t, 'flux': new_flux,'flux_err': new_flux_err, 'band': new_band})

print(f"Generated {len(all_lightcurves)} light curves!")

# --------------------- Save Generated Light Curves -----------------------#


with open('simulated_lightcurves.pkl', 'wb') as f:
    pickle.dump(all_lightcurves, f)

print("Light curves saved to simulated_lightcurves.pkl")


# --------------------- Feature Extraction --------------------------------#

all_features = []

for i, lc in enumerate(all_lightcurves):
    try:
        # Fit RainbowFit to the simulated light curve
        values_sim = rainbow_model(lc['t'], lc['flux'], sigma=lc['flux_err'], band=lc['band'])
        
        # Store simulation parameters alongside fitted features
        feature_dict = {'z': lc['z'], 'rise': lc['rise'],'fall': lc['fall'], 'temp': lc['temp'], 'ebv': lc['ebv'], 'sampling': lc['sampling']}
        
        for name, val in zip(rainbow_model.names, values_sim):
            feature_dict[name] = val
            
        all_features.append(feature_dict)
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_lightcurves)} light curves fitted")
        if i % 100 == 0:
            df_temp = pd.DataFrame(all_features)
            df_temp.to_csv("simulated_features.csv", index=False)
        
    except Exception as e:
        print(f"Light curve {i} failed: {e}")
        continue

print(f"Successfully extracted features for {len(all_features)} light curves!")

# Save all features to CSV
df_all_features = pd.DataFrame(all_features)
df_all_features.to_csv("simulated_features.csv", index=False)
print("Simulated features saved to simulated_features.csv!")

end = time.time()
print(f"Total run time:{end - start}")


