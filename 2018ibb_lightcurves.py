#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:30:28 2026

@author: sofiabalasundaram
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from light_curve.light_curve_py import RainbowFit

# ---------------------------- Constants -----------------------------------#

c = 3e5      # Speed of light [km/s]
H0 = 67.8    # Hubble constant [km/s/Mpc]
z_0 = 0.166    # Redshift of SN 2018ibb


# ---------------------------- Load Data -----------------------------------#

# Read photometric data (magnitudes, errors, filters, time)
df = pd.read_csv("2018ibb_Photometric_Data.csv")

# Create working copy
data = df.copy()

# Separate data by filter for independent analysis
zr_data = df[df['filter'] == 'zr'].copy()
zg_data = df[df['filter'] == 'zg'].copy()


# --------------------- Absolute Magnitude ---------------------------------#

def distance_modulus(z, c, H0):
    """
    Compute distance modulus using low-redshift approximation:
    d_L ≈ (c/H0) * z
    """
    d_l_Mpc = (c / H0) * z          # Luminosity distance [Mpc]
    d_l_pc = d_l_Mpc * 1e6          # Convert to parsec
    return 5 * np.log10(d_l_pc / 10)

def calc_abs_mag_DM(apparent_mag, z, c, H0):
    """
    Convert apparent magnitude to absolute magnitude.
    Includes cosmological (1+z) correction.
    """
    DM = distance_modulus(z, c, H0)
    return apparent_mag - DM + 2.5 * np.log10(1 + z)


# Compute absolute magnitudes for both filters
zg_data['abs_mag'] = calc_abs_mag_DM(zg_data['mag'], z_0, c, H0)
zr_data['abs_mag'] = calc_abs_mag_DM(zr_data['mag'], z_0, c, H0)


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
band_wave_aa = {
    'g': 4770.0,
    'r': 6231.0
}

# Convert magnitudes → flux (RainbowFit works in flux)
flux = 10**(-0.4 * data['mag'].values)

# Propagate magnitude errors to flux errors 
flux_err = flux * (0.4 * np.log(10)) * data['magerr'].values

# Extract time and band arrays
t = data['mjd'].values
band = data['band'].values

# --------------------- Fit Rainbow Model ----------------------------------#

# Initialize model
rainbow_model = RainbowFit.from_angstrom(
    band_wave_aa,
    with_baseline=False,
    temperature='sigmoid',
    bolometric='bazin'
)

# Fit model to data
values = rainbow_model(t, flux, sigma=flux_err, band=band)

# Print fitted parameters
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:
        print(f"{name:15}: {val:.2e}")
    else:
        print(f"{name:15}: {val:.2f}")

# --------------------- Smooth Model for Plotting --------------------------#

# Evaluate model on dense time grid for smooth visualization
smooth_time = np.linspace(58350, 58750, 100)

# Generate model curves for each band separately
smooth_flux_g = rainbow_model.model(
    smooth_time,
    np.repeat('g', len(smooth_time)),
    *values[:-1]
)

smooth_flux_r = rainbow_model.model(
    smooth_time,
    np.repeat('r', len(smooth_time)),
    *values[:-1]
)


# --------Simulated Observational Effects on Supernova Light Curves---------#

# ------------------- Redshift / Time Dilation ------------------------------#

def apply_redshift(t, z):
    """
    Apply time dilation due to redshift.
    Higher redshift stretches observed time by (1 + z).
    """
    return t * (1 + z) / (1 + z_0)


# ------------------- Light Curve Stretch ----------------------------------#

def apply_stretch(t, stretch_factor):
    """
    Stretch or compress the light curve in time.
    """
    return t * stretch_factor


# ------------------- Magnitude Shift (Brightness Change) ------------------#

def apply_magnitude_shift(flux, mag_shift):
    """
    Apply a shift in magnitude to the flux.
    Converts magnitude difference into a multiplicative flux scaling.
    """
    return flux * 10**(-0.4 * mag_shift)


# ------------------- Signal-to-Noise / Noise Model ------------------------#

def apply_snr_noise(flux, snr_scale):
    """
    Add Gaussian observational noise to simulate measurement uncertainty.
    Noise amplitude scales with flux and inverse SNR.
    """
    noise = np.random.normal(0, flux / snr_scale)
    return flux + noise


# --------------------- Survey Sampling ------------------------------------#

def apply_sampling(t, flux, band, sampling_fraction):
    """
    Randomly removing data points to mimic real life data
    """
    mask = np.random.rand(len(t)) < sampling_fraction
    return t[mask], flux[mask], band[mask]


# --------------------- Redshift Comparison Plot --------------------------#

plt.figure(figsize=(12, 6))

# Create a smooth time axis (not actually used in final plot here, but kept for potential interpolation)
smooth_time = np.linspace(min(t), max(t), 200)

# Define redshifts to compare
z_values = [0.05, 0.166, 0.4]

# Plot light curve at different redshifts
for z in z_values:

    # Apply cosmological time dilation
    t_z = apply_redshift(t, z)

    # Shift each curve so they start at zero for easier visual comparison
    t_z = t_z - np.min(t_z)

    # Plot flux vs shifted time
    plt.plot(t_z, flux, linewidth=2, label=f"z = {z}")

# Axis labels and plot formatting
plt.xlabel("Time (shifted)")
plt.ylabel("Flux")
plt.title("Effect of Redshift on Light Curve")
plt.legend()
plt.grid(alpha=0.3)

# Save figure to file with high resolution
plt.savefig("redshift_lightcurve.png", dpi=300, bbox_inches="tight")

# Display plot
plt.show()

"""
# --------------------- Smooth Model for Plotting --------------------------#

# Evaluate model on dense time grid for smooth visualization
smooth_time = np.linspace(58350, 58750, 100)

# Generate model curves for each band separately
smooth_flux_g = rainbow_model.model(
    smooth_time,
    np.repeat('g', len(smooth_time)),
    *values[:-1]
)

smooth_flux_r = rainbow_model.model(
    smooth_time,
    np.repeat('r', len(smooth_time)),
    *values[:-1]
)


# --------------------- Plot Fit -------------------------------------------#

plt.figure(figsize=(8,5))

# Plot data points
mask_g = band == 'g'
plt.errorbar(t[mask_g], flux[mask_g], yerr=flux_err[mask_g],
             fmt='o', color='darkgreen', label='g-band data', alpha=0.7)

mask_r = band == 'r'
plt.errorbar(t[mask_r], flux[mask_r], yerr=flux_err[mask_r],
             fmt='o', color='orange', label='r-band data', alpha=0.7)

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
"""