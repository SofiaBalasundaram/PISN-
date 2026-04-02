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

# Constants
## hello 

c = 3e5  # speed of light in km/s
H0 = 67.8  # Hubble constant in km/s/Mpc
z = 0.166  # redshift

### ----------------------------File-----------------------------------------###

# Reading the CSV file
df = pd.read_csv("2018ibb_Photometric_Data.csv")
data = df.copy()

# Seperating the filters zg and zr in the file
zr_data = df[df['filter'] == 'zr']
zg_data = df[df['filter'] == 'zg']


### ---------------------Finding Absolute Magnitude--------------------------###


# Distance modulus
def distance_modulus(z, c, H0):
    # Luminosity distance in Mpc.  Using low-redshift approximation: 
    #d_L ≈ (c/H0) * z
    d_l_Mpc = (c / H0) * z
    d_l_pc = d_l_Mpc * 1e6  # Convert to parsecs
    return 5 * np.log10(d_l_pc / 10)

# Absolute magnitude using DM(z) + 2.5 log(1+z)
def calc_abs_mag_DM(apparent_mag, z, c, H0):
    DM = distance_modulus(z, c, H0)
    return apparent_mag - DM + 2.5 * np.log10(1 + z)

# Note to self - astropy can do this rewrite accordingly


# Separate the filters and make copies
zr_data = df[df['filter'] == 'zr'].copy()
zg_data = df[df['filter'] == 'zg'].copy()



### ---------------------Finding Maximum Brightness--------------------------###

# Calculate absolute magnitudes first
zg_data['abs_mag'] = calc_abs_mag_DM(zg_data['mag'], z, c, H0)
zr_data['abs_mag'] = calc_abs_mag_DM(zr_data['mag'], z, c, H0)

# Find peak indices using absolute magnitude (brightest = smallest value)
idx_zg = zg_data['abs_mag'].idxmin()
idx_zr = zr_data['abs_mag'].idxmin()

# Extract peak magnitude and time
min_mag_zg = zg_data.loc[idx_zg, 'abs_mag']
t_max_zg = zg_data.loc[idx_zg, 'mjd']

min_mag_zr = zr_data.loc[idx_zr, 'abs_mag']
t_max_zr = zr_data.loc[idx_zr, 'mjd']

print("Maximum brightness for G-band:", min_mag_zg)
print("Time at maximum brightness for G-band:", t_max_zg)

print("Maximum brightness for R-band:", min_mag_zr)
print("Time at maximum brightness for R-band:", t_max_zr)

### --------------------Changing from MJD to rest frame---------------------###

# Calculating rest frame
zg_data['t_rest'] = (zg_data['mjd'] - t_max_zg) / (1 + z)
zr_data['t_rest'] = (zr_data['mjd'] - t_max_zr) / (1 + z)


### ------------------------Plotting-----------------------------------------###

# Errors
zg_data['abs_mag_err'] = zg_data['magerr']
zr_data['abs_mag_err'] = zr_data['magerr']


# Plotting lightcurve for zr
plt.figure(figsize=(8, 5))
plt.plot(zr_data['t_rest'], zr_data['abs_mag'], color='red', label='zr')
plt.errorbar(zr_data['t_rest'], zr_data['abs_mag'], yerr=zr_data['abs_mag_err'],
             fmt='none', label='zr', ecolor='black', alpha=0.9, zorder=1)
plt.gca().invert_yaxis()  # magnitudes are brighter when smaller
plt.xlabel('Days Since Maximum')
plt.ylabel('Apparent Magnitude')
plt.title('Light Curve for R - Band filter')
plt.legend()
plt.show()

# Plotting lightcurve for zg
plt.figure(figsize=(8, 5))
plt.plot(zg_data['t_rest'], zg_data['abs_mag'], color='green', label='zg')
plt.errorbar(zg_data['t_rest'], zg_data['abs_mag'], yerr=zg_data['abs_mag_err'],
             fmt='none', label='zr', ecolor='black', zorder=1)
plt.gca().invert_yaxis()
plt.xlabel('Days Since Maximum')
plt.ylabel('Apparent Magnitude')
plt.title('Light Curve for G-Band filter')
plt.legend()
plt.show()

###------------------------Feature Extraction------------------------------###

# we want to combine the filters into one array and map filter names to simpler
# band labels for RainbowFit
data['band'] = data['filter'].map({
    'zg': 'g',
    'zr': 'r'
})

# Each band corresponds to a effective wavelength (in Angstrom) 
# so assign wavelength
band_wave_aa = {
    'g': 4770.0,
    'r': 6231.0
}

# RainbowFit works with fluxes so convert mag to flux 
flux = 10**(-0.4 * data['mag'].values)

# Propagation of errors for magnitudes into fluxes 
flux_err = flux * (0.4 * np.log(10)) * data['magerr'].values

# Listing inputs for Rainbowfit 
t = data['mjd'].values
band = data['band'].values

# Create RainbowFit model
rainbow_model = RainbowFit.from_angstrom(
    band_wave_aa,      # your band → wavelength mapping
    with_baseline=False,
    temperature='sigmoid',  # default logistic function
    bolometric='bazin'      # default Bazin flux evolution
)
# Fit Rainbow model to your data
values = rainbow_model(
    t,         # observation times
    flux,      # fluxes
    sigma=flux_err,  # flux errors
    band=band       # band labels
)

# Print parameters neatly with rounding
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:          # very small numbers → scientific notation
        print(f"{name:15}: {val:.2e}")
    else:                        # normal numbers → 2 decimal places
        print(f"{name:15}: {val:.2f}")
        

# Generate model fluxes using the fitted parameters
# model_flux = rainbow_model.model(t, band, *values[:-1])  # exclude chi2
 
# Smoothing 
smooth_time = np.linspace(58350, 58750, 1000)

smooth_band_g = np.array(['g'] * len(smooth_time))
smooth_band_r = np.array(['r'] * len(smooth_time))

model_flux_g = rainbow_model.model(smooth_time, smooth_band_g, *values[:-1])
model_flux_r = rainbow_model.model(smooth_time, smooth_band_r, *values[:-1])

plt.figure(figsize=(8,5))

# G-band
mask_g = band == 'g'
plt.errorbar(t[mask_g], flux[mask_g], yerr=flux_err[mask_g],
             fmt='o', color='darkgreen', label='g-band data', alpha=0.7)
plt.plot(smooth_time, model_flux_g, '-', color='green', label='g-band RainbowFit')

# R-band
mask_r = band == 'r'
plt.errorbar(t[mask_r], flux[mask_r], yerr=flux_err[mask_r],
             fmt='o', color='orange', label='r-band data', alpha=0.7)
plt.plot(smooth_time, model_flux_r, '-', color='red', label='r-band RainbowFit')

plt.xlabel("MJD")
plt.ylabel("Flux (relative)")
plt.title("SN 2018ibb RainbowFit")
plt.legend()


# Save figure AFTER plotting everything
plt.savefig("SN2018ibb_RainbowFit.png", dpi=300, bbox_inches='tight')
plt.show()  # optional, displays plot
plt.close()  # close figure to avoid overlap

# Round or format numbers for CSV
feature_dict = {}
for name, val in zip(rainbow_model.names, values):
    if abs(val) < 1e-2:  # very small → scientific
        feature_dict[name] = f"{val:.2e}"
    else:  # normal numbers → 2 decimals
        feature_dict[name] = round(val, 2)

# Create dataframe and save
df_features = pd.DataFrame([feature_dict])
df_features.to_csv("SN2018ibb_RainbowFit_Features.csv", index=False)