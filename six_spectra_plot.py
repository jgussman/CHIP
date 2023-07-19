import pandas as pd 
import numpy as np
import os 


# Number of stars in plot 
n = 5

chip_run_dir = r'data\chip_runs\2023-07-19_01-35-52' # Directory of chip run

# Read in HIRES_Filename_snr.csv
snr_df = pd.read_csv(os.path.join(chip_run_dir,'HIRES_Filename_snr.csv')) # Columns: ["HIRESid", "FILENAME", "SNR"]

# Read in stellar parameters csv 
stellar_parameters_df = pd.read_csv('data/spocs/stellar_parameters.csv')

# Grab HIRESID & TEFF from stellar parameters csv
teff_df = stellar_parameters_df[['HIRESID','TEFF']]

# Remove Stars from teff_df that are not in snr_df
teff_df = teff_df[teff_df['HIRESID'].isin(snr_df['HIRESID'])]

# Sort by TEFF 
teff_df = teff_df.sort_values(by=['TEFF'])

# Find 6 equally spaced temperature values
min_temp = teff_df['TEFF'].min()
max_temp = teff_df['TEFF'].max()
step_temp = (max_temp - min_temp)/(2*n)

print(teff_df)
equally_spaced_list = []
current_temp_baseline = min_temp
for _ in range(n):
    # Remove stars with temp values that are smaller than the current baseline
    teff_df = teff_df[teff_df['TEFF'] >= current_temp_baseline]
    # Add the smallest temp star to equally_spaced_list
    # put values in a tuple (HIRESID, TEFF)
    equally_spaced_list.append(teff_df.iloc[0])
    # Set the new baseline
    current_temp_baseline = teff_df.iloc[0]['TEFF'] + step_temp

print("equally_spaced_list", equally_spaced_list, sep='\n')

# Using HIRESid from equally_spaced_list, find the corresponding FILENAME
filename_list = []
for hiresid,teff  in equally_spaced_list:
    # Get the filename 
    filename = snr_df[snr_df['HIRESID'] == hiresid]['FILENAME'].values
    # Print the filename and SNR
    filename_list.append((filename[0],teff))


# Obtain the spectra from the filename_list
spectra_dir = os.path.join(chip_run_dir,'inter')
spectra_list = [] 
print("filename_list", filename_list, sep='\n')
for filename,teff in filename_list:
    # Read in the spectra
    spectra_filename_path = os.path.join(spectra_dir, filename + "_resampled_flux.npy")
    spectra = np.load(spectra_filename_path)
    # Add the spectra to the list
    spectra_list.append((spectra,teff, filename))


from matplotlib.transforms import blended_transform_factory
from matplotlib.pyplot import subplots,plot,grid,xlabel,ylabel,text,legend, show
wl = np.load(os.path.join(spectra_dir,"interpolated_wl.npy"))
fig,ax = subplots(figsize=(8,4))
trans = blended_transform_factory(ax.transAxes,ax.transData)
bbox = dict(facecolor='white', edgecolor='none',alpha=0.8)
step = 1
shift = 0
for i,spectra_teff in enumerate(spectra_list):
    spec,teff, filename = spectra_teff
    plot(wl,spec + shift,color='RoyalBlue',lw=0.5)
    # Zoom in to the 5160-5200 Angstroms region
    ax.set_xlim(5120,5200) 
    s = f"{filename:s}, Teff={teff:.0f}"   
    text(0.01, 1+shift, s, bbox=bbox, transform=trans)
    shift+=step

grid()
xlabel('Wavelength (Angstroms)')
ylabel('Normalized Flux (Arbitrary Offset)')
legend()
show()


