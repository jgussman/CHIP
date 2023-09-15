import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import joblib
import TheCannon

# Read in stellar parameters csv
stellar_parameters_df = pd.read_csv('data/spocs/stellar_parameters.csv')

# Import training set and test set IDs
ds = joblib.load(r"data\chip_runs\2023-07-31_23-58-29\training_results\3_0.4_Mean-Error_2\ds.joblib")
tr_ID = ds.tr_ID
test_ID = ds.test_ID

# Get all the training set and test set IDs
train_df = stellar_parameters_df[stellar_parameters_df['HIRESID'].isin(tr_ID)]
test_df = stellar_parameters_df[stellar_parameters_df['HIRESID'].isin(test_ID)]

# Set up the figure and the subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharey=True, sharex=True)

# Create the scatter plots with reduced marker size
im1 = ax1.scatter(train_df['TEFF'], train_df['LOGG'], c=train_df['FeH'], cmap='viridis', s=10, alpha=0.8)
im2 = ax2.scatter(test_df['TEFF'], test_df['LOGG'], c=test_df['FeH'], cmap='viridis', s=10, alpha=0.8)

# Remove the tick labels on the y-axis for the right subplot
ax2.tick_params(axis='y', which='both', labelleft=False)

# Add axis labels and a title
ax1.set_xlabel('Effective Temperature (K)')
ax1.set_ylabel('Surface Gravity (dex)')
ax1.set_title('Training Set')
ax2.set_xlabel('Effective Temperature (K)')
ax2.set_title('Testing Set')

# Use make_axes_locatable() to create an axis for the colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)

# Add the colorbar to the new axis
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label('[Fe/H]')

# Invert both x and y axes
ax1.invert_xaxis()
ax1.invert_yaxis()

# Remove the whitespace between the subplots
fig.subplots_adjust(wspace=0)

# Show the plot
plt.show()

