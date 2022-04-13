# Tutorial: Getting Started with CHIP
---

Note: This tutorial assumes that you already have the CHIP environment activated. If not, run the command ```conda activate chip```.

There is one key file that the user will edit to run CHIP: *config.txt* (Step 1).

## Step 1
Open *config.txt* to view all of the adjustable variables. The format of each variable is as follows:

```
# Description of what the variable will be used for.
Variable name = "Example"
```

All text on the same line and on right side of the = will assigned to that variable.

Note: When reading this file, any line that starts with a hashtag will be ignored; these lines are used only for reference.

In *config.txt*, there are two Python files you will need to set the variables for: CHIP.py and main_The_Cannon.py. CHIP.py is used to download and reduce iodine-imprinted spectra from the NASA Exoplanet Science Institute's Precision Radial Velocity pipeline. main_The_Cannon.py uses those reduced spectra as a training set for The Cannon, conducting a grid search of allowable parameters to determine the "best" model and save the final results.



## Step 2 (optional)
We include helper functions to visualize the results of The Cannon within the Plot_Results.py file.
