# Tutorial: Getting Started with CHIP
---

Note: This tutorial assumes that you already have the CHIP environment activated. If not, run the command ```conda activate chip```.

There is one key files that the user will edit to run CHIP: *config.txt* (Step 1).

## Step 1
Open *config.txt* to see all the adjustable variables. The format is as follows:

```
# Description of what the variable will be used for.
Variable name  = Example
```

All text on the same line and on right side of the = will assigned to that variable.

Note: When reading this file, any line that starts with a hashtag will be ignored. 

In *config.txt* there are two Python files you will need to set the variables for: CHIP.py and main_The_Cannon.py. CHIP.py is used to download iodine imprinted spectra from The KOA and reduce them. main_The_Cannon.py will take those reduced spectra and train The Cannon models, then determine the "best" model and save the results. 



## Step 2 (optional)
To visualize the results of The Cannon "best", you may run Plot_Results.py.
