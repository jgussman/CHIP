# The Cannon HIRES Iodine Pipeline (CHIP)
---

## Overview 
---

The CHIP pipeline is designed to extract stellar parameters from iodine-imprinted radial velocity spectra obtained with the HIRES instrument. To accomplish this, CHIP:
- Downloads deblazed spectra from [The Keck Observatory Archive (KOA)](https://koa.ipac.caltech.edu/UserGuide/about.html)
- Prepares the spectra to be used for training of [The Cannon](https://annayqho.github.io/TheCannon/intro.html)
- "Learns" patterns in the input dataset by optimizing a set of model hyperparameters with The Cannon 
- Plots the results extracted from the best-fitting model


## Dependencies
---

To access the input Keck/HIRES data, you must have a KOA account. To request an account, submit a ticket [here](https://koa.ipac.caltech.edu/cgi-bin/Helpdesk/nph-genTicketForm?projname=KOA).

To set up the environment required to use CHIP, you will need [Anaconda 3](https://www.anaconda.com).


## Installation
---

### Step 1

Clone this repository by entering the following into your terminal:

```git clone https://github.com/jgussman/CHIP'''

### Step 2
Set up the environment required to use CHIP, using [Anaconda 3](https://www.anaconda.com).

```conda create --name chip python=3.8.5```

```conda activate chip``` 

```cd CHIP```

```pip3 install -r requirements.txt```

### Step 2.5 (Windows Users Only)

If you are using a Windows machine, you need to download [geos_c.dll](https://www.dll-files.com/geos_c.dll.html) and move the file to the CHIP python environment's Library/bin folder. You can read more about this problem [here](https://github.com/Toblerity/Shapely/pull/1108).

### Step 3
You're now ready to run CHIP! Inside the [ScriptsAndData](https://github.com/jgussman/CHIP/tree/main/ScriptsAndData) directory, you will find another README file with details about how to initialize and run CHIP.


---
# Contact Info
--- 

The author of this pipeline can be reached via email at jgussman@iu.edu.
