# The Cannon HIRES Iodine Pipeline (CHIP)
---

## Overview 
---

- CHIP downloads deblazed spectra from [The Keck Observatory Archive (KOA)](https://koa.ipac.caltech.edu/UserGuide/about.html)
- Prepares the spectra to be used for training of [The Cannon](https://annayqho.github.io/TheCannon/intro.html)
- Preform a comprehensive search of hyperparameters for The Cannon 
- Plot Results of the "best" model

## Installation
---

### Step 0

To access KOA data you must have an account. To get an account, submit a ticket [here](https://koa.ipac.caltech.edu/cgi-bin/Helpdesk/nph-genTicketForm?projname=KOA).

### Step 1

Clone this repository.

### Step 2
To setup the environment used during the testing of CHIP you'll need [Anaconda 3](https://www.anaconda.com).

```conda create --name chip python=3.8.5```

```conda activate chip``` 

```cd CHIP```

```pip3 install -r requirements.txt```

### Step 2.5 (Windows Users Only)

If you are on Windows, you need to download [geos_c.dll](https://www.dll-files.com/geos_c.dll.html) then put it in the chip python envoriment's Library\bin folder. You can read more about this problem [here](https://github.com/Toblerity/Shapely/pull/1108).

### Step 3
You're now ready to run CHIP! Inside the [ScriptsAndData](https://github.com/jgussman/CHIP/tree/main/ScriptsAndData) directory you'll find another README that will detail how to run CHIP. 


---
# Contact Info
--- 

Email: jgussman@iu.edu 