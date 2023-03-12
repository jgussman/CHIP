# The Cannon HIRES Iodine Pipeline (CHIP)


---
## Overview 


The CHIP pipeline is designed to extract stellar parameters from iodine-imprinted radial velocity spectra obtained with the HIRES instrument. To accomplish this, CHIP:
- Downloads deblazed spectra from [The Keck Observatory Archive (KOA)](https://koa.ipac.caltech.edu/UserGuide/about.html)
- Prepares the spectra to be used for training of [The Cannon](https://annayqho.github.io/TheCannon/intro.html)


---
## Dependencies


- To be able to access Keck/HIRES data, you will need a KOA (Keck Observatory Archive) account. To obtain this account, you can submit a ticket by following the link provided [here](https://koa.ipac.caltech.edu/cgi-bin/Helpdesk/nph-genTicketForm?projname=KOA).

- To use CHIP with Python, you will need to set up a Python environment with the necessary dependencies installed. We recommend using [Anaconda 3](https://www.anaconda.com) to manage your Python environment.

---
## Installation

1. Clone this repository by entering the following into your terminal:

   ```git clone https://github.com/jgussman/CHIP```

2. Set up the environment required to use CHIP, using [Anaconda 3](https://www.anaconda.com).

    ```conda env create```


3. (Windows Users Only) 

    If you are using a Windows machine, you **need** to download [geos_c.dll](https://www.dll-files.com/geos_c.dll.html) and move the file to the CHIP python environment's Library/bin folder. You can read more about this problem [here](https://github.com/Toblerity/Shapely/pull/1108).

4. You're now ready to run CHIP! Inside the src directory, you will find another [README](https://github.com/jgussman/CHIP/blob/main/src/README.md) with details about how to run CHIP.


---
### Contact Info

Email: judegussman@gmail.com.