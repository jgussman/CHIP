# Tutorial 
---
(Assumes that you have the chip environment already activated if not ```conda activate chip```)


## Step 1
Open *main_pipeline.py* in your IDE of choice.  

Use CTRL+F to find the following line of ocde 

```
if __name__ == '__main__':
```
The only code you should ever have to change is in this if-statement, particularly the following 2-lines

```{python} 
    crossMatchedNames = pd.read_csv("../spocData/testStars.txt",sep=" ")
    chipObject.Run(use_data=False,slow_normalize = True)
```

**crossMatchedNames** will be assigned to be a N by 2 dataframe. The file read in must contain two columns, the first column is the star’s identifier in any dataset you want (in our case SPOCS). The second column MUST be the identifier for the star in the NEXSCI database with the column header name ```HIRES```. CHIP will name all the data files corresponding to the first column's id. 

**chipObject.Run( ... , ... )** CHIP's Run method has 2 parameters that are extremely important (both are defaulted to False). The first parameter (use_data) if set to True will use the download spectra and normalized Spectra from your previous run. The second parameter (slow_normalize) if set to True use Alpha normalization instead of the default normalization. The alpha normalization is much slower than the default normalization from specutils but it produces more accurate results. 

## Step 2
You're now all set to run *main_pipeline.py*. 

## Step 3
Once *main_pipeline.py* is completed open *main_The_Cannon.py*. 

You should only need to can change the valariables in the following dotted area to fit your specificities:  
```{python} 
###
###Anything BELOW this point (to the stop point) can be editted to work with your needs
###

...

###
###Anything ABOVE this point (to the start point) can be editted to work with your needs
###
```

Each variable has a comment above them detailing as short as possible what it is used for. 

## Step 4
Now you can run *main_The_Cannon.py*! The defaulted settings hopefully only take your machine a hour or less to run. 

*main_The_Cannon.py* will produce a file called Results.csv. 

Results.csv contains useful information about all the models trained and at the bottom of the file is the best model. 

## Step 5
If you so wish you can plot the results of the best model using Plot_Results.py. 