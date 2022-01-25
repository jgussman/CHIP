# Instructions 

Step 0: If you do not have an Keck Observatory Archive (KOA) account please click
[here](https://koa.ipac.caltech.edu/cgi-bin/Helpdesk/nph-genTicketForm?projname=KOA). Submit a ticket asking for an account. 

Step 1: Clone this repo 

Step 2: cd into the cloned repo. Now we are going to setup the environment used during testing using [anaconda](https://www.anaconda.com)(if you are on Windows you need to be in the anaconda prompt for the next two commands).


```conda create --name chip python=3.8.5```

```conda activate chip``` 

```cd CHIP```

```pip3 install -r requirements.txt```



If you are on Windows, you need to download [geos_c.dll](https://www.dll-files.com/geos_c.dll.html) then put it in your python envoriment's Library\bin folder. You can read more about this [here](https://github.com/Toblerity/Shapely/pull/1108).

Step 3: cd into the ScriptsAndData folder and now open main_pipeline.py.  

Find the line at the bottom of the script (just CTRL+F __name__). 

```
if __name__ == '__main__':
```
The only code you should ever have to change is in this if statement. 

```{python} 
start_time = time.time()
    ...
time_elap = time.time() - start_time 
print(f"This took {time_elap/60} minutes!")
```

These 3 lines of code all serve the purpose of tracking how long the program took to run. If can delete these 3 lines if you wish. 

For the purposes of making sure everything is working correctly change the path of the variable crossMatchedNames to from "../spocData/starnames_crossmatch_SPOCS_NEXSCI.txt" to "../spocData/testStars.txt". testStars.txt contains ~20 stars so the pipeline shouldn't take long to run. These files contain two columns, the first column is the star’s identifier in any dataset you want (in this case it is the SPOC). The second column must be the identifier for the star in the NEXSCI database. The pipeline will name all the data that is directly used for The Cannon with the names in the first column of the file. 

The only parameter that is not defaulted for the CIR object is the IDs of the stars you want to run through the pipeline. The IDs will be pulled from the second column of whichever filepath you previously specified (in this case "../spocData/testStars.txt").  

To run the pipeline all you need to do is call the method Run on the CIR object. The run method has 2 parameters that are extremely important and both are defaulted to False. The first parameter if set to False will use the download spectra and normalized Spectra from your previous run. The second parameter is to use Alpha normalization instead of the default normalization. The alpha normalization is much slower than the default normalization from specutils but it produces more accurate results. 

Example: If I wanted to use Alpha normalization but not data from my past run of the pipeline I would do 
```{python} 
.Run(False,True)
```

If I wanted to use a past run but not alpha normalization 

```{python} 
.Run(True,False)
```
Etc...

Step 5: Now you can run the file. 

Step 6: Once the pipeline is completed open main_The_Cannon.py. 

Step 7: You can change all the parameter's values in the following area:  
```{python} 
###
###Anything BELOW this point (to the stop point) can be editted to work with your needs
###

...

###
###Anything ABOVE this point (to the start point) can be editted to work with your needs
###
```

Step 8: Now you can run main_The_Cannon.py! Depending your parameters this process can take anywhere from a long time (1 hour) to a long long time (several days). 

Step 9: main_The_Cannon will produce a file called Results.csv. Results.csv contains useful information about all the models it trained and at the bottom of the file is the best model. 

Step 10: If you so wish you can plot the results of the best model using Plot_Results.py. 


# Contact Info
--- 

Email: jgussman@iu.edu 
