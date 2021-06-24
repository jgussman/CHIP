Step 0: If you do not have an Keck Observatory Archive (KOA) account please click
[here](https://koa.ipac.caltech.edu/cgi-bin/Helpdesk/nph-genTicketForm?projname=KOA). Submit a ticket asking for an account. 

Step 1: Clone this repo 

Step 2: cd into the cloned repo. Any verison of Python 3.8 should work, but all testing was done using 3.8.5.  In the terminal type


```pip install -r requirements.txt``` 


or 

```pip3 install -r requirements.txt```

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

For the purposes of making sure everything is working correctly change the path of the variable crossMatchedNames to from "../spocData/starnames_crossmatch_SPOCS_NEXSCI.txt" to "../spocData/testStars.txt". testStars.txt contains ~20 stars so the pipeline shouldn't take long to run. 

The only parameter that is not defaulted for the CIR object is the IDs of the stars you want to run through the pipeline. Make sure the inputed list is a 1D iterable object with Keck/HIRES star IDs. 

To run the pipeline all you need to do is call the method Run on the CIR object. The run method has 2 parameters that are extremely important and both are defaulted to False. The first parameter if set to False will use the download spectra and normalized Spectra from your previous run. The second parameter is to use Alpha normalization instead of the default normalization. The alpha normalization is much slower than the default normalization from specutils but it produces more accurate results. 

Example: If I wanted to use Alpha normalization but not data from my past run of the pipeline I would do .Run(False,True). If I wanted to use a past run but not alpha normalization .Run(True,False). Etc...


Step 5: Now you can run the file. 

Step 6: Once the pipeline is completed open main_The_Cannon.py. You should be able to excute the whole script without problems. As soon as the program has completed you can open up TheCannonReports.csv to check the results of the last run. You can also go into the Element_Pictures folder to look at a visual representation of how The Cannon did with the testing set. 
