Step 1: Clone the repo 

Step 2: In your terminal cd to the cloned repo. Mind you this program was only tested on python 3.8.5, it probably works for all python 3.8 but I'm not positive. Create a virutal envorment if you so please. Now with what ever python verison you are using (lets say it is the default python3) type in the terminal 


```{python}pip install -r requirements.txt``` or 
```{python}pip3 install -r requirements.txt```

Step 3: Now you can open up your IDE of choice, and choose either main.py or main.ipynd to run the pipeline. I suggest Jupyter Notebook. 

Step 4: In the main file scroll down to the 

```{python}
if __name__ == '__main__':
```
the code inside this if statement should be the only code you should have to change for your own runs. Now to make sure everything is working change the varriable
```
crossMatchedNames
```
to read in the file "../spocData/testStar.txt". Make sure the Run method is has input parameters of or (),(False,False) or (False,True). All should work. 

Step 5: Run that bad boy

Step 6: When main is completed go to main_The_Cannon.ipynd or main_The_Cannon.py and you should be able to run the code right away without doing anything.  :) 
