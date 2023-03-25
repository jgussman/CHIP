# CHIP Tutorial 

---
## Table of Contents
 - [Quick-start Tutorial](#quick-start-tutorial)
 - [Use your own dataset](#use-your-own-dataset)
 - [Reproduce Results](#reproduce-results)

--- 

## Quick-start Tutorial

To ensure that CHIP is set up correctly, you can run the preconfigured config.json file. First, activate the *chip* conda environment with `conda activate chip` and make sure that the current working directory is the root directory of CHIP. Then, execute the following command in the terminal: `python src/CHIP.py`. After a few seconds, you will be prompted to enter your KOA user credentials.


After you initiate the preprocessing step in the CHIP Python environment, you can find a new subdirectory in the "data" directory named "chip_runs". Each subdirectory inside "chip_runs" represents a different run instance, and is named after the GMST (Greenwich Mean Sidereal Time) you started the preprocessing step, using the format: YYYY-MM-DD_HH_mm_SS. You should take note of the name of the newest directory, which will be the most recent preprocessing run instance.

After preprocessing is complete, open the config.json file and change two arguments. 

First, change the "Preprocessing" run argument's value from true to false. It should look like this:

```
    "Preprocessing": {

        "description": "Arguments for preprocessing.",
    
        "run" : { 
            "val": false,
            "description": "Do you want to run CHIP (true/false)? If you want to use data from a past run, you need to specify the run's name and what data you want to use from that folder ex: [2022-12-28_22-13-57, rv_obs]."
        },
```

Second, change the Training run argument's value from false to the name of the newest directory you noted earlier. It should look similar to this:

```
    "Training":{
    
        "description": "Arguments for training.",

        "run" : { 
            "val": "YYYY-MM-DD_HH_mm_SS",
            "description": "If you want to run The Cannon set val to the name of the dir inside data/chip_runs you want to use (ex: 2022-12-28_22-13-57). Otherwise set to false."
        },
```

Now to begin training, run `python src/CHIP.py` from the main CHIP directory. When the training completes, a new subdirectory called "training_results" will be created inside the YYYY-MM-DD_HH_mm_SS directory. The directory naming convention inside "training_results" is "\<random seed integer>\_\<test fraction>\_\<cost function name>\_<# of k-folds>", where the variables are the parameters set in config.json. In this quick-start tutorial, you will find a directory called "3_0.4_Mean Error_2". 

Inside 3_0.4_Mean Error_2 you'll be able to find 7 files:

- CHIP.log : Contains the logs during this training run. The second-to-last line of the log file tells you the best-fit parameters for The Cannon model for your dataset. 

- best_model.joblib : [TheCannon.model object](https://annayqho.github.io/TheCannon/_modules/TheCannon/model.html) containing the "best" trained model.

- ds.joblib : [TheCannon.dataset object](https://annayqho.github.io/TheCannon/api.html#TheCannon.dataset.Dataset) used to train the best model.

- inferred_labels.joblib : A numpy array containing the best model's parameter predictions for the test stars in ds.joblib 

- y_param.joblib : A numpy array containing the "true" parameter values of the test stars in ds.joblib.

- parameters_names.joblib : A list containing the parameter names.

- standard_scaler.joblib : A sklearn.preprocessing.StandardScaler object used to scale all the parameter values. 

You can use the [joblib library](https://joblib.readthedocs.io/en/latest/index.html) to [load](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html) any of these files joblib files. For example, you can load the best model and make predictions with it.

```
import joblib 

model = joblib.load("best_model.joblib")

model.infer_labels( <TheCannon.dataset object> ) 

inferred_labels = ds.test_label_vals
```

That concludes the quick-start tutorial. 

---
---
## Use your own dataset

If you'd now like to apply CHIP to your own dataset, CHIP also allows you to specify stars and spectra on which your model will be trained. We assume in this tutorial that you have followed the quick-start tutorial before trying your own dataset. 

To customize CHIP to your dataset, you will primarily adjust the [config.json](config.json) file. This file contains settings for two pipeline stages: preprocessing and training.


The preprocessing arguments are at the top of the config file: 

```
{   
 
    "Preprocessing": {

        "description": "Arguments for preprocessing.",
    
        "run" : { 
            "val": true,
            "description": "Do you want to run CHIP (true/false)? If you want to use data from a past run, you need to specify the run's name and what data you want to use from that folder ex: [2022-12-28_22-13-57, rv_obs]."
        },
        
        "cross_match_stars" : {
            "val": "data/spocs/tutorialStars.txt",
            "description": "Cross Match Names file path ---> this txt file must have two columns seperated by a space with headers, one of the columns will the names of the stars in the catalogue you want. The other column must have the column name as 'HIRES' and contain the names of the HIRES identification for the stars. Ex: data/spocs/starnames_crossmatch_SPOCS_NEXSCI.txt"
        },

        "trim_spectrum" : {
            "val": 50,
            "description": "Remove this many pixels from the left and right edges of EACH echelle order."
        },

        "cores" : { 
            "val": 2,
            "description": "How many cores on your machine would you like to use? -1 for all."
        }

    }, 
```

The training arguments are in the latter half of the file: 

```
    "Training":{
        
        "description": "Arguments for training.",

        "run" : { 
            "val": false,
            "description": "If you want to run The Cannon set val to the name of the dir inside data/chip_runs you want to use (ex: 2022-12-28_22-13-57). Otherwise set to false."
        },

        "stellar parameters path" : {
            "val": "data/spocs/stellar_parameters.csv",
            "description": "Path to the stellar parameters file. The file must have a column named 'HIRESID' with the same names as the cross match stars file."
        },

        "stellar parameters" : {
            "val": ["TEFF"],
            "description": "A list of the stellar parameters you want to train The Cannon on. The names need to reflect the spelling in stellar parameters path"
        },

        "cores" : { 
            "val": 2,
            "description": "How many cores on your machine would you like to use? -1 for all."
        },

        "random seed" : { 
            "val": 3,
            "description": "Random seed for reproducability."
        },

        "train test split" :{
            "val": 0.4,
            "description": "what fraction of the data set do you want to use for testing (ex: 0.4, 60% of the data set will be used for training, and 40% will be used for testing the best model)"
        },

        "cost function" : {
            "name": "Mean Error",
            "function" : "lambda true_array,predicted_array:  np.mean( true_array - predicted_array )",
            "description" : "A lambda function and associated name of function, for computing the cost of a model."
        },

        "masks" : {
            "val" : [ ["None",false]],
            "description" : "List of masks to apply to all stars; for example: [['Name of Mask','file path to mask'],...,['None',False]]; shape of mask = (data/spocs/wl_solution.npy.shape[0],)"
        },

        "epochs" : {
            "val" : 1,
            "description" : "number of epochs to train the model for"
        },
        
        "kfolds" : {
            "val" : 2,
            "description" : "number of folds for kfold cross validation"
        },

        "batch size" : {
            "val" : [4],
            "description" : "number of batches to split the training set into for mini-batch training"
        },

        "poly order" : {
            "val" : [2],
            "description" : "What degree polynomials do you want The Cannon to try to fit to the data? ex: [1,2]"
        }

    }

}
```

Users should modify the values of the "val" keys to adjust these settings according to their needs. The "description" keys provide further information on what each setting does. 


1. Get a file with at least one column named "HIRES". The values in this column are the IDs of the stars you want to use from the KOA. The other column must have matching IDs to that of the ID columns in the file you will set for Training's stellar parameters path. 
    ```
    "Training":{
        
        ...

        "stellar parameters path" : {
            "val": "data/spocs/stellar_parameters.csv",
            "description": "Path to the stellar parameters file. The file must have a column named 'HIRESID' with the same names as the cross match stars file."
        },
        
        ...
    ```

    An example of this is [tutorialStars.txt](../data/spocs/tutorialStars.txt). 

2. Next you will want to configure the "Preprocessing" section of [config.json](config.json). 

3. Once configured, run `conda activate chip` to activate your anaconda environment. 

4. Then, from the root directory of CHIP, run `python src/CHIP.py` to complete the pre-processing step. The results of the pre-processing will be located in the `data/chip_runs` directory (described in more detail in the quick-start tutorial).  

5. You must have a file that contains the parameter values of the stars you would like to train the models on. The first column must be an identifier and the IDs MUST match up with the non-HIRES column in. Example: [stellar_parameter.csv](../data/spocs/stellar_parameters.csv)

6. To train, set the `run` parameter in the `Training` section of `src/config.json` to your specific run folder name. Make sure that the `run` parameter in the `Preprcoessing` section of `src/config.json` is set to `false`. Then, you are ready to train your model by running `python src/CHIP.py` from the root directory of CHIP.
---
---

## Reproduce Results

If you would like to reproduce the results of the paper use the following parameters... *****UPDATE AFTER FINAL RESULTS COME IN
