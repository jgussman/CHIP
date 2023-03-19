# CHIP Tutorial 

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

Now to begin training, `python src/CHIP.py`. The results of training will end up in a subdir called training_results in the YYYY-MM-DD_HH_mm_SS directory. 


When the training completes you will be able to find a new directory inside the YYYY-MM-DD_HH_mm_SS directory called "training_results". The directory naming convention inside "training_results" is "\<random seed integer>\_\<test fraction>\_\<cost function name>\_<# of k-folds>" which are the parameters set in config.json. In this quick-start tutorial you will find a directory called "3_0.4_Mean Error_2". 

Inside 3_0.4_Mean Error_2 you'll be able to find 7 files:

- CHIP.log : Contains the logs during this training run. Particularly the second to last line of the file tells you the bet parameters for The Cannon model for your dataset. 

- best_model.joblib : TheCannon.model object containing the "best" trained model.

- ds.joblib : TheCannon.dataset object used to train the best model.

- inferred_labels.joblib : A numpy array containing the best model's predictions of the test stars in ds.joblib 

- y_param.joblib : A numpy array containing the "true" parameter values of the test stars in ds.joblib.

- parameters_names.joblib : A list containing the parameter names 

- standard_scaler.joblib : A sklearn.preprocessing.StandardScaler object used to scale all the parameter values. 




---
## Adjust to your own dataset

CHIP is divided into two steps: 
 - The Preprocessing Step
 - The Training Step 



To customize CHIP to your particular dataset you'll primarily be using the [config.json](config.json). The config.json file contains settings for two pipeline stages: preprocessing and training.

For the preprocessing stage, users can specify whether to run it, the path to the cross-match names file, the number of pixels to remove from the left and right edges of each echelle order, and the number of cores to use.

For The Cannon stage, users can specify the run name, the number of cores to use, the random seed, the train-test split fraction, the stellar parameters to use, the cost function to use, any masks to apply, the number of folds for k-fold cross-validation, the batch size, and the polynomial order.

Users should modify the values of the "val" keys to adjust these settings according to their needs. The "description" keys provide further information on what each setting does.
---


---
## Reproduce Results

If you would like to reproduce the results of the paper use the following parameters... *****UPDATE AFTER FINAL RESULTS COME IN


### How-To


1. Get a file with at least one column named "HIRES". The values in this column are the IDs of the stars you want to use from the KOA. The other column must have matching IDs to that of the ID columns in stellar_parameter.csv. An example of this is [starnames_crossmatch_SPOCS_NEXSCI.txt](https://github.com/jgussman/CHIP/blob/updated/data/spocs/starnames_crossmatch_SPOCS_NEXSCI.txt). 

2. Next you will want to configure the [config.json](https://github.com/jgussman/CHIP/blob/updated/src/config.json). You can set the "cores" parameter to a higher number if you want the preprocessing to run faster.

```

    "CHIP": {

        "description": "Arguments for preprocessing pipeline",
    
        "run" : { 
            "val": True,
            "description": "Do you want to run CHIP (true/false)? If you want to use data from a past run, you need to specify the run's name and what data you want to use from that folder ex: [2022-12-28_22-13-57, rv_obs]."
        },
        
        "cross_match_stars" : {
            "val": "data/spocs/testStars.txt",
            "description": "../SPOCSdata/starnames_crossmatch_SPOCS_NEXSCI.txt, ../SPOCSdata/testStars.txt Cross Match Names file path ---> this txt file must have two columns seperated by a space with headers, one of the columns will the names of the stars in the catalogue you want. The other column must have the column name as 'HIRES' and contain the names of the HIRES identification for the stars."
        },

        "trim_spectrum" : {
            "val": 50,
            "description": "Remove this many pixels from the left and right edges of EACH echelle order."
        },

        "cores" : { 
            "val": 1,
            "description": "How many cores on your machine would you like to use? -1 for all."
        }

    }

```

3. Once configured, run the following commands in your anaconda prompt `conda activate chip` then while you are in the root directory of the CHIP run `python src/CHIP.py` 

4. Once step 3 is finished, you can look in `data/chip_runs` folder to find the results of preprocessing. 

1. You must have file that contains the abudances of the stars you would like to train The Cannon on. The first column must be an identifier and the IDs should match up with the non-HIRES column in the next step. Example: [stellar_parameters.csv](https://github.com/jgussman/CHIP/blob/updated/data/spocs/stellar_parameters.csv)

5. To train The Cannon, make the following changes to `src/config.json`, make sure you change The Cannon's run parameter to your specific run folder name. 

```

{   
 
    "CHIP": {

        "description": "Arguments for preprocessing pipeline",
    
        "run" : { 
            "val": false,
            "description": "Do you want to run CHIP (true/false)? If you want to use data from a past run, you need to specify the run's name and what data you want to use from that folder ex: [2022-12-28_22-13-57, rv_obs]."
        },
        
        ...

    },

    "The Cannon":{
        
        "description": "Arguments for The Cannon",

        "run" : { 
            "val": "<YOUR SPECIFIC RUN NAME>",
            "description": "If you want to run The Cannon set val to the name of the dir inside data/chip_runs you want to use (ex: 2022-12-24_06-57-37). Otherwise set to false."
        },

        "cores" : { 
            "val": 1,
            "description": "How many cores on your machine would you like to use? -1 for all."
        },

        "random seed" : { 
            "val": 3,
            "description": "Random seed for reproducability."
        },

        "train test split" :{
            "val": 0.1,
            "description": "what fraction of the data set do you want to use for testing (ex: 0.4, 60% of the data set will be used for training, and 40% will be used for testing the best model)"
        },

        "stellar parameters" : {
            "val": ["TEFF"],
            "description": "A list of the stellar parameters you want to train The Cannon on. The names need to reflect the spelling in data/spocs/stellar_parameters.csv."
        },

        "cost function" : {
            "name": "Mean Error",
            "function" : "lambda true_array,predicted_array:  np.mean( true_array - predicted_array )",
            "description" : "A lambda function and associated name of function, for computing the cost of a model."
        },
 
        "masks" : {
            "val" : [["Iodine","data/constants/masks/by_eye_iodine_mask.npy"], ["None",false]],
            "description" : "List of masks to apply to all stars; for example: [['Name of Mask','file path to mask'],...,['None',False]]; shape of mask = (data/spocs/wl_solution.npy.shape[0],)"
        },
        
        "kfolds" : {
            "val" : 2,
            "description" : "number of folds for kfold cross validation"
        },

        "batch size" : {
            "val" : [3,4],
            "description" : "number of batches to split the training set into for mini-batch training"
        },

        "poly order" : {
            "val" : [1,2],
            "description" : "What degree polynomials do you want The Cannon to try to fit to the data? ex: [1,2]"
        }

    }

}

```

6. The results of The Cannon will be stored in a subfolder called "cannon_results" in the same run folder you specified in the config file.



---

## Reproduce Results from Paper
