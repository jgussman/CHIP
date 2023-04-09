
## Use your own dataset

If you'd now like to apply CHIP to your own dataset, CHIP also allows you to specify stars and spectra on which your model will be trained. We assume in this tutorial that you have followed the quick-start tutorial before trying your own dataset. 

To customize CHIP to your dataset, you will primarily adjust the [config.json](../config/config.json) file. This file contains settings for two pipeline stages: pre-proocessing and training.

    Note: all paths in the config file are relative to the current working directory at the time of running CHIP. Or you can specify an absolute path to the file. 


The pre-proocessing arguments are at the top of the config file: 

```
{
    "Pre-processing": {

        "description": "Arguments for pre-proocessing.",
    
        "run" : { 
            "val": true,
            "description": "Do you want to run CHIP (true/false)? If you want to use data from a past run, you need to specify the run's name and what data you want to use from that folder ex: [2022-12-28_22-13-57, rv_obs]."
        },
        
        "HIRES stars IDs" : {
            "val": "data/spocs/one_star.txt",
            "description": "Path to the file containing a column with a header of 'HIRESID' with the names (using KOA conventions) of the stars you want to use."
        },

        "trim spectrum" : {
            "val": 0,
            "description": "Remove this many pixels from the left and right edges of EACH echelle order."
        },

        "cores" : { 
            "val": -1,
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

2. Next you will want to configure the "Pre-processing" section of [config.json](config.json). 

3. Once configured, run `conda activate chip` to activate your anaconda environment. 

4. Then, from the root directory of CHIP, run `python src/CHIP.py` to complete the pre-processing step. The results of the pre-processing will be located in the `data/chip_runs` directory (described in more detail in the quick-start tutorial).  

5. You must have a file that contains the parameter values of the stars you would like to train the models on. The first column must be an identifier and the IDs MUST match up with the non-HIRES column in. Example: [stellar_parameter.csv](../data/spocs/stellar_parameters.csv). 

6. To train, set the `run` parameter in the `Training` section of the [config.json](../config/config.json) to the specific run folder name you would like to train using. You can choose any folder inside your [data/chip_runs](../data/chip_runs/) folder. Make sure that the `run` parameter in the `Preprcoessing` section of [config.json](../config/config.json) is set to `false`. Then, you are ready to train your model by running `python src/CHIP.py` from the root directory of CHIP.
---
---