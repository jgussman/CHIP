# Tutorial 

## How-To

1. You must have file that contains the abudances of the stars you would like to train The Cannon on. The first column must be an identifier and the IDs should match up with the non-HIRES column in the next step. Example: [stellar_parameters.csv](https://github.com/jgussman/CHIP/blob/updated/data/spocs/stellar_parameters.csv)

1. Get a file with at least one column named "HIRES". The values in this column are the IDs of the stars you want to use from the KOA. The other column must have matching IDs to that of the ID columns in step 1. An example of this is [starnames_crossmatch_SPOCS_NEXSCI.txt](https://github.com/jgussman/CHIP/blob/updated/data/spocs/starnames_crossmatch_SPOCS_NEXSCI.txt). 

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
