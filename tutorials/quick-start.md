
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