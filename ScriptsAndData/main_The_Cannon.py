import numpy as np 
import pandas as pd 
from math import ceil
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats 
from TheCannon import model
from TheCannon import dataset
import TheCannon

###
###Anything BELOW this point (to the stop point) can be editted to work with your needs
###

# N = number of pixels in wavelength file 
# S = number of stars 
# A = number of abudances (labels)
# X = doesn't matter the number 

# Wavelengths for all Stars, Shape = (N,) 
wavelength_file_path = "interpolated_wl.csv"

# Flux for each individual Star, Shape = (S,N)
fluxes_file_path = "fluxes_for_HIRES.npy"

# Ivar for each individual Star, Shape = (S,N)
ivar_file_path = "ivars_for_HIRES.npy"

# The Star's identification, Shape = (S,)
id_file_path = "stellar_names_for_flux_and_ivar.npy"

# Masks to apply for all Stars , example: [("Name of Mask","file Path to mask"),...,("No Mask",False)], shape of mask = (N,)
masks_list = [("No Mask",False)]  # ("iodine","../Constants/Masks/by_eye_iodine_mask.npy"),

#Abudances for all the Parameters, dataframe shape = (X,A)
# The 0th column will be converted to the index!!! This means the 0th column needs to be the identifier corresponding 
# to the 0th column of the file3 you put in for crossMatchedNames in main_pipeline.py
abundances_file_path = "../spocData/df_all.csv"

# Parameter names that reflect the spelling in the file for abundances_file_path
parameters = ['TEFF', 'LOGG','VSINI', 'FeH', 'CH', 'NH','OH','NaH', 'MgH', 'AlH', 'SiH', 'CaH', 'TiH', 'VH', 'CrH', 'MnH','NiH', 'YH']  

# Random Seed for replication
random_seed = 3

# Number of Parameters to Train at the same time list, must be an int and can't be left empty
group_sizes_list = [8] #1,2,3,4,5,6,7,8

# The % of the stars that will be used for testing set
testing_percentage = 10#%

# The % of the training stars that will be used for validation set
validation_percentage = 10#%

# Name and function for computing loss for model
loss_metric_name = "(mu-mu*)/mu"
loss_metric_fun = lambda true_array,predicted_array:  (np.mean(true_array) - np.mean(predicted_array)) / np.mean(true_array)

# Scale the abudances list, [("Name of Scaler",instance of the scaler class),...,("No Scaler",False)], the instances must have a fit_transform
abundance_scaler_list = [("std_scaler",StandardScaler())] # ,("No Scaler",False) ,("MinMaxScaler",MinMaxScaler())

###
###Anything ABOVE this point (to the start point) can be editted to work with your needs
###

# Loading in the data 
wl = np.genfromtxt(wavelength_file_path,skip_header=1)[::-1]
fluxes = np.load(fluxes_file_path).T
ivars = np.load(ivar_file_path).T
ids = np.load(id_file_path,allow_pickle=True)
abundances_df = pd.read_csv(abundances_file_path,index_col=0)[parameters] 
abundances_file_labels = abundances_df.to_numpy()
testing_percentage /= 100
validation_percentage /= 100

print(testing_percentage)


# Removes Spaces from the IDs in the df
remove_spaces_f = lambda s: s.replace(" ","")
abundances_df.rename(index = remove_spaces_f,inplace=True)

# Sort df to match ids' order (this also removes stars that we don't have flux&ivars for)
#this is "slow" but it is fine
new_abundances_df = abundances_df.iloc[0:0] #Makes a dataframe with no data but same columns
abundances_array = np.zeros((1,len(parameters)))
new_fluxes,new_ivars,new_ids,id_list = [],[],[],[]
for i in range(len(ids)): 
    id = ids[i]
    # Is the star in the dataframe
    bool_array = abundances_df.index == id
    star_abundances = abundances_df[bool_array]
    new_abundances_df = new_abundances_df.append(star_abundances)
    #If an ID is not found then the flux and ivar don't move on 
    if np.any(bool_array):
        new_fluxes.append(fluxes[i])
        new_ivars.append(ivars[i])
        new_ids.append(id)
fluxes = np.array(new_fluxes)
ivars  = np.array(new_ivars)
ids    = np.array(new_ids)
abundances_array = new_abundances_df.to_numpy()


# Creation of Test, Training and Validation set
flux_train, flux_test, abun_train, abun_test = train_test_split(fluxes, abundances_array,
                test_size=testing_percentage,
                random_state=random_seed) 
ivar_train, ivar_test, id_train, id_test = train_test_split(ivars, ids,
                test_size=testing_percentage,
                random_state=random_seed) 
flux_train, flux_valid, abun_train, abun_valid = train_test_split(flux_train, abun_train,
                test_size=validation_percentage,
                random_state=random_seed) 
ivar_train, ivar_valid, id_train, id_valid = train_test_split(ivar_train, id_train,
                test_size=validation_percentage,
                random_state=random_seed) 


def TrainingBuddys(labels_list,size):
    '''
    Assigns which labels will be trained with which other labels, 
    based on the order they are in labels_list and size 
    Ex: 
    labels_list = ['CH', 'NH', 'OH','NaH', 'MgH', 'AlH', 'SiH']
    size = 3
    Then the output will be [['CH', 'NH', 'OH'],['NaH', 'MgH', 'AlH'],['SiH']]
    
    Input: labels_list (list)-> column names of abudances df 
           size (int)-> maximum # of memebers in each training group

    Output: list of lists 

    *****DO NOT CHANGE THIS UNLESS YOU WILL CHANGE THE ABUDANCE SLICING 
        CODE RIGHT BEFORE THE MODEL IS TRAINED*****
    '''
    groups_list = []
    for i in range(ceil(len(labels_list)/size)):
        groups_list.append(labels_list[i*size:i*size + size])
    return groups_list
        




num_training_parameters = len(parameters) 
results_df = pd.DataFrame(columns=["# Stars","test %","Valid %","Seed #","Mask","Scale Abun","Loss-Metric"] + [label for label in parameters] + ['Training Groups'])



for mask_info in masks_list:
    mask_name, mask_file_path = mask_info
    if mask_file_path:
        mask = np.load(mask_file_path)
        u_flux_train = flux_train * mask
        u_flux_valid = flux_valid * mask 
        u_ivar_train = ivar_train * mask
        u_ivar_valid = ivar_valid * mask 
    else: 
        u_flux_train = flux_train 
        u_flux_valid = flux_valid 
        u_ivar_train = ivar_train 
        u_ivar_valid = ivar_valid 

    for scaler_info in abundance_scaler_list:
        scaler_name,scaler_instance = scaler_info 
	# Must apply scale later since the shape needs to have the same columns 
        

        for size in group_sizes_list:
            training_groups = TrainingBuddys(parameters,size)

            #Format for Results
            group_results = {"# Stars":len(ids), 
                            "test %" :testing_percentage *100, 
                            "Valid %": validation_percentage *100, 
                            "Seed #":random_seed,
                            "Mask": mask_name,
                            "Scale Abun":scaler_name,
                            "Loss-Metric":loss_metric_name}
            # Add your labels to the results
            for label in parameters:
                group_results[label] = float('-inf')
            group_results['Training Groups'] = training_groups

            for i in range(len(training_groups)):
                group_parameters = training_groups[i]
                j = size*i
                k = j + size
                # Apply Scaling Abudances
                if scaler_instance:
                    u_abun_train = scaler_instance.fit_transform(abun_train[:,j:k])
                else:
                    u_abun_train = abun_train
                    u_abun_valid = abun_valid
                # Training the model
                ds = dataset.Dataset(wl, id_train, u_flux_train, u_ivar_train, u_abun_train, id_valid, u_flux_valid, u_ivar_valid)
                ds.set_label_names(group_parameters) 
                ds.ranges= [[min(wl),max(wl)]]
                # #Doesn't change if we change 1 to anything else
                md = model.CannonModel(1, useErrors=False)
                md.fit(ds)

                # Infer Labels & Usefull Plots
                label_errs = md.infer_labels(ds)
                infered_valid_labels = ds.test_label_vals

                #Calculate MSE and store
                for y in range(size):
                    if j+y < num_training_parameters:
                        if scaler_instance:
                            u_infered_valid_labels = scaler_instance.inverse_transform(infered_valid_labels)
                        else:
                            u_infered_valid_labels = infered_valid_labels
 

                        group_results[parameters[j+y]] = loss_metric_fun(abun_valid[:,j+y],u_infered_valid_labels[:,y])
            results_df = results_df.append([group_results])


#Reapply Train Test Split 
#so the validation set is inside the training set
flux_train, flux_test, abun_train, abun_test = train_test_split(fluxes, abundances_array,
                test_size=testing_percentage,
                random_state=random_seed) 
ivar_train, ivar_test, id_train, id_test = train_test_split(ivars, ids,
                test_size=testing_percentage,
                random_state=random_seed) 



best_model = results_df.iloc[0]
for row in range(1,results_df.shape[0]-1):
    current_best_model_score = 0
    temp_model_score         = 0 
    temp_model = results_df.iloc[row]

    #Compare the current best model and the next model
    #which ever model does performs better on more parameters 
    #will be the best model 
    for param in parameters: 
        best_m_param = np.abs(best_model[param]) 
        temp_m_param = np.abs(temp_model[param])
        if best_m_param < temp_m_param:
            current_best_model_score += 1 
        elif best_m_param > temp_m_param:
            temp_model_score += 1 
        #On the off chance they are equal neither gets a point
    if current_best_model_score < temp_model_score:
        best_model = temp_model

print(f'''
The best model has the follow results 
{best_model}
''')

#Format for Results
group_results = {"# Stars":len(ids), 
                "test %" :best_model["test %"] *100, 
                "Valid %": best_model["Valid %"] *100, 
                "Seed #":best_model["Seed #"],
                "Mask": best_model["Mask"],
                "Scale Abun":best_model["Scale Abun"],
                "Loss-Metric":best_model["Loss-Metric"]}
# Add your labels to the results
for label in parameters:
    group_results[label] = float('-inf')
group_results['Training Groups'] = training_groups

## Apply Best Mask
best_mask_name = best_model["Mask"]
mask_file_path = [mask[1] for mask in masks_list if mask[0] == best_mask_name][0]
if mask_file_path:
    mask = np.load(mask_file_path)
    u_flux_train = flux_train * mask
    u_flux_test  = flux_test * mask 
    u_ivar_train = ivar_train * mask
    u_ivar_test  = ivar_test * mask
else: 
    u_flux_train = flux_train 
    u_flux_test  = flux_test 
    u_ivar_train = ivar_train 
    u_ivar_test  = ivar_test

## Apply Best Scaler
best_scaler_name = best_model["Scale Abun"]
scaler_instance = [scaler[1] for scaler in abundance_scaler_list if scaler[0] == best_scaler_name][0]


## Best Training Groups 
training_groups = best_model["Training Groups"]
size = len(training_groups[0])

def SaveTrueAndPredicted(true_label,predicted_label,label_name):
    '''
    Save data to npy files 
    '''
    both_true_and_predicted = np.vstack((true_label,predicted_label))
    np.save(f"Element_Data/{label_name}.npy",both_true_and_predicted)

# Training and Testing
for i in range(len(training_groups)):
    group_parameters = training_groups[i]
    j = size*i
    k = j + size
    if scaler_instance:
        u_abun_train = scaler_instance.fit_transform(abun_train[:,j:k])
    else:
        u_abun_train = abun_train
        u_abun_test = abun_test

    # Training the model
    ds = dataset.Dataset(wl, id_train, u_flux_train, u_ivar_train, u_abun_train, id_test, u_flux_test, u_ivar_test)
    ds.set_label_names(group_parameters) 
    ds.ranges= [[min(wl),max(wl)]]
    # #Doesn't change results if we change 1 to anything else
    md = model.CannonModel(1, useErrors=False)
    md.fit(ds)

    # Infer Labels & Usefull Plots
    label_errs = md.infer_labels(ds)
    md.diagnostics_leading_coeffs(ds)
    md.diagnostics_plot_chisq(ds)
    infered_test_labels = ds.test_label_vals

    #Calculate MSE and store
    for y in range(size):
        if j+y < num_training_parameters:
            u_abun_test = abun_test
            if scaler_instance:
                u_infered_test_labels = scaler_instance.inverse_transform(infered_test_labels)
            else:
                u_infered_test_labels = infered_test_labels

            group_results[parameters[j+y]] = loss_metric_fun(u_abun_test[:,j+y],u_infered_test_labels[:,y])
            SaveTrueAndPredicted(u_abun_test[:,j+y],u_infered_test_labels[:,y],parameters[j+y])
    results_df = results_df.append([group_results])
results_df.to_csv("Results")
print("The very last model added to Results.csv is the model trained with the best hyperparameters")
print("Run Plot_Results.py to graph the results")

