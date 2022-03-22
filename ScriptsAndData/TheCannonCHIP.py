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

def TheCannonCHIP(wavelength_file_path,fluxes_file_path,ivar_file_path,id_file_path,masks_list,abundances_file_path,parameters,random_seed,group_sizes_list,testing_percentage, validation_percentage, loss_metric_name, loss_metric_fun):
    '''
    Trains several The Cannon 2 Models on given data. Creates Results.csv for storing the results

    ---Notation---
    N = number of pixels in wavelength file 
    S = number of stars 
    A = A = number of labels (number of stellar parameters fitted with the model)
    X = doesn't matter the number 
    -------------

    Input:
        wavelength_file_path =  Wavelengths for all Stars, Shape = (N,) 
        fluxes_file_path = Flux for each individual Star, Shape = (S,N)
        ivar_file_path =  Ivar for each individual Star, Shape = (S,N)
        id_file_path = The Star's identification, Shape = (S,)
        masks_list =  Masks to apply for all Stars, Shape = (N,)
        abundances_file_path =  Abudances for all the Parameters, dataframe shape = (X,A)
        parameters = Parameter names that reflect the spelling in the file for abundances_file_path 
        random_seed = Random Seed for replication
        group_sizes_list = Number of Parameters to Train at the same time list, must be an int and can't be left empty
        testing_percentage = The % of the stars that will be used for testing set 
        validation_percentage = The % of the training stars that will be used for validation set
        loss_metric_name = Name of the function for computing loss for model
        loss_metric_fun = function for computing loss for model

    Output: 
        None
    '''
 
    # Scale the abudances list, [("Name of Scaler",instance of the scaler class),...,("No Scaler",False)], the instances must have a fit_transform
    # Found that Minmaxscaler and std_scaler get pretty much the same results, while No scaler gets a memory error 
    abundance_scaler_list = [("std_scaler",StandardScaler())] # ,("No Scaler",False) 

    # Loading in the data 
    wl = np.genfromtxt(wavelength_file_path,skip_header=1)[::-1]
    fluxes = np.load(fluxes_file_path).T
    ivars = np.load(ivar_file_path).T
    ids = np.load(id_file_path,allow_pickle=True)
    abundances_df = pd.read_csv(abundances_file_path,index_col=0)[parameters] 
    abundances_file_labels = abundances_df.to_numpy()
    testing_percentage /= 100
    validation_percentage /= 100


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
            

    def ApplyMask(file_path,flux_tr,flux_te,ivar_tr,ivar_te):
        '''
        Applies mask if there is a given file path
        '''
        if mask_file_path:
            mask = np.load(mask_file_path)
            u_flux_tr = np.copy(flux_tr) * mask
            u_flux_te  = np.copy(flux_te) * mask 
            u_ivar_tr = np.copy(ivar_tr) * mask
            u_ivar_te  = np.copy(ivar_te) * mask
        else: 
            u_flux_tr = np.copy(flux_tr)
            u_flux_te  = np.copy(flux_te)
            u_ivar_tr = np.copy(ivar_tr)
            u_ivar_te  = np.copy(ivar_te)
        return u_flux_tr, u_flux_te, u_ivar_tr, u_ivar_te

    num_stars = len(ids)
    def IntialParamRecording(s,t_p,v_p,m_n,s_n,l_m_n):
        g_results = {"# Stars":num_stars, 
                                "Test %" :t_p *100, 
                                "Valid %": v_p *100, 
                                "Seed":random_seed,
                                "Group Size": s,
                                "Mask": m_n,
                                "Scale Abun":s_n,
                                "Loss-Metric":l_m_n}
        # Add your labels to the results
        for label in parameters:
            g_results[label] = float('-inf')
        return g_results




    # Create Df for storing Results
    results_df = pd.DataFrame(columns=["# Stars","Test %","Valid %","Seed","Group Size","Mask","Scale Abun","Loss-Metric"] + [label for label in parameters])
    num_training_parameters = len(parameters) 

    for mask_info in masks_list:
        mask_name, mask_file_path = mask_info
        u_flux_train,u_flux_valid,u_ivar_train,u_ivar_valid = ApplyMask(mask_file_path,flux_train,flux_test,ivar_train,ivar_test) 

        for scaler_info in abundance_scaler_list:
            scaler_name,scaler_instance = scaler_info 
            # Must apply scale later since the shape needs to have the same columns 
            

            for size in group_sizes_list:
                training_groups = TrainingBuddys(parameters,size)
                group_results = IntialParamRecording(size,testing_percentage,validation_percentage,mask_name,scaler_name,loss_metric_name)
                

                for i in range(len(training_groups)):
                    group_parameters = training_groups[i]
                    j = size*i
                    k = j + size
                    # Apply Scaling Abudances
                    if scaler_instance:
                        u_abun_train = scaler_instance.fit_transform(abun_train[:,j:k])
                    else:
                        u_abun_train = abun_train

                    # Training the model
                    ds = dataset.Dataset(wl, id_train, u_flux_train, u_ivar_train, u_abun_train, id_valid, u_flux_valid, u_ivar_valid)
                    ds.set_label_names(group_parameters) 
                    ds.ranges= [[min(wl),max(wl)]]
                    # #Doesn't change if we change 1 to anything else
                    md = model.CannonModel(1, useErrors=False)
                    md.fit(ds)

                    # Infer labels & useful plots
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


    # Find Best Model
    best_model = results_df.iloc[0]
    for row in range(1,results_df.shape[0]):
        print("Row number ",row)
        current_best_model_score = 0
        temp_model_score         = 0 
        temp_model = results_df.iloc[row]

        #Compare the current best model and the next model
        #which ever model does performs better on more parameters 
        #will be the best model 
        curr_model_score = np.sum(np.abs(best_model[parameters]) < np.abs(temp_model[parameters])) 
        if curr_model_score < num_training_parameters:
            best_model = temp_model
            

    print(f'''
    The best model has the follow results 
    {best_model}
    ''')

    group_results = IntialParamRecording(best_model["Group Size"],testing_percentage,validation_percentage, best_model["Mask"],best_model["Scale Abun"],best_model["Loss-Metric"])
    
    
    ## Apply Best Mask
    best_mask_name = best_model["Mask"]
    mask_file_path = [mask[1] for mask in masks_list if mask[0] == best_mask_name][0] # Find the mask tuple
    u_flux_train,u_flux_test,u_ivar_train,u_ivar_test = ApplyMask(mask_file_path,flux_train,flux_test,ivar_train,ivar_test)

    ## Apply Best Scaler
    best_scaler_name = best_model["Scale Abun"]
    scaler_instance = [scaler[1] for scaler in abundance_scaler_list if scaler[0] == best_scaler_name][0]


    ## Best Training Groups 
    size = best_model["Group Size"]
    training_groups = TrainingBuddys(parameters,size)

    def SaveTrueAndPredicted(true_label,predicted_label,label_name):
        '''
        Save data to npy files 
        '''
        both_true_and_predicted = np.vstack((true_label,predicted_label))
        np.save(f"Best_Model_Results/{label_name}.npy",both_true_and_predicted)

    # Training 
    for i in range(len(training_groups)):
        group_parameters = training_groups[i]
        j = size*i
        k = j + size
        if scaler_instance:
            u_abun_train = scaler_instance.fit_transform(abun_train[:,j:k])
        else:
            u_abun_train = abun_train

        # Training the model
        ds = dataset.Dataset(wl, id_train, u_flux_train, u_ivar_train, u_abun_train, id_test, u_flux_test, u_ivar_test)
        ds.set_label_names(group_parameters) 
        ds.ranges= [[min(wl),max(wl)]]
        # #Doesn't change results if we change 1 to anything else
        md = model.CannonModel(1, useErrors=False)
        md.fit(ds)

        # Infer labels & useful plots
        label_errs = md.infer_labels(ds)
        # md.diagnostics_leading_coeffs(ds)
        # md.diagnostics_plot_chisq(ds)
        infered_test_labels = ds.test_label_vals

        #Calculate score and store
        for y in range(size):
            if j+y < num_training_parameters:
                if scaler_instance:
                    u_infered_test_labels = scaler_instance.inverse_transform(infered_test_labels)
                else:
                    u_infered_test_labels = infered_test_labels

                group_results[parameters[j+y]] = loss_metric_fun(abun_test[:,j+y],u_infered_test_labels[:,y])
                SaveTrueAndPredicted(abun_test[:,j+y],u_infered_test_labels[:,y],parameters[j+y])
        results_df = results_df.append([group_results])
        
    results_df.to_csv("Results",index=False)
    print("The very last model added to Results.csv is the model trained with the best hyperparameters and whole training set (including the validation set)")
    print("Run Plot_Results.py to graph the results")

