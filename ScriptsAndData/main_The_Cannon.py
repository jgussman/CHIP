import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from TheCannon import model
from TheCannon import dataset
import TheCannon
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
### wl
wl = np.genfromtxt("interpolated_wl.csv",skip_header=1)[::-1]

ID_and_flux = np.load("fluxes_for_HIRES.npy")
ivar = np.load("ivars_for_HIRES.npy")

#tr_ID
tr_ID = np.load("stellar_names_for_flux_and_ivar.npy",allow_pickle=True)
# #tr_flux
tr_flux = ID_and_flux.transpose()
#tr_ivar
tr_ivar = ivar.T
#Telluric (If you want to use the telluric mask add the path otherwise, set to False)
telluric_q = False #'../Constants/Masks/telluric_mask.txt'
#Iodine Mask (If you want to use the telluric mask add the path otherwise, set to False)
iodine_q = "../Constants/Masks/by_eye_iodine_mask.npy" 


# tr_label
#'TEFF', 'LOGG', 'VSINI', 'CH', 'NH', 'OH',
#'NaH', 'MgH', 'AlH', 'SiH', 'CaH', 'TiH', 'VH', 'CrH', 'MnH', 'FeH',    
#       'NiH', 'YH', 'DIR'
labels = ['CH', 'NH', 'OH'] #Labels you want to use
d = pd.read_csv("../spocData/df_all.csv",index_col=0)
d = d[labels] #Slicing the labels you want 

testing_percentage = 0.10
#Random seed to check
random_seed_start = 3
random_seed_end = 4


#***You shouldn't have to change anything beyond this point***

tr_label = d.to_numpy()

removeList = []
for i in d.index:
    test = i.replace(" ","")
    if test not in tr_ID:
        removeList.append(i)
for name in removeList:
    d = d.drop(name)  

    
index_d = d.index
index_d = np.array([i.replace(" ","") for i in index_d])
length_d = len(index_d)
restruc = []
array_d = d.to_numpy()
checking_index = 0
for i in range(len(tr_ID)):
    temp_ID = tr_ID[i-checking_index]
    loc = np.where(index_d == temp_ID)
    
    temp_list = []
    for l in array_d[loc]:
        for j in l:
            temp_list.append(float(j))
    if len(temp_list) == 0:
        tr_ID = np.delete(tr_ID,i-checking_index)
        tr_flux = np.delete(tr_flux,i-checking_index,0)
        tr_ivar = np.delete(tr_ivar,i-checking_index,0)
        checking_index+=1 
    else:
        restruc.append(temp_list)
tr_label = np.array(restruc)


if telluric_q: 
    telluric = np.genfromtxt(telluric_q)
    tr_flux *= telluric
    tr_ivar *= telluric
if iodine_q:
    iodine_mask = np.load(iodine_q)
    tr_flux *= iodine_mask
    tr_ivar *= iodine_mask
    

t1, t2, t3, t4 = tr_ID, tr_flux,tr_ivar,tr_label
 
for RS in range(random_seed_start,random_seed_end):
    #RS = randseed #Random Seed # 
    tr_ID, tr_flux,tr_ivar,tr_label = t1, t2, t3, t4
    
    np.random.seed(RS)
    train_ID, test_ID, tr_flux, test_flux = train_test_split(tr_ID, tr_flux, test_size = testing_percentage)
    np.random.seed(RS)
    tr_ID2, _, tr_ivar, test_ivar = train_test_split(tr_ID, tr_ivar, test_size = testing_percentage)
    np.random.seed(RS)
    tr_ID, _, tr_label, true_test_labels = train_test_split(tr_ID, tr_label, test_size = testing_percentage)

    ds = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

    
    ds.set_label_names(labels)

    ds.ranges= [[min(wl),max(wl)]]

    md = model.CannonModel(1, useErrors=False)
    md.fit(ds)


    md.diagnostics_leading_coeffs(ds)
    md.diagnostics_plot_chisq(ds)
    label_errs = md.infer_labels(ds)
    Cannon_test_labels = ds.test_label_vals
    #Mean-Squared Error 
    good_for_reports = True 
    for lab in ['TEFF', 'LOGG', 'VSINI']:
        if lab not in labels:
            good_for_reports = False
    if good_for_reports:
        results_log = pd.read_csv("TheCannonReports.csv")
        results = {"# of Stars":len(tr_ID) + len(test_ID), 
                   "test %" :testing_percentage *100, 
                   "Seed #":RS,
                   "Telluric": telluric_q,
                   "MSE T_{eff}":float('-inf'),
                   "MSE log g":float('-inf'),
                  "MSE vsini":float('-inf'),
                  "MSE [Fe/H]":float('-inf'),
                  "Notes": "NA"}
        for i in range(len(labels)):
            results["MSE " + labels[i]] = mean_squared_error(true_test_labels[:,i],Cannon_test_labels[:,i])

        results_log = results_log.append(results,ignore_index=True)
        results_log.to_csv("TheCannonReports.csv",index=False)

    def MakeTrueVsPredictedPlots(true_label_vals,predicted_label_vals,col_num,labels):
        '''

        '''
        x = true_label_vals[:,col_num]
        y = predicted_label_vals[:,col_num]
        res = stats.linregress(x, y)
        plt.plot(x, y, 'o', label='true')
        plt.plot(x, res.intercept + res.slope*x, 'r', label='Best Fit')
        plt.legend()
        plt.title(f"RS={RS},Test\%={testing_percentage}")
        plt.xlabel(f'true ${labels[col_num]}$')
        plt.ylabel(f'predicted ${labels[col_num]}$')
        temp_name_change = labels[col_num]
        plt.savefig(f"Element_Pictures/{temp_name_change}.png")  
        plt.show()

    def SaveTrueAndPredicted(true_label,predicted_label,col_num,labels):
        '''
        '''
        x = true_label[:,col_num]
        y = predicted_label[:,col_num]
        both_true_and_predicted = np.vstack((x,y))
        np.save(f"Element_Pictures/{labels[col_num]}.npy",both_true_and_predicted)

    for i in range(len(labels)):
        MakeTrueVsPredictedPlots(true_test_labels,Cannon_test_labels,i,labels)
        SaveTrueAndPredicted(true_test_labels,Cannon_test_labels,i,labels)