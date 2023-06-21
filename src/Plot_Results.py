import joblib
import numpy as np
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r","--results_path",dest="results_path",help="Path to the results folder")

args = parser.parse_args()
cannon_results_path = args.results_path

# Load Scaler
scaler = joblib.load( os.path.join(cannon_results_path, "standard_scaler.joblib" ))
# Load y_param 
y_param = joblib.load( os.path.join(cannon_results_path, "y_param.joblib" ))
y_param = scaler.inverse_transform(y_param)
# Load inferred labels 
inferred_labels = joblib.load( os.path.join(cannon_results_path, "inferred_labels.joblib" ))
inferred_labels = scaler.inverse_transform(inferred_labels) 
# Load parameter names
param_names = joblib.load( os.path.join(cannon_results_path, "parameters_names.joblib" ))


# Remove [ 78,  94, 101, 114, 129] from the inferred labels and y_param
inferred_labels = np.delete(inferred_labels,[ 78,  94, 101, 114, 129],axis=0)
y_param = np.delete(y_param,[ 78,  94, 101, 114, 129],axis=0)


rice_std = {"Y/H":0.08,"Ni/H":0.04,"Fe/H":0.03,"Mn/H":0.05,"Cr/H":0.04,"V/H":0.06,"Ti/H":0.04,
"Ca/H":0.03,"Si/H":0.03,"Al/H":0.04,"Mg/H":0.04,"Na/H":0.05,"O/H":0.07,"N/H":0.08,
"C/H":0.05,"Teff":56,r"$log g$":0.09,r"V sin i":0.87}

brewer_std = {"Y/H":0.015,"Ni/H":0.006,"Fe/H":0.005,"Mn/H":0.01,"Cr/H":0.007,"V/H":0.017,"Ti/H":0.006,
"Ca/H":0.007,"Si/H":0.004,"Al/H":0.014,"Mg/H":0.006,"Na/H":0.007,"O/H":0.018,"N/H":0.021,
"C/H":0.013,"Teff":12.5,r"$log g$":0.014,r"V sin i":0.35}



metrics = {"Parameter":[],"mean(Cannon value - SPOCS value)":[]
                         ,"std(Cannon value - SPOCS value)":[]
                         ,"Rice std":[]
                         ,"Brewer std":[]
                         ,"Relative Difference":[]} 




def Relative_difference(true,preditcted):
    return (true - preditcted) / true

# Enumerate over the order of labels
for param_i, param  in enumerate(param_names):
    nameWithoutBrackets = param

    #Not params, thus I dont want to put a /H behind it
    if param not in ["LOGG","TEFF","VSINI"]:
        param = '[' + param.replace("H","") + "/H]"
    else:
        if param == "LOGG":
            param = r"$log g$"
        elif param == "TEFF":
            param = r"Teff"
        elif param == "VSINI":
            param = r"V sin i"

    true_data,predicted_data = y_param[:,param_i],inferred_labels[:,param_i]
    sigma = np.std(predicted_data)
    
    #Save values for table
    param = param.replace('[','').replace(']','')
    metrics["Parameter"].append(param)
    p_mu = np.mean(predicted_data)
    t_mu = np.mean(true_data)

    difference = predicted_data - true_data

    metrics["Relative Difference"].append(Relative_difference(t_mu,p_mu))
    metrics["mean(Cannon value - SPOCS value)"].append(np.mean(difference))
    metrics["std(Cannon value - SPOCS value)"].append(np.std(difference))
    metrics["Rice std"].append(rice_std[param])
    metrics["Brewer std"].append(brewer_std[param])


    
metrics = pd.DataFrame.from_dict(metrics)
metrics = np.round(metrics,3)

caption = "The mean values for each stellar parameters along with the relative difference between the SPOC value and the predicted value."

latex_str = metrics.to_latex(index=False,caption=caption)
latex_str = latex_str.replace("Teff", "$T_{\rm eff}$ ")
latex_str = latex_str.replace("V sin i", "$v \sin i$ ")
latex_str = latex_str.replace("\$log g\$", "$\log g$")

print("\n"*5)
print(latex_str)
#### </ TABLE SECTION /> ####


#### < PLOT SECTION > ####

### Only for paper, this will not work for all other datasets. 
fontsize = 50
i = 0
j = 0
fig, axs = plt.subplots(3, 6,figsize = (16,8)) # (# of rows, # of cols)
gs1 = gridspec.GridSpec(16,8)
gs1.update(wspace=0.025, hspace=0.05)

o = 0
for param_i, param  in enumerate(param_names):
    nameWithoutBrackets = param

    #Not params, thus I dont want to put a /H behind it
    if param not in ["LOGG","TEFF","VSINI"]:
        param = '[' + param.replace("H","") + "/H]"
    else:
        if param == "LOGG":
            param = r"log$g$ $[cm/s^2]$"
        elif param == "TEFF":
            param = r"$T_{eff}$ $[K]$"
        elif param == "VSINI":
            param = r"$v\sin i$ $[km/s]$"

    true_data,predicted_data = y_param[:,param_i],inferred_labels[:,param_i]
    sigma = np.std(predicted_data)
    #axs[i,j].set_title(f"{param}, " + r"$\sigma = $" + f"{sigma:.2f}",size=10) # With Sigma 
    axs[i,j].set_title(f"{param}",size=10) #Without sigma 

    axs[i,j].scatter(true_data,predicted_data,alpha=0.2,c="blue",s=40)
    res = stats.linregress(true_data,predicted_data)
    
    minn = min(np.min(true_data),np.min(predicted_data)) - 0.01
    maxx = max(np.max(true_data),np.max(predicted_data)) + 0.01
    if param == r"$T_{eff}$ $[K]$":
        minn -= 100
        maxx += 100
    elif param == r"$V\sin i$ $[km/s]$":
        minn -= 0.02
        maxx += 0.02
    
    
    axs[i,j].plot([minn,maxx], np.array([minn,maxx]), c= 'hotpink', label='One-to-One')
    axs[i,j].set_xlim([minn,maxx])
    axs[i,j].set_ylim([minn,maxx])
    
    axs[i,j].tick_params(bottom=True, top=True, left=True, right=True)
    axs[i,j].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    axs[i,j].tick_params(axis='both',
                        direction="in",
                        which='major',
                        length=3,
                        width=1)
    axs[i,j].tick_params(axis='both',
                    direction="in",
                    which='minor',
                    top=True,
                    right=True)
    axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[i,j].yaxis.set_major_locator(plt.MaxNLocator(6))


    axs[i,j].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i,j].yaxis.set_minor_locator(AutoMinorLocator())
    # axs[i,j].tick_params(axis="x", direction="in")
    # axs[i,j].tick_params(axis="y", direction="in")
    axs[i,j].xaxis.set_tick_params(labelsize=8)
    axs[i,j].yaxis.set_tick_params(labelsize=8)

    
    #axs[i,j].tick_params(axis='both',length=10,labelsize=15)
    
    # #Line of Best Fit
    # m,b = np.polyfit(true_data,predicted_data,1) 
    # axs[i,j].plot([minn,maxx],m*np.array([minn,maxx]) + b,c="grey", label="Linear Regression")

    if i == 2:
        i=0
        j+=1
    else:
        i+=1
    
    if j==6:
        
        plt.subplots_adjust(left  = 0.05,right = 0.981,bottom = 0.06,top = 0.957,wspace = 0.205,hspace = 0.205)
        
        fig.text(0.503, 0.015, 'SPOCS Label', ha='center', va='center',size=20)
        fig.text(0.011, 0.5, "The Cannon Label", ha='center', va='center', rotation='vertical',size=20)
        #plt.tight_layout()
        plt.savefig(f"{cannon_results_path}/All_in_one.png",dpi=300)
        plt.show()
