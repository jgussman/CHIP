import numpy as np
import matplotlib.pyplot as plt
# from numpy.core.defchararray import capitalize, index
# from numpy.core.fromnumeric import size 
from scipy import stats
import glob
# import matplotlib.ticker as ticker
import pandas as pd 

###
###Anything BELOW this point (to the stop point) can be editted to work with your needs
###
folder_location = "Best_Model_Results" #The folder where results of the best model ended up
###
###Anything ABOVE this point (to the start point) can be editted to work with your needs
###

mean_values = {"Parameter":[],"mu Predicted":[],"mu SPOC":[],"Relative Difference":[]}


def Relative_difference(true,preditcted):
    return (true - preditcted) / true

i = 0
j = 0
o = 0
for path in glob.glob(folder_location+"/*.npy"):
    element = path.split("/")[-1].split("\\")[-1].replace(".npy","")
    nameWithoutBrackets = element

    #Not elements, thus I dont want to put a /H behind it
    if element not in ["LOGG","TEFF","VSINI"]:
        element = '[' + element.replace("H","") + "/H]"
    else:
        if element == "LOGG":
            element = r"$log g$ $[cm/s^2]$"
        elif element == "TEFF":
            element = r"$T_{eff}$ $[K]$"
        elif element == "VSINI":
            element = r"$v\sin i$ $[km/s]$"
    fig = plt.figure(figsize=(9, 9))

    ax1 = fig.add_subplot()
    
    ax1.set_xlabel("SPOC Value",size=20)
    ax1.set_ylabel("Predicted Value",size=20)
    true_data,predicted_data = np.load(path)
    sigma = np.std(np.load(path)) #Re-dun but idc
    ax1.set_title(f"{element}, " + r"$\sigma = $" + f"{sigma:.2f}",size=30)

    ax1.scatter(true_data,predicted_data,alpha=0.4,c="dodgerblue",s=80)
    res = stats.linregress(true_data,predicted_data)
    
    minn = min(np.min(true_data),np.min(predicted_data)) - 0.2
    maxx = max(np.max(true_data),np.max(predicted_data)) + 0.2
    if element == r"$T_{eff}$ $[K]$":
        minn -= 100
        maxx += 100
    elif element == r"$v\sin i$ $[km/s]$":
        minn -= 1
        maxx += 1

    ax1.plot([minn,maxx], np.array([minn,maxx]), c= 'crimson', label='Line of Equality')
    ax1.set_xlim([minn,maxx])
    ax1.set_ylim([minn,maxx])
    ax1.tick_params(axis='both',length=10,labelsize=15)
    
    #Line of Best Fit
    m,b = np.polyfit(true_data,predicted_data,1) 
    ax1.plot([minn,maxx],m*np.array([minn,maxx]) + b,c="grey", label="Linear Best Fit")

    plt.legend()
    plt.savefig(folder_location + f"/{nameWithoutBrackets}",dpi=300)
    print(nameWithoutBrackets) 
   
    #Save values for table
    mean_values["Parameter"].append(element.replace('[','').replace(']',''))
    p_mu = np.mean(predicted_data)
    t_mu = np.mean(true_data)
    mean_values["mu Predicted"].append(p_mu)
    mean_values["mu SPOC"].append(t_mu)
    mean_values["Relative Difference"].append(Relative_difference(t_mu,p_mu))
    

mean_values = pd.DataFrame.from_dict(mean_values)

caption = "The mean values for each stellar parameters along with the relative difference between the SPOC value and the predicted value."
print(mean_values.to_latex(index=False,caption=caption))






### Only for paper, this will not work for all other datasets. 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import glob

fontsize = 12
i = 0
j = 0
fig, axs = plt.subplots(3, 6,figsize = (16,8)) # (# of rows, # of cols)

o = 0
for path in glob.glob(folder_location+"/*.npy"):
    element = path.split("/")[-1].split("\\")[-1].replace(".npy","")
    nameWithoutBrackets = element

    #Not elements, thus I dont want to put a /H behind it
    if element not in ["LOGG","TEFF","VSINI"]:
        element = '[' + element.replace("H","") + "/H]"
    else:
        if element == "LOGG":
            element = r"$log g$ $[cm/s^2]$"
        elif element == "TEFF":
            element = r"$T_{eff}$ $[K]$"
        elif element == "VSINI":
            element = r"$v\sin i$ $[km/s]$"

    true_data,predicted_data = np.load(path)
    sigma = np.std(np.load(path)) #Re-dun but idc
    axs[i,j].set_title(f"{element}, " + r"$\sigma = $" + f"{sigma:.2f}",size=10)

    axs[i,j].scatter(true_data,predicted_data,alpha=0.2,c="dodgerblue",s=40)
    res = stats.linregress(true_data,predicted_data)
    
    minn = min(np.min(true_data),np.min(predicted_data)) - 0.2
    maxx = max(np.max(true_data),np.max(predicted_data)) + 0.2
    if element == r"$T_{eff}$ $[K]$":
        minn -= 100
        maxx += 100
    elif element == r"$v\sin i$ $[km/s]$":
        minn -= 1
        maxx += 1

    axs[i,j].plot([minn,maxx], np.array([minn,maxx]), c= 'crimson', label='One-to-One')
    axs[i,j].set_xlim([minn,maxx])
    axs[i,j].set_ylim([minn,maxx])
    axs[i,j].tick_params(axis='both',length=10,labelsize=15)
    
    #Line of Best Fit
    m,b = np.polyfit(true_data,predicted_data,1) 
    axs[i,j].plot([minn,maxx],m*np.array([minn,maxx]) + b,c="grey", label="Linear Regression")

    if i == 2:
        i=0
        j+=1
    else:
        i+=1
    
    if j==6:
        fig.text(0.503, 0.01, 'SPOC Value', ha='center', va='center')
        fig.text(0.01, 0.5, "The Cannon's Predicted Value", ha='center', va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig("Element_Data/All_in_one.png",dpi=300)
        plt.show()
    