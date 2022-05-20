import numpy as np
import matplotlib.pyplot as plt
# from numpy.core.defchararray import capitalize, index
# from numpy.core.fromnumeric import size 
from scipy import stats
from matplotlib.ticker import AutoMinorLocator
import glob
# import matplotlib.ticker as ticker
import pandas as pd 
import matplotlib.gridspec as gridspec

###
###Anything BELOW this point (to the stop point) can be editted to work with your needs
###
folder_location = "Best_Model_Results" #The folder where results of the best model ended up
###
###Anything ABOVE this point (to the start point) can be editted to work with your needs
###

mean_values = {"Parameter":[],"mu Predicted":[],"std Predicted":[],"mu SPOC":[],"std SPOC":[],"Relative Difference":[]}


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
            element = r"$log g$"
        elif element == "TEFF":
            element = r"Teff"
        elif element == "VSINI":
            element = r"V sin i"
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
    # Adjust the boundaries to make the plots look nice
    if element == r"Teff":
        minn -= 100
        maxx += 100
    elif element == r"V sin i":
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
    p_std = np.std(predicted_data)
    t_std = np.std(true_data)

    mean_values["mu Predicted"].append(p_mu)
    mean_values["std Predicted"].append(p_std)
    mean_values["mu SPOC"].append(t_mu)
    mean_values["std SPOC"].append(t_std)
    
    mean_values["Relative Difference"].append(Relative_difference(t_mu,p_mu))

    plt.close() #Closes plot 

mean_values = pd.DataFrame.from_dict(mean_values)
mean_values = np.round(mean_values,3)

caption = "The mean values for each stellar parameters along with the relative difference between the SPOC value and the predicted value."

latex_str = mean_values.to_latex(index=False,caption=caption)
latex_str = latex_str.replace("Teff", "$T_{eff}$ ")
latex_str = latex_str.replace("V sin i", "$V \sin i$ ")
latex_str = latex_str.replace("\$log g\$", "$\log g$")


print(latex_str)



### Only for paper, this will not work for all other datasets. 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import glob

fontsize = 50
i = 0
j = 0
fig, axs = plt.subplots(3, 6,figsize = (16,8)) # (# of rows, # of cols)
gs1 = gridspec.GridSpec(16,8)
gs1.update(wspace=0.025, hspace=0.05)

o = 0
for path in glob.glob(folder_location+"/*.npy"):
    element = path.split("/")[-1].split("\\")[-1].replace(".npy","")
    nameWithoutBrackets = element

    #Not elements, thus I dont want to put a /H behind it
    if element not in ["LOGG","TEFF","VSINI"]:
        element = '[' + element.replace("H","") + "/H]"
    else:
        if element == "LOGG":
            element = r"log$g$ $[cm/s^2]$"
        elif element == "TEFF":
            element = r"$T_{eff}$ $[K]$"
        elif element == "VSINI":
            element = r"$v\sin i$ $[km/s]$"

    true_data,predicted_data = np.load(path)
    sigma = np.std(np.load(path)) #Re-dun but idc
    #axs[i,j].set_title(f"{element}, " + r"$\sigma = $" + f"{sigma:.2f}",size=10) # With Sigma 
    axs[i,j].set_title(f"{element}",size=10) #Without sigma 

    axs[i,j].scatter(true_data,predicted_data,alpha=0.2,c="blue",s=40)
    res = stats.linregress(true_data,predicted_data)
    
    minn = min(np.min(true_data),np.min(predicted_data)) - 0.1
    maxx = max(np.max(true_data),np.max(predicted_data)) + 0.1
    if element == r"$T_{eff}$ $[K]$":
        minn -= 100
        maxx += 100
    elif element == r"$V\sin i$ $[km/s]$":
        minn -= 1
        maxx += 1
    
    
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
        
        plt.subplots_adjust(left  = 0.05,right = 0.981,bottom = 0.052,top = 0.957,wspace = 0.205,hspace = 0.205)
        
        fig.text(0.503, 0.01, 'SPOCS Label', ha='center', va='center',size=20)
        fig.text(0.009, 0.5, "The Cannon Label", ha='center', va='center', rotation='vertical',size=20)
        #plt.tight_layout()
        plt.savefig(f"{folder_location}/All_in_one.png",dpi=300)
        plt.show()
    
    