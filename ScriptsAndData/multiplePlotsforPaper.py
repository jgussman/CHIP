

# --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import glob

fontsize = 12
i = 0
j = 0
fig, axs = plt.subplots(3, 6,figsize = (16,8)) # (# of rows, # of cols)

o = 0
for path in glob.glob("Element_Data/*.npy"):
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

    sigma = np.std(np.load(path)) #Re-dun but idc
    
    true_data,predicted_data = np.load(path)
    
    axs[i, j].locator_params(axis="x", nbins=6)
    axs[i, j].locator_params(axis="y", nbins=6)
    axs[i, j].axis('equal')
    axs[i, j].scatter(true_data,predicted_data,alpha=0.5)
    res = stats.linregress(true_data,predicted_data)
    axs[i, j].plot(true_data, res.intercept + res.slope*true_data, 'r', label='Best Fit')
    axs[i, j].set_title(f"{element} with " + r"$\sigma = $" + f"{sigma:.2f}")

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
    


# --------------------------------------------

# With histogram 

# from struct import unpack
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.core.defchararray import capitalize, index
# from numpy.core.fromnumeric import size 
# from scipy import stats
# import glob
# import matplotlib.ticker as ticker
# import pandas as pd 

# mean_values = {"Parameter":[],"mu Predicted":[],"mu SPOC":[],"Relative Difference":[]}

# def Relative_difference(true,preditcted):
#     return (true - preditcted) / true

# i = 0
# j = 0
# o = 0
# for path in glob.glob("Element_Data/*.npy"):
#     element = path.split("/")[-1].split("\\")[-1].replace(".npy","")
#     nameWithoutBrackets = element

#     #Not elements, thus I dont want to put a /H behind it
#     if element not in ["LOGG","TEFF","VSINI"]:
#         element = '[' + element.replace("H","") + "/H]"
#     else:
#         if element == "LOGG":
#             element = r"$log g$ $[cm/s^2]$"
#         elif element == "TEFF":
#             element = r"$T_{eff}$ $[K]$"
#         elif element == "VSINI":
#             element = r"$v\sin i$ $[km/s]$"
#     fig = plt.figure(figsize=(9, 9))


#     ax1 = fig.add_subplot()
    
#     ax1.set_xlabel("SPOC Value",size=20)
#     ax1.set_ylabel("Predicted Value",size=20)
#     true_data,predicted_data = np.load(path)
#     sigma = np.std(np.load(path)) #Re-dun but idc
#     ax1.set_title(f"{element}, " + r"$\sigma = $" + f"{sigma:.2f}",size=30)

#     ax1.scatter(true_data,predicted_data,alpha=0.4,c="dodgerblue",s=80)
#     res = stats.linregress(true_data,predicted_data)
    
#     minn = min(np.min(true_data),np.min(predicted_data)) - 0.2
#     maxx = max(np.max(true_data),np.max(predicted_data)) + 0.2
#     if element == r"$T_{eff}$ $[K]$":
#         minn -= 100
#         maxx += 100
#     elif element == r"$v\sin i$ $[km/s]$":
#         minn -= 1
#         maxx += 1

#     ax1.plot([minn,maxx], np.array([minn,maxx]), c= 'crimson', label='Linear Fit')
#     ax1.set_xlim([minn,maxx])
#     ax1.set_ylim([minn,maxx])
#     ax1.tick_params(axis='both',length=10,labelsize=15)
    
#     #For Changing the Position of Plots 
#     positions_for_inner_plot = [(0.10,0.75)]*20
#     x,y = positions_for_inner_plot[i]
#     i +=1
    

#     inner_plot = ax1.inset_axes([x,y,0.25,0.25],alpha=.5)
#     diff = true_data-predicted_data
#     ranges = np.max(np.abs(diff))
#     if ranges < 0.5:
#         ranges = 0.5 
#     inner_plot.hist(true_data-predicted_data,bins=20,range=(-ranges,ranges),color="mediumslateblue")
#     inner_plot.tick_params(axis='both',length=4,labelsize=15)
#     inner_plot.set_xlabel("SPOC - Predicted",size=15)
#     inner_plot.set_ylabel("# of stars",size=15)
    

#     # plt.show()
#     # break 
#     plt.savefig(f"Element_Data/{nameWithoutBrackets}",dpi=300)
#     print(nameWithoutBrackets) 
   
#     #Save values for table
#     mean_values["Parameter"].append(element.replace('[','').replace(']',''))
#     p_mu = np.mean(predicted_data)
#     t_mu = np.mean(true_data)
#     mean_values["mu Predicted"].append(p_mu)
#     mean_values["mu SPOC"].append(t_mu)
#     mean_values["Relative Difference"].append(Relative_difference(t_mu,p_mu))
    

    
    
    

# mean_values = pd.DataFrame.from_dict(mean_values)

# caption = "The mean values for each stellar parameters along with the relative difference between the SPOC value and the predicted value."
# print(mean_values.to_latex(index=False,caption=caption))



    


