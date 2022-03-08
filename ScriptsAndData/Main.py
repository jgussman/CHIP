import numpy as np
import pandas as pd 
#from CHIP import CHIP 
from TheCannonCHIP import TheCannonCHIP


config_values_array = np.loadtxt("config.txt", dtype=str, delimiter='=', usecols=(1))

# #CHIP 
# chip_q = config_values_array[0].replace(" ","").lower() == "true"
# if chip_q:
#     config_values_array = np.loadtxt("config.txt", dtype=str, delimiter='=', usecols=(1))
#     crossMatchedNames = pd.read_csv(config_values_array[1],sep=" ")
#     hiresNames = crossMatchedNames["HIRES"].to_numpy()
#     chipObject = CHIP(hiresNames)
#     past_q = (config_values_array[3].replace(" ","").lower() == "True")
#     alpha_q = (config_values_array[2].replace(" ","").lower() == "True")
#     chipObject.Run(use_past_data=past_q,alpha_normalize = alpha_q)

#The Cannon 
cannon_q = config_values_array[4].replace(" ","").lower() == "true"
if cannon_q:
    #Need to check lists are correct
    TheCannonCHIP(config_values_array[5],
                  config_values_array[6],
                  config_values_array[7],
                  config_values_array[8],
                  list(config_values_array[9]),
                  config_values_array[10],
                  list(config_values_array[11]),
                  int(config_values_array[12]),
                  list(config_values_array[13]),
                  float(config_values_array[14]),
                  float(config_values_array[15]),
                  config_values_array[16].replace(" ",""),
                  enumerate(config_values_array[17]))

