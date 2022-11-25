import numpy as np
import pandas as pd 
from CHIP import CHIP 
from TheCannonCHIP import TheCannonCHIP


config_values_array = np.loadtxt("config.txt", dtype=str, delimiter='=', usecols=(1))

#CHIP 
chip_q = config_values_array[0].replace(" ","").lower()[0] == "t"
if chip_q:
    crossMatchedNames = pd.read_csv(config_values_array[1].replace(" ",""),sep=" ")
    hiresNames = crossMatchedNames["HIRES"].to_numpy()
    chipObject = CHIP(hiresNames)
    alpha_q = (config_values_array[2].replace(" ","").lower()[0] == "t")
    past_q = (config_values_array[3].replace(" ","").lower()[0] == "t")
    chipObject.Run(use_past_data=past_q,alpha_normalize = alpha_q)
    print('''               ****************
            Check the Spectra in fluxes_for_HIRES.npy to ensure they are correct!
            Their wavelength is in interpolated_wl.csv. 
                            ****************
                ''')




#The Cannon 
cannon_q = config_values_array[4].replace(" ","").lower()[0] == "t"


if cannon_q:

    cannon_parameters_list = []
    for i in range(5,len(config_values_array)):
        temp_param = config_values_array[i]
        
        
        # Fixing parameters so TheCannonCHIP.py can use them 
        if i in [5,6,7,8,10]:
            temp_param = temp_param.replace(" ","")
            
        elif i in [9,]:
            #Deletes "[" & "]" & any spaces coming before "["
            i_open_bracket = temp_param.find("[") 
            i_close_bracket = temp_param.find("]")
            temp_param = temp_param[i_open_bracket+1:i_close_bracket-len(temp_param)]
            
            lst_elements = temp_param.split(",")
            new_param = []
            for j in range(0,len(lst_elements),2):
                #Creating Tubles
                firstElem= lst_elements[j].replace("(","").replace("'",'').replace('"','') #Should be the name 
                secondElem = lst_elements[j+1].replace(" ","").replace(")","").replace("'",'').replace('"','') #Should be false or a file path
                
                if secondElem.lower() == "false":
                    secondElem = False
                new_param.append((firstElem,secondElem))
                    
            temp_param = new_param
        
        elif i in [11,13]:
            temp_param = temp_param.replace(" ","").replace("[","").replace("]","").split(",")
            for j in range(len(temp_param)):
                new_elem = temp_param[j].replace("'","").replace('"',"")
                if i == 13:
                    new_elem = int(new_elem)
                temp_param[j] = new_elem
        
        elif i in [12,14,15]:
            temp_param = int(temp_param)
        elif i in [16,]:
            temp_param = temp_param.replace(" ","")
        elif i in [17,]:
            temp_param = eval(temp_param)
        
        cannon_parameters_list.append(temp_param)
            
    #Need to check lists are correct
    TheCannonCHIP(cannon_parameters_list[0],
                    cannon_parameters_list[1],
                    cannon_parameters_list[2],
                    cannon_parameters_list[3],
                    cannon_parameters_list[4],
                    cannon_parameters_list[5],
                    cannon_parameters_list[6],
                    cannon_parameters_list[7],
                    cannon_parameters_list[8],
                    cannon_parameters_list[9],
                    cannon_parameters_list[10],
                    cannon_parameters_list[11],
                    cannon_parameters_list[12])






#============= TESTING =============
if False: #Set to True if you want to do testing on reading config.txt 

    #Testing 
    tests_list = ["interpolated_wl.csv", 
                    "fluxes_for_HIRES.npy",
                    "ivars_for_HIRES.npy",
                    "stellar_names_for_flux_and_ivar.npy",
                    [("No Mask",False), ("iodine","../Constants/Masks/by_eye_iodine_mask.npy")],
                    "../spocData/df_all.csv",
                    ['TEFF', 'LOGG','VSINI', 'FeH', 'CH', 'NH','OH','NaH', 'MgH', 'AlH', 'SiH', 'CaH', 'TiH', 'VH', 'CrH', 'MnH','NiH', 'YH'],
                    3,
                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                    15,
                    16,
                    "(mu-mu*)/mu",
                    lambda true_array,predicted_array:  (np.mean(true_array) - np.mean(predicted_array)) / np.mean(true_array)
                    ]
                    
        
    #Testing 
    for i in range(5,len(config_values_array)):
        temp_param = config_values_array[i]
        test_param = tests_list[i-5]
        correct = True 
        
        # Fixing parameters so TheCannonCHIP.py can use them 
        if i in [5,6,7,8,10]:
            temp_param = temp_param.replace(" ","")
            
        elif i in [9,]:
            #Deletes "[" & "]" & any spaces coming before "["
            i_open_bracket = temp_param.find("[") 
            i_close_bracket = temp_param.find("]")
            temp_param = temp_param[i_open_bracket+1:i_close_bracket-len(temp_param)]
            
            lst_elements = temp_param.split(",")
            new_param = []
            for j in range(0,len(lst_elements),2):
                #Creating Tubles
                firstElem= lst_elements[j].replace("(","").replace("'",'').replace('"','') #Should be the name 
                secondElem = lst_elements[j+1].replace(" ","").replace(")","").replace("'",'').replace('"','') #Should be false or a file path
                
                if secondElem.lower() == "false":
                    secondElem = False
                new_param.append((firstElem,secondElem))
                    
            temp_param = new_param
        
        elif i in [11,13]:
            temp_param = temp_param.replace(" ","").replace("[","").replace("]","").split(",")
            for j in range(len(temp_param)):
                new_elem = temp_param[j].replace("'","").replace('"',"")
                if i == 13:
                    new_elem = int(new_elem)
                temp_param[j] = new_elem
        
        elif i in [12,14,15]:
            temp_param = int(temp_param)
        elif i in [16,]:
            temp_param = temp_param.replace(" ","")
        elif i in [17,]:
            temp_param = eval(temp_param)
            
            
        if type(test_param) == list:
            if len(test_param) != len(temp_param):
                print("Lengths don't match!")
                print(f"{len(temp_param)} should be {len(test_param)}")
                
                correct = False 
            else: 
                for j in range(len(test_param)):
                    if temp_param[j] != test_param[j]:
                        correct = False 
                        print(f"The {j+1} element is incorrect")
                        print(f"Read in: {temp_param[j]}")
                        print(f"Should be: {test_param[j]}")
                        
        elif (type(test_param) in [str,int]):
            if (temp_param != test_param):
                correct = False  
        elif callable(test_param):     
            for test_tuple in [(np.random.random(10,),np.random.random(10,))]:
                if test_param(test_tuple[0],test_tuple[1]) != temp_param(test_tuple[0],test_tuple[1]):
                    correct = False 
        
        
        
        
        if not correct: 
            print("--"*10)
            print(f"Index {i} does not match the test value!")
            print(f"Read in: {temp_param}")
            print(f"Should be: {test_param}")
            print("--"*10)
            break

    #END of testing 

