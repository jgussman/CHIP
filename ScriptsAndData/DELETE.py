from alpha_shapes import * 
from astropy import units as u
from astropy.io import fits
from hiresprv.auth import login
from hiresprv.idldriver import Idldriver
from hiresprv.database import Database
from hiresprv.download import Download
from PyAstronomy import pyasl
from specutils.fitting import fit_generic_continuum
from specutils.spectra import Spectrum1D
from scipy import interpolate 

import numpy as np
import pandas as pd 
import os

import concurrent.futures 
#from multiprocessing import Pool

# Need to download 43030 spectra
# max 321 KB size 
# ~14 GB of storage space needed 



if __name__ == "__main__":
    config_values_array = np.loadtxt("config.txt", dtype=str, delimiter='=', usecols=(1))
    crossMatchedNames = pd.read_csv(config_values_array[1].replace(" ",""),sep=" ")
    star_ID_array = crossMatchedNames["HIRES"].to_numpy()

    # For logging into the NExSci servers 
    login('prv.cookies')            
    # For retrieving data from HIRES                    
    state = Database('prv.cookies') 

    def get_HIRES_filenames(star_ID):
           
    
        search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"

        url = state.search(sql=search_string)
        obs_df = pd.read_html(url, header=0)[0]
        return obs_df.shape[0]
    

    c = 0
    try:
        for star_ID in star_ID_array:
            num = get_HIRES_filenames(star_ID)
            c += num
            print(c, end='\r')
    except:
        pass 
    finally:
        print(c)
        



    

    import sys
    sys.exit(0)

        # if obs_df.empty or not ("FILENAME" in obs_df.columns): # There are no RV Observations
        #     removed_stars["no RV observations"].append(star_ID)
        # else:
        #     for filename in obs_df["FILENAME"]:
        #         spectra_to_download_list.append((star_ID,filename))                            
        




    wl_solution_path = '../SPOCSdata/wl_solution.csv'
    rvOutputPath='./RVOutput'
    spectraOutputPath = './SpectraOutput'

    os.makedirs(rvOutputPath, exist_ok=True)
    os.makedirs(spectraOutputPath, exist_ok=True)

    
    dataSpectra = Download('prv.cookies',spectraOutputPath)       # For downloading Spectra 
    star_ID_array = star_ID_array                                 # HIRES ID 
    wl_solution = np.genfromtxt(wl_solution_path,skip_header=2)   # UNITS: Angstrom
    Ivar = {}                                                     # For storing sigma valeus 
    filename_rv_df = pd.DataFrame()                               # For storing meta Data
    spectraDic = {}                                               # For storing spectra filenames


    def calculate_SNR(spectrum):
        spectrum = spectrum[7:10].flatten() # Using echelle orders in the middle 
        SNR = np.mean(np.sqrt(spectrum))
        return SNR 

    def DeleteSpectrum(filename):

        #print(f"Deleting {filename}.fits")
        file_path = os.path.join(dataSpectra.localdir,filename + ".fits")
        try:
            os.remove(file_path)
        except FileNotFoundError as e:
            # file was already deleted 
            pass 

    def find_rv_obs_with_highest_snr():
        removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], "SNR < 100":[],"NAN value in spectra":[]}
        hiresName_fileName_snr_dic = {"HIRESName": [],"FILENAME":[],"SNR":[]}  

        spectra_to_download_list = []

        


    #find_rv_obs_with_highest_snr()



    # def DownloadSpectrum(filename,SNR = True):
    #     '''Download Individual Spectrum and ivar 

    #     Input: filename (str): name of spectrum you want to download
    #         SNR (bool): if you want to calculate the SNR of the spectrum 

    #     Output: None
    #     '''
    #     #print(f"Downloading {filename}")
    #     dataSpectra.spectrum(filename.replace("r",""))  #Download spectra  
    #     file_path = os.path.join(dataSpectra.localdir,filename + ".fits")
    #     try: 
    #         temp_deblazedFlux = fits.getdata(file_path)
    #     except (OSError,AttributeError): 
    #         return -1
        
    #     if SNR: # Used to find best SNR 
    #         # Delete Spectrum to save space
    #         SNR = calculate_SNR(temp_deblazedFlux)
    #         del temp_deblazedFlux 
    #         DeleteSpectrum(filename)
    #         return SNR

    #     else:
    #         return temp_deblazedFlux



    # def find_best_snr(star_ID):
    #     search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"

    #     url = state.search(sql=search_string)
    #     obs_df = pd.read_html(url, header=0)[0]

        
    #     if obs_df.empty or not ("FILENAME" in obs_df.columns): # There are no RV Observations
    #         return star_ID, "no RV observations", np.NaN
        
    #     else:   

    #         best_SNR = 0
    #         best_SNR_filename = False 
    #         for filename in obs_df["FILENAME"]:
    #             temp_SNR = DownloadSpectrum(filename)
                
    #             # Check if the SNR is the highest out of 
    #             # all the star's spectras 
    #             if best_SNR < temp_SNR:
    #                 best_SNR = temp_SNR

    #                 best_SNR_filename = filename 

    #         # Save the Best Spectrum
    #         spectraDic[best_SNR_filename] = DownloadSpectrum(best_SNR_filename, 
    #                                                         SNR=False)
            
    #         # Calculate ivar
    #         # TODO: Uncomment the link beblow
    #         #SigmaCalculation(best_SNR_filename)
            
    #         return star_ID, best_SNR_filename, best_SNR

            

    # def find_rv_obs_with_highest_snr():

        

    #     removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], "SNR < 100":[],"NAN value in spectra":[]}
    #     hiresName_fileName_snr_dic = {"HIRESName": [],"FILENAME":[],"SNR":[]}  

        
    #     with concurrent.futures.ThreadPoolExecutor( max_workers=1 ) as executor:
    #     #with Pool( ) as pool:
    #         #results = pool.imap_unordered(find_best_snr, star_ID_array)
    #         results = [val for val in executor.map(find_best_snr, star_ID_array)]
        
    #         for data in results:
    #             star_ID = data[0]
    #             star_FILENAME = data[1]
    #             star_SNR = data[2]
    #             if np.isnan( star_SNR ):
    #                 removed_stars[star_FILENAME].append(star_ID)
    #             else:
    #                 hiresName_fileName_snr_dic["HIRESName"].append(star_ID)
    #                 hiresName_fileName_snr_dic["FILENAME"].append(star_FILENAME)
    #                 hiresName_fileName_snr_dic["SNR"].append(star_SNR)


    #     filename_snr_df = pd.DataFrame(hiresName_fileName_snr_dic) 
    #     filename_snr_df.to_csv("HIRES_Filename_snr.csv",index_label=False,index=False)
    #     print(removed_stars)
    #     print("Downloading Spectra Has Finished")

    # find_rv_obs_with_highest_snr()