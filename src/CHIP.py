import logging
# logging 
logging.basicConfig(filename= 'CHIP.log',
                    format='%(asctime)s - %(message)s', 
                    datefmt="%Y/%m/%d %H:%M:%S",  
                    level=logging.INFO)
import json
import os
import numpy as np
import pandas as pd
import time 

from alpha_shapes import contfit_alpha_hull
from astropy.io import fits
from hiresprv.auth import login
from hiresprv.database import Database
from hiresprv.download import Download
from hiresprv.idldriver import Idldriver
from joblib import Parallel, delayed 







class CHIP:

    def __init__(self):
        '''
        
        '''
        # Create a new storage location for this pipeline
        self.create_storage_location()

        # login into the NExSci servers
        login('data/prv.cookies')

        # Get arguments from config.json
        self.get_arguments()

        # HIRES Filename : spectrum as a (16,4021) np.array  
        self.spectraDic = {}            

        # For storing sigma valeus
        self.ivarDic = {}                

                                                                                     
    
    def create_storage_location(self):
        ''' Creates a unique subdirectory in the data directory to store the outputs of CHIP
        
        Inputs: None

        Outputs: None 
        '''
        logging.debug("CHIP.create_storage_location( )")

        # Greenwich Mean Time (UTC) right now 
        gmt_datetime = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

        self.storage_path = os.path.join("data","chip_runs" , gmt_datetime )

        os.makedirs(self.storage_path,exist_ok=True)
        
        
    def run(self):
        ''' Run the pipeline from end to end. 

        Inputs: None

        Outputs: None
        
        '''
        # TODO: implement end-to-end run of CHIP

        pass 


    def get_arguments(self):
        ''' Get arguments from src/config.json 

        Inputs: None

        Outputs: (dict) representing src/config.json
        '''

        logging.debug("CHIP.get_arguments( )")

        with open("src/config.json", "r") as f:
            self.config = json.load(f)
        
        logging.info( f"config.json : {self.config}" )
    

    @staticmethod
    def calculate_SNR(spectrum):
        ''' Calculates the SNR of spectrum using the 8th and 9th echelle orders

        Inputs: spectrum (np.array) - spectrum with at least 9 rows representing echelle orders 

        Outputs: (float) estimated SNR of inputed spectrum
        '''
        logging.debug(f"CHIP.calculate_SNR( {spectrum} )")

        spectrum = spectrum[7:10].flatten() # Using echelle orders in the middle 
        SNR = np.mean(np.sqrt(spectrum))
        return SNR 


    def delete_spectrum(self,filename):
        '''Remove spectrum from local storage

        Inputs: filename (str): HIRES file name of spectrum you want to delete

        Outputs: None 
        '''
        logging.debug(f"CHIP.delete_spectrum( {filename} )")
        file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")
        try:
            os.remove(file_path)
        except FileNotFoundError as e:
            # file was already deleted 
            pass 


    def download_spectrum(self,filename,SNR = True):
            '''Download Individual Spectrum and ivar 

            Input: filename (str): HIRES file name of spectrum you want to download
                   SNR (bool): if you want to calculate the SNR of the spectrum 

            Output: None
            '''
            logging.debug(f"CHIP.download_spectrum( filename={filename}, SNR={SNR} )")
            
            #Download spectra
            self.dataSpectra.spectrum(filename.replace("r",""))    
            file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")

            try: 
                temp_deblazedFlux = fits.getdata(file_path)
            except OSError:
                # There is a problem with the downloaded fits file 
                return -1
            
            if SNR: # Used to find best SNR 
                SNR = self.calculate_SNR(temp_deblazedFlux)

                # delete Spectrum variable so it can be delete if needed
                del temp_deblazedFlux 
                return SNR
            else:
                # Trim the left and right sides of each echelle order 
                trim =  self.config["CHIP"]["trim_spectrum"]["val"]
                return temp_deblazedFlux[:, trim: -trim]
        
        
    def download_spectra(self):
        ''' Downloads all the spectra for each star in the NExSci, calculates 
        the SNR and saves the spectrum with the highest SNR for each star.

        Inputs: None

        Outputs: None
        '''
        logging.info("CHIP.download_spectra( )")

        # Cross matched names 
        cross_matched_file_path = self.config["CHIP"]["cross_match_stars"]["val"]
        cross_matched_df = pd.read_csv( cross_matched_file_path, sep=" " )
        hires_names_array = cross_matched_df["HIRES"].to_numpy()

        # Record results 
        self.removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], 
                              "SNR < 100":[],"NAN value in spectra":[],
                              "No Clue":[],"Nan in IVAR":[],
                              "Normalization error":[]}
        hiresID_fileName_snr_dic = {"HIRESid": [],"FILENAME":[],"SNR":[]}  

        # Location to save spectra
        spectra_download_location = os.path.join( self.storage_path, "rv_obs" )
        os.makedirs( spectra_download_location, exist_ok=True )

        # For retrieving data from HIRES
        self.state = Database('data/prv.cookies')
        self.dataSpectra = Download('data/prv.cookies', spectra_download_location)
                                  
        for star_ID in hires_names_array:
            logging.debug(f"Finding highest SNR spectrum for {star_ID}")
                
            
            # SQL query to find all the RV observations of one particular star
            search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"
            url = self.state.search(sql=search_string)
            obs_df = pd.read_html(url, header=0)[0]

            # Check if there are any RV Observations 
            if obs_df.empty: 
                logging.debug(f"{star_ID} has no RV observations")
                self.removed_stars["no RV observations"].append(star_ID) 
                continue 
            else:   
                logging.debug(f"RV Observation Filenames: {obs_df['FILENAME'] }")

                best_SNR = 0
                best_SNR_filename = False 

                for filename in obs_df["FILENAME"]:
                    temp_SNR = self.download_spectrum(filename)
                    

                    # Check if the SNR is the highest out of 
                    # all the star's previous spectras 
                    if best_SNR < temp_SNR:
                        # Since this spectrum is not longer the best, we will delete it
                        delete_spectrum_filename = best_SNR_filename 

                        best_SNR = temp_SNR
                        best_SNR_filename = filename 
                    else:
                        delete_spectrum_filename = filename 

                    # Delete unused spectrum
                    # Will not trigger in the intial case
                    if delete_spectrum_filename: 
                        self.delete_spectrum(delete_spectrum_filename)

                if best_SNR < 100: 
                    logging.debug(f"{star_ID}'s best spectrum had an SNR lower than 100. Thus it was removed.")

                    self.removed_stars["SNR < 100"].append(star_ID)

                else:
                    # Save the Best Spectrum
                    self.spectraDic[best_SNR_filename] = self.download_spectrum(best_SNR_filename, 
                                                                                SNR=False)
                    
                    logging.debug(f"{star_ID}'s best SNR spectrum came from {best_SNR_filename} with an SNR={best_SNR}")
                    hiresID_fileName_snr_dic["HIRESid"].append(star_ID)
                    hiresID_fileName_snr_dic["FILENAME"].append(best_SNR_filename)
                    hiresID_fileName_snr_dic["SNR"].append(best_SNR)
                
                    # Calculate ivar
                    self.sigma_calculation(best_SNR_filename , star_ID)


        # Save SNR meta data in csv file 
        self.hires_filename_snr_df = pd.DataFrame(hiresID_fileName_snr_dic) 
        self.hires_filename_snr_df.to_csv( os.path.join(self.storage_path ,"HIRES_Filename_snr.csv"),
                                           index_label=False,
                                           index=False)
        
        pd.DataFrame(self.removed_stars).to_csv( os.path.join(self.storage_path ,"removed_stars.csv"),
                                           index_label=False,
                                           index=False) 

        # Delete unused instance attributes
        del self.dataSpectra
        del self.state
        

    def sigma_calculation(self,filename , star_ID):
        '''Calculates sigma for inverse variance (IVAR) 

        Inputs: filename (str): HIRES file name of spectrum you want to calculate IVAR for
                star_ID (str): HIRES identifer 

        Outputs: None
        '''
        logging.debug("CHIP.sigma_calculation( filename = {filename} )")
        gain = 1.2 #electrons/ADU
        readn = 2.0 #electrons RMS
        xwid = 5.0 #pixels, extraction width

        sigma = np.sqrt((gain*self.spectraDic[filename]) + (xwid*readn**2))/gain 
        #Checkinng for a division by zeros 
        if not np.isnan(sigma).any(): #Happens in the IVAR
                self.ivarDic[filename] = sigma
        else:
            logging.info(f"{star_ID} has NAN value in IVAR") 
            self.removed_stars["Nan in IVAR"].append(star_ID)
            del self.spectraDic[filename]


    def alphanormalization(self):
        ''' Rolling Continuum Normalization.  

        Input: None

        Output: None
        '''
        logging.info("CHIP.alphanormalization( )")
        
        # Trim wl_solution 
        self.wl_solution = np.load("data/spocs/wl_solution.npy")
        trim =  self.config["CHIP"]["trim_spectrum"]["val"]
        self.wl_solution = self.wl_solution[:, trim: -trim]
            
        # Create Normalized Spectra dir
        norm_spectra_dir_path = os.path.join( self.storage_path, "norm_spectra" )
        os.mkdir( norm_spectra_dir_path )

        start_time = time.time()
        Parallel( n_jobs = 1 )(delayed( contfit_alpha_hull )(star_name,
                                    self.spectraDic[star_name],
                                    self.ivarDic[star_name],
                                    self.wl_solution,
                                    norm_spectra_dir_path) for star_name in self.spectraDic)

        # for star_name in self.spectraDic:                
        #         contfit_alpha_hull(star_name,
        #                             self.spectraDic[star_name],
        #                             self.ivarDic[star_name],
        #                             self.wl_solution,
        #                             norm_spectra_dir_path) 

        end_time = time.time()
        logging.info(f"It took alphanorm, {end_time - start_time} to finish!")
        print(f"It took alphanorm, {end_time - start_time} to finish!")
        for star_name in list(self.spectraDic):
            try:
                specnorm_path = os.path.join( norm_spectra_dir_path , f"{star_name}_specnorm.npy")
                sigmanorm_path = os.path.join( norm_spectra_dir_path , f"{star_name}_sigmanorm.npy")
                self.spectraDic[star_name] = np.load(specnorm_path).flatten()
                self.ivarDic[star_name] = np.load(sigmanorm_path).flatten()
            except:
                logging.info(f'''Something went wrong with {star_name}'s normalization. We have removed the star.''')
                del self.spectraDic[star_name]
                del self.ivarDic[star_name]
                self.removed_stars["Normalization error"].append(star_name)
                        

        




if __name__ == "__main__":
    
    # Instantiation
    chip = CHIP()

    # logs to file and stdout
    logging.getLogger().addHandler(logging.StreamHandler())
    # set datefmt to GMT
    logging.Formatter.converter = time.gmtime

    chip.download_spectra()

    chip.alphanormalization()

    







