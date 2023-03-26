import logging
import joblib
import json
import os
import numpy as np
import pandas as pd
import shutil
import time 
import itertools 
import sys


from alpha_shapes import contfit_alpha_hull
from astropy.io import fits
from hiresprv.auth import login
from hiresprv.database import Database
from hiresprv.download import Download
from hiresprv.idldriver import Idldriver
from joblib import Parallel, delayed 
from PyAstronomy import pyasl
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from TheCannon import dataset, model





class CHIP:
    chip_version = "v1.0.1"
    thecannon_version = "v1.0.0"

    def __init__(self):
        '''
        
        '''
        # Get arguments from config.json
        self.get_arguments()

        # create storage location 
        self.create_storage_location()

        # if running CHIP HIRES, Filename : spectrum as a (16,N pixels) np.array  
        # if running The Cannon, HIRES ID : spectrum as a (16,N pixels) np.array 
        self.spectraDic = {}            
        # For storing sigma valeus
        self.ivarDic = {}                

                                                                                     
    def create_storage_location(self):
        ''' Creates a unique subdirectory in the data directory to store the outputs of CHIP
        
        Inputs: None

        Outputs: None 
        '''
        logging.debug("CHIP.create_storage_location( )")


        # If we are running preprocessing then we want a new 
        # CHIP run sub dir else we are running The Cannon
        # and want to use a previous run 
        if self.config["Preprocessing"]["run"]["val"]:
            # Create a new storage location for this pipeline
            # Greenwich Mean Time (UTC) right now 
            gmt_datetime = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

            self.storage_path = os.path.join("data","chip_runs" , gmt_datetime )

            # login into the NExSci servers
            login('data/prv.cookies')

        else:
            # What chip run is going to be used
            chip_run_subdir = self.config["Training"]["run"]["val"]
            self.data_dir_path = os.path.join("data/chip_runs",chip_run_subdir)

            if not(os.path.exists(self.data_dir_path)):
                logging.error(f'Your inputed sub dir name for ["Training"]["run"]["val"] does not exist. {self.data_dir_path}')
            
            # Make The Cannon Results dir 
            # semi-unique subdir naming convention
            # random seed _ testing fraction _ validation fraction _ cost function name (spaces filled with -)
            rand_seed = self.config["Training"]["random seed"]["val"]
            test_frac = self.config["Training"]["train test split"]["val"]
            cost_fun  = self.config["Training"]["cost function"]["name"].replace(" ", "-")
            k_fold    = self.config["Training"]["kfolds"]["val"]
            semi_unique_subdir = f"{rand_seed}_{test_frac}_{cost_fun}_{k_fold}"
            
            self.storage_path = os.path.join( self.data_dir_path, "training_results", semi_unique_subdir )
            
        os.makedirs( self.storage_path, exist_ok= True )

        
    def run(self):
        ''' Run the pipeline from end to end. 

        Inputs: None

        Outputs: None
        '''
        
        if self.config["Preprocessing"]["run"]["val"]:
            logging.info(f"CHIP {self.chip_version}")

            # Record results 
            self.removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], 
                                "SNR < 100":[],"NAN value in spectra":[],
                                "No Clue":[],"Nan in IVAR":[],
                                "Normalization error":[]}

            if isinstance(self.config["Preprocessing"]["run"]["val"],bool):

                self.download_spectra()

                self.alpha_normalization()

                self.cross_correlate_spectra()

                self.interpolate()

            else:
                past_run, data_folder = self.config["Preprocessing"]["run"]["val"][0], self.config["Preprocessing"]["run"]["val"][1]
                logging.info(f"Using the past run {past_run}'s {data_folder}")

                past_run_path = os.path.join(os.path.dirname(self.storage_path), past_run)

                if os.path.exists(past_run_path):
                    

                    data_folder_path = os.path.join(past_run_path,data_folder)
                    if os.path.exists(data_folder_path):
                        # Transfer all files from past run to new run
                        shutil.rmtree(self.storage_path)
                        shutil.copytree(past_run_path,
                                        self.storage_path)

                        self.hires_filename_snr_df = pd.read_csv( os.path.join(self.storage_path, "HIRES_Filename_snr.csv"))
                        if data_folder in ["rv_obs","norm"]:
                            self.load_past_rv_obs( os.path.join(self.storage_path,"rv_obs"))
                    
                            # Continue with normal operations 
                            self.alpha_normalization()
                            self.cross_correlate_spectra()
                            self.interpolate()

                        else: 
                            logging.error(f"{data_folder} is not currently a supported starting location for using past CHIP runs")

                    else:
                        logging.error(f"The data folder {data_folder} in {past_run} does not exist! We are looking at {data_folder_path}")
                else:
                    logging.error(f"The past run {past_run} does not exist! We are looking at {past_run_path}")
                    
    
        elif self.config["Training"]["run"]["val"]:
            logging.info(f"The Cannon {self.thecannon_version}")
    

            self.load_the_cannon()
            self.hyperparameter_tuning()

        else:
            logging.error("Neither CHIP or The Cannon were selected to run in config.json! Please select!")


    def load_past_rv_obs(self,data_folder_path):
        ''' Load in past data to continue preprocessing

        Input: data_folder_path (str): File path to rv_obs folder

        Output: None 
        '''
        for _, row in self.hires_filename_snr_df.iterrows():
            # Save the Best Spectrum
            star_id = row["HIRESid"]
            filename = row["FILENAME"]
            self.spectraDic[filename] = self.download_spectrum(filename, 
                                                                SNR=False,
                                                                past_rv_obs_path=data_folder_path)
            # Calculate ivar
            self.sigma_calculation(filename , star_id)


    def get_arguments(self):
        ''' Get arguments from src/config.json and store in self.config 

        Inputs: None

        Outputs: None
        '''
        logging.debug("CHIP.get_arguments( )")

        with open("src/config.json", "r") as f:
            self.config = json.load(f)
        
        self.cores = self.config["Preprocessing"]["cores"]["val"]
        self.trim  = self.config["Preprocessing"]["trim_spectrum"]["val"]
        logging.info( f"config.json : {self.config}" )
    

    @staticmethod
    def calculate_SNR(spectrum, gain = 2.09):
        ''' Calculates the SNR of spectrum using the 5th echelle orders

        Inputs: spectrum (np.array) - spectrum with at least 5 rows representing echelle orders 
                gain (float) - gain of the detector

        Outputs: (float) estimated SNR of inputed spectrum
        '''
        logging.debug(f"CHIP.calculate_SNR( {spectrum} )")

        spectrum = spectrum[4].flatten() # Using echelle orders in the middle 
        SNR = np.sqrt(np.median(spectrum * gain))
        
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


    def download_spectrum(self,filename,SNR = True, past_rv_obs_path = False):
        '''Download Individual Spectrum and calculate the ivar 

        Input: filename (str): HIRES file name of spectrum you want to download
            SNR (bool): if you want to calculate the SNR of the spectrum 
            past_rv_obs_path (str): if you want to load in old rb obs set to rb_obs dir path

        Output: None
        '''
        logging.debug(f"CHIP.download_spectrum( filename={filename}, SNR={SNR}, past_rv_obs_path={past_rv_obs_path} )")
        
        if not past_rv_obs_path:
            #Download spectra
            self.dataSpectra.spectrum(filename.replace("r",""))    
            file_path = os.path.join(self.dataSpectra.localdir, filename + ".fits")

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
                return temp_deblazedFlux[:, self.trim: -self.trim]
        else:
            # Load past data 
            file_path = os.path.join(past_rv_obs_path, filename + ".fits")
            if os.path.exists(file_path):
                temp_deblazedFlux = fits.getdata(file_path)
                return temp_deblazedFlux[:, self.trim: -self.trim]
            else:
                # Star isn't in file location
                self.download_spectrum(filename,SNR,past_rv_obs_path = False)


    def update_removedstars(self):
        ''' Helper method to insure removed_stars will be updated properly accross each method.
        
        Input: None

        Output: None
        '''
        logging.debug("CHIP.update_removedstars( )")

        # Update removed_stars
        logging.info(f"Current removed stars: {self.removed_stars}")
        # pd.DataFrame(self.removed_stars).to_csv( os.path.join(self.storage_path ,"removed_stars.csv"),
        #                                          index_label=False,
        #                                          index=False)
        # Use joblib to save removed_stars
        joblib.dump(self.removed_stars, os.path.join(self.storage_path ,"removed_stars.pkl"))


    def download_spectra(self):
        ''' Downloads all the spectra for each star in the NExSci, calculates 
        the SNR and saves the spectrum with the highest SNR for each star.

        Inputs: None

        Outputs: None
        '''
        logging.info("CHIP.download_spectra( )")

        start_time = time.perf_counter()

        # Cross matched names 
        cross_matched_file_path = self.config["Preprocessing"]["cross_match_stars"]["val"]
        cross_matched_df = pd.read_csv( cross_matched_file_path, sep=" " )
        hires_names_array = cross_matched_df["HIRES"].to_numpy()

        hiresID_fileName_snr_dic = {"HIRESid": [],"FILENAME":[],"SNR":[]}  

        # Location to save spectra
        spectra_download_location = os.path.join( self.storage_path, "rv_obs" )
        os.makedirs( spectra_download_location, exist_ok=True )

        # For retrieving data from HIRES
        self.state = Database('data/prv.cookies')
        self.dataSpectra = Download('data/prv.cookies', spectra_download_location)
                                  
        for star_ID in hires_names_array:
                try:
                    logging.debug(f"Finding highest SNR spectrum for {star_ID}")
                        
                    
                    # SQL query to find all the RV observations of one particular star
                    search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"
                    url = self.state.search(sql=search_string)
                    obs_df = pd.read_html(url, header=0)[0]

                    # Check if there are any RV Observations 
                    if obs_df.empty or not ("FILENAME" in obs_df.columns): 
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
                except:
                    logging.debug(f"Something went wrong with {star_ID} ")
                    self.removed_stars["No Clue"].append(star_ID) 
                    continue 

        # Save SNR meta data in csv file 
        self.hires_filename_snr_df = pd.DataFrame(hiresID_fileName_snr_dic) 
        self.hires_filename_snr_df.to_csv( os.path.join(self.storage_path ,"HIRES_Filename_snr.csv"),
                                           index_label=False,
                                           index=False)
        
        self.update_removedstars()

        # Delete unused instance attributes
        del self.dataSpectra
        del self.state

        end_time = time.perf_counter()
        logging.info(f"It took CHIP.download_spectra, {end_time - start_time} seconds to finish!")
    

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


    def alpha_normalization(self):
        ''' Rolling Continuum Normalization.  

        Input: None

        Output: None
        '''
        logging.info("CHIP.alpha_normalization( )")
        start_time = time.perf_counter()
        
        # Trim wl_solution 
        self.wl_solution = np.load("data/spocs/wl_solution.npy")
        self.wl_solution = self.wl_solution[:, self.trim: -self.trim]
    
        # Create Normalized Spectra dir
        self.norm_spectra_dir_path = os.path.join( self.storage_path, "norm" )
        os.makedirs(self.norm_spectra_dir_path,exist_ok=True) 

        # Start parallel computing
        Parallel( n_jobs = self.cores )\
                (delayed( contfit_alpha_hull )\
                        (star_name,
                         self.spectraDic[star_name],
                         self.ivarDic[star_name],
                         self.wl_solution,
                         self.norm_spectra_dir_path) for star_name in self.spectraDic)

        # Load all the normalized files into their respective dictionaries 
        for star_name in list(self.spectraDic):
            try:
                specnorm_path = os.path.join( self.norm_spectra_dir_path , f"{star_name}_specnorm.npy")
                sigmanorm_path = os.path.join( self.norm_spectra_dir_path , f"{star_name}_sigmanorm.npy")
                self.spectraDic[star_name] = np.load(specnorm_path) 
                self.ivarDic[star_name] = np.load(sigmanorm_path) 
            except FileNotFoundError as e:
                if isinstance(e,FileNotFoundError):
                    logging.error(f'''{star_name}'s normalization files were not found. We have removed the star.''')
                    del self.spectraDic[star_name]
                    del self.ivarDic[star_name]
                    self.removed_stars["Normalization error"].append(star_name)
        
        self.update_removedstars()
        
        end_time = time.perf_counter()
        logging.info(f"It took CHIP.alpha_normalization, {end_time - start_time} seconds to finish!")


    def cross_correlate_spectra(self):
        ''' Shift all spectra and sigmas to the rest wavelength. 

        Input: None

        Output: None
        '''
        logging.info("CHIP.cross_correlate_spectra( )")

        start_time = time.perf_counter()
        # the amount of pixels to ignore during cross-corrl
        numOfEdgesToSkip = 100

        # Load in stellar data
        solar = np.load('data/Constants/solarAtlas.npy')
        sun_wvlen = solar[:,1][::-1]
        sun_flux = solar[:,4][::-1]

        # The echelle orders that will be used for calculating 
        # each stars' cross correlation 
        # Found that the 15th order works just fine alone
        # If there are 16 echelle orders, the first echelle order is 0
        echelle_orders_list = [15,]
        # Key (int) echelle order number : tuple containing two 1-D numpy arrays representing 
        # wavelength of the echelle order then the flux in that echelle order.
        sollar_echelle_dic = {} 
        # Remove all the wavelengths that do fall in 
        # the HIRES wavelength solution's range 
        for echelle_order_num in echelle_orders_list:
            # The HIRES spectra's wavelength set needs to be 
            # in the range of the max and min of the solar's wavelength 
            offset_wvlens = 0
            echelle_mask = np.logical_and( (self.wl_solution[echelle_order_num][0]  - offset_wvlens) <= sun_wvlen,
                                           (self.wl_solution[echelle_order_num][-1] + offset_wvlens) >= sun_wvlen)
            # Apply Mask
            sun_echelle_wvlen = sun_wvlen[echelle_mask]
            sun_echelle_flux = sun_flux[echelle_mask]

            sollar_echelle_dic[echelle_order_num] = (sun_echelle_wvlen,
                                                    sun_echelle_flux)

        # Make cross correlate dir
        self.cross_correlate_dir_path = os.path.join(self.storage_path, "cr_cor")
        os.makedirs(self.cross_correlate_dir_path,exist_ok=True) 

        def cross_correlate_spectrum(filename):
            ''' Uses Pyastronomy's crosscorrRV function to compute the cross correlation.
                This will alter the key:value structure of self.spectraDic. self.spectraDic's
                key value pair will be  filename:(wavelength array, flux array)
            
            Input: filename (str): HIRES file name of spectrum you want to cross correlate 

            Output: None
            '''
            logging.info("CHIP.cross_correlate_spectrum( )")

            # Range of radial velocity values to choose from 
            # for example if set to 20, crosscorrRV will check 
            # [-20,20] in steps of 20/200.
            # 60 also gave good results
            RV = 80  

            #Going to take the average of all the echelle shifts
            z_list = []  
            for echelle_num in sollar_echelle_dic: #echelle orders
                # HIRES (h)
                h_wv = self.wl_solution[echelle_num]  
                h_flux = self.spectraDic[filename][echelle_num]
                # Solar (s)
                s_wv = sollar_echelle_dic[echelle_num][0]        
                s_flux = sollar_echelle_dic[echelle_num][1]

                rv, cc = pyasl.crosscorrRV(h_wv, h_flux,
                                        s_wv,s_flux, 
                                        -1*RV, RV, RV/200., 
                                        skipedge=numOfEdgesToSkip)
                
                argRV = rv[np.argmax(cc)]  #UNITS: km/s 
                z = (argRV/299_792.458) #UNITS: None 
                z_list.append(z)
                
            avg_z = np.mean(z_list)   
            shifted_wl = self.wl_solution.copy() / (1 + avg_z)

            # Save cross-correlated spectra 
            np.save( os.path.join(self.cross_correlate_dir_path, filename + "_shiftedwavelength.npy" ), shifted_wl )


        Parallel( n_jobs = self.cores )\
                (delayed( cross_correlate_spectrum )\
                (star_name) for star_name in list(self.spectraDic))
        
        end_time = time.perf_counter()
        logging.info(f"It took CHIP.cross_correlate_spectra, {end_time - start_time} seconds to finish!")


    def interpolate(self):
        ''' This method downloads the interpolated wavelength to interpolated_wl.npy 

        Input: None

        Output: None
        '''
        logging.info("CHIP.interpolate( )")

        start_time = time.perf_counter()

        # Make interpolation dir
        self.interpolate_dir_path = os.path.join(self.storage_path, "inter")
        os.makedirs(self.interpolate_dir_path,exist_ok=True) 

        # Create an array filled with -np.inf,np.inf 
        # 0th column is the min wavelength for that echelle order
        # 1st column is the min wavelength for that echelle order
        all_min_max_echelle = np.full((16,2), np.inf)
        all_min_max_echelle[:,0] = -all_min_max_echelle[:,0]

        for star_name in list(self.spectraDic):
            star_file_path = os.path.join(self.cross_correlate_dir_path, star_name + "_shiftedwavelength.npy")
            star_rest_wavelength = np.load( star_file_path )
            
            # Given an array star_rest_wavelength, that is (16,N). Produce array B, 
            # that is (16,2) where the 0th column is the first 
            # element from each row from star_rest_wavelength, and the 1st column 
            # is the last element of each row from star_rest_wavelength.
            star_min_max_echelle = np.column_stack((star_rest_wavelength[:,0], star_rest_wavelength[:,-1]))

            # Update the values in all_min_max_echelle with the corresponding values in star_min_max_echelle
            all_min_max_echelle[:,0] = np.maximum(all_min_max_echelle[:,0], star_min_max_echelle[:,0])
            all_min_max_echelle[:,1] = np.minimum(all_min_max_echelle[:,1], star_min_max_echelle[:,1])

        # Create a boolean mask indicating which elements of wl_solution fall within the ranges in all_min_max_echelle
        mask = np.logical_and(self.wl_solution >= all_min_max_echelle[:,0:1], 
                              self.wl_solution <= all_min_max_echelle[:,1:2])

        # Use the mask to select the desired elements of wl_solution then save 
        filtered_wl_solution = self.wl_solution[mask]
        np.save( os.path.join(self.interpolate_dir_path, "wl.npy"), filtered_wl_solution )

        # Apply mask to all the spectra and ivars then save 
        for star_name in list(self.spectraDic):
            spec_path = os.path.join( self.interpolate_dir_path , f"{star_name}_spec.npy")
            ivar_path = os.path.join( self.interpolate_dir_path , f"{star_name}_ivar.npy")

            np.save( spec_path, self.spectraDic[star_name][mask] )
            np.save( ivar_path, self.ivarDic[star_name][mask]    )

        end_time = time.perf_counter()
        logging.info(f"It took CHIP.interpolate, {end_time - start_time} seconds to finish!")


    def load_the_cannon(self):
        ''' Load in the data The Cannon will use. 
        
        Input: None

        Output: None
        '''
        logging.info("CHIP.load_the_cannon( )") 

        self.random_seed = self.config["Training"]["random seed"]["val"]

        interpolated_dir_path = os.path.join(self.data_dir_path, "inter")

        # Load wavelength solution
        self.wl_solution = np.load( os.path.join( interpolated_dir_path , "wl.npy") )

        # load spectra 
        # Note: the key is hiresid instead of filename 
        cross_match_array = pd.read_csv( os.path.join( self.data_dir_path,"HIRES_Filename_snr.csv"))[["HIRESid",'FILENAME']].to_numpy()
        for hiresid, filename in cross_match_array:
            spec_path = os.path.join( interpolated_dir_path, filename + "_spec.npy" )
            ivar_path = os.path.join( interpolated_dir_path, filename + "_ivar.npy" )
            if os.path.exists( spec_path ):
                if os.path.exists( ivar_path ):
                    self.spectraDic[hiresid] = np.load(spec_path)
                    self.ivarDic[hiresid]    = np.load(ivar_path)
                else:
                    logging.info(f"{filename} does not what an ivar in {interpolated_dir_path}")
            else:
                logging.info(f"{filename} does not what an spectrum in {interpolated_dir_path}")
        
        logging.info(f"A total of {len(self.spectraDic)} stars were loaded.")

        # Load parameters 
        self.parameters_list = self.config["Training"]["stellar parameters"]["val"]
        hiresid_parameters_list = ["HIRESID"] + self.parameters_list
        stellar_parameters_path = self.config["Training"]["stellar parameters path"]["val"]
        if not os.path.exists( stellar_parameters_path ):
            logging.info(f"stellar parameters path {stellar_parameters_path} does not exist")
            sys.exit(1)
        self.parameters_df = pd.read_csv(stellar_parameters_path)[ hiresid_parameters_list ]
        # Extract only the stars that were preprocessed 
        self.parameters_df = self.parameters_df[self.parameters_df["HIRESID"].isin( cross_match_array[:,0] )]

        logging.info("before scaling\n" + self.parameters_df.to_string())
        # Create a StandardScaler object
        self.parameters_scaler = StandardScaler()
        # Fit the scaler to the selected columns and transform the selected columns
        parameters_transformed = self.parameters_scaler.fit_transform( self.parameters_df[self.parameters_list] )

        # change values in df 
        self.parameters_df[self.parameters_list] = parameters_transformed

        logging.info( "scaled features\n" + self.parameters_df.to_string() )

        self.cannon_splits(self.parameters_df)

        # load cost function 
        self.cost_function = eval(self.config["Training"]["cost function"]["function"])
        # Load masks 
        masks_list = self.config["Training"]["masks"]["val"]
        self.masks = {mask_name: self.create_mask_array(mask_path) for mask_name, mask_path in masks_list}


    def create_mask_array(self,mask_path):
        ''' Create a mask array for the The Cannon. 
        
        Input: mask_path (str) - path to the mask
               
        Output: (np.array((wl_solution.shape[0],))) - masked boolean array
        '''
        logging.debug("CHIP.create_mask_array( mask_path={mask_path}})") 

        # Create a (wl_solution.shape[0],) array of Falses
        masked_bool_array = np.zeros(self.wl_solution.shape[0], dtype=bool)

        if not mask_path:
            # If no mask path is given, return the array of Falses
            return masked_bool_array

        mask = np.load(mask_path)
        mask_flux = mask[1,:]
        mask_wl = mask[0,:]

        wl_range = []
        continuing_range = False 
        beginning_of_range = 0
        for i,pixel_val in enumerate(mask_flux):
            if pixel_val == 0:
                if not continuing_range:
                    beginning_of_range = mask_wl[i]
                    continuing_range = True
            else:
                if continuing_range: 
                    wl_range.append( (beginning_of_range,mask_wl[i]) )
                    continuing_range = False

        
        # Mask the wavelength solution
        for masked_range in wl_range:
            start_masked_wl = masked_range[0]
            end_masked_wl = masked_range[1]

            # Use numpy logical and to mask the wavelength solution
            mask_out_i = np.logical_and(self.wl_solution >= start_masked_wl, self.wl_solution <= end_masked_wl)
            masked_bool_array = np.logical_or(masked_bool_array, mask_out_i)
        
        return masked_bool_array


    def evaluate_model(self, md, ds, true_labels, save = False):
        ''' Evaluate how well a model was trained.

        Input: md (TheCannon.model.CannonModel) - 
               ds (TheCannon.dataset.Dataset) -
               true_labels (np.array((M,))) - true values for the corresponds inferred labels 
        
        '''
        label_errors = md.infer_labels(ds)
        inferred_labels = ds.test_label_vals

        if save:
            # Save inferred labels
            joblib.dump(inferred_labels, os.path.join(self.storage_path,'inferred_labels.joblib'))


        return self.cost_function(true_labels, inferred_labels)
        

    def split_data(self, X_indecies, y_indecies):
        ''' Split data to be put in TheCannon.dataset.Dataset

        Input: X_indecies (np.array((N,))) - indecies to be used for the training set
               y_indecies (np.array((M,))) - indecies to be used for the testing set
        
        Output: training id, training spectra, training ivar, training parameters, testing id, testing spectra, testing ivar, testing parameters
        '''
        # np.array( array , dtype=np.float64) is necessary, otherwise you would recieve the following type error 
        # TypeError: No loop matching the specified signature and casting was found for ufunc solve1
        X_id = self.train_id[X_indecies]
        X_spec = np.array(self.train_spectra[X_indecies], dtype=np.float64)
        X_ivar = np.array(self.train_ivar[X_indecies], dtype=np.float64)
        # Remove 0th column that contains HIRES IDs
        X_parameters = np.array(self.train_parameter.to_numpy()[X_indecies][:,1:], dtype=np.float64)

        y_id = self.train_id[y_indecies]
        y_spec = np.array(self.train_spectra[y_indecies], dtype=np.float64)
        y_ivar = np.array(self.train_ivar[y_indecies], dtype=np.float64)
        # Remove 0th column that contains HIRES IDs
        y_parameters = np.array(self.train_parameter.to_numpy()[y_indecies][:,1:], dtype=np.float64)

        return X_id, X_spec, X_ivar, X_parameters, y_id, y_spec, y_ivar, y_parameters


    def train_model(self,batch_size, poly_order, mask_name, test_set=False):
        ''' Train a Cannon model using mini-batch and k-fold cv

        Input: batch_size (int) : number of batches to split the training set into for mini-batch training
               poly_order (int) : A positive int, tells the model what degree polynomial to fit
               mask_name (str) : name of the mask to use
               test_set (bool) : If True, the model will be evaluated on the test set, instead of the validation set
        
        Output: The mean evaluation score 
        '''
        logging.info(f"train_model(batch_size = {batch_size}, poly_order={poly_order}, mask_name={mask_name}, test_set={test_set})")
        
        def mini_batch(cannon_model,batch_size,X_id, X_spec, X_ivar, X_param, y_id, y_spec, y_ivar):
            ''' Train a Cannon model using mini-batch

            Input: cannon_model (TheCannon.model.CannonModel) - A Cannon model
                   batch_size (int) : number of batches to split the training set into for mini-batch training
                   X_id (np.array((M,))) - HIRES identifers that correspond to the rows in X_flux, X_ivar, and X_parameter
                   X_flux (np.array((M,N))) - flux for each star in X_id 
                   X_ivar (np.array((M,N))) - ivar for each star in X_id 
                   X_parameter (np.array((M,P))) - contains all the parameters for each star in X_id (in the exact same order)
                   y_id (np.array((V,))) - HIRES identifers that correspond to the rows in y_flux, y_ivar
                   y_flux (np.array((V,N))) - flux for each star in y_id 
                   y_ivar (np.array((V,N))) - ivar for each star in y_id 
            
            Output: cannon_model (TheCannon.model.CannonModel) - A trained Cannon model

            '''
            # Create mini-batches of the data
            num_batches = int(np.ceil(X_spec.shape[0] / batch_size))

            break_out = False
            for i in range(num_batches):
                # Get the start and end indices of the current mini-batch
                start = i * batch_size
                end = min((i + 1) * batch_size, X_spec.shape[0])

                # Number of training examples can't be smaller than 3  
                # check the next batch if it is smaller than 3 than add it to the current batch
                # Only do this if it is not the last batch
                if (i == num_batches - 2):
                    next_start = (i + 1) * batch_size
                    next_end = min((i + 2) * batch_size, X_spec.shape[0])
                    if (next_end - next_start) < 3:
                        end = next_end
                        break_out = True

                # Initialize the dataset
                ds = self.initailize_dataset(self.wl_solution,
                                             X_id[start:end], X_spec[start:end], X_ivar[start:end], X_param[start:end], 
                                             y_id, y_spec, y_ivar, self.parameters_list)

                # Fit the model on the current batch
                try:
                    cannon_model.fit(ds)
                except:
                    logging.error(f"Error: cannon_model.fit(ds) failed on batch {i}")
                    return cannon_model

                if break_out:
                    break

            return cannon_model

        def apply_mask(X_spec, X_ivar, y_spec, y_ivar,mask):
            ''' Mask the spectra and ivars

            Input:  X_spec (np.array((M,N))) - flux for each star in X_id
                    X_ivar (np.array((M,N))) - ivar for each star in X_id
                    y_spec (np.array((V,N))) - flux for each star in y_id
                    y_ivar (np.array((V,N))) - ivar for each star in y_id

            Output: X_spec (np.array((M,N))) - mask applied flux for each star in X_id
                    X_ivar (np.array((M,N))) - mask applied ivar for each star in X_id
                    y_spec (np.array((V,N))) - mask applied flux for each star in y_id
                    y_ivar (np.array((V,N))) - mask applied ivar for each star in y_id
            '''
            # Create copies of the arrays to avoid changing the original arrays
            Xspec, Xivar, yspec, yivar = X_spec.copy(), X_ivar.copy(), y_spec.copy(), y_ivar.copy()

            # To do it without using a for-loop 
            Xspec[:,mask], Xivar[:,mask] = 0,0
            yspec[:,mask], yivar[:,mask] = 0,0


            return Xspec, Xivar,yspec, yivar

        if not test_set:
            # Store the evaluations
            evaluation_list = []
            # Initialize The Cannon model 
            
            for X_i, y_i in self.kfold_train_validation_splits:

                # Initialize new model
                cannon_model = self.initialize_model(poly_order = poly_order)

                # Split training and validation
                X_id, X_spec, X_ivar, X_param, y_id, y_spec, y_ivar, y_param = self.split_data(X_i, y_i)

                # Mask the spectra and ivars
                X_spec, X_ivar, y_spec, y_ivar = apply_mask(X_spec, X_ivar, y_spec, y_ivar, self.masks[mask_name])

                # Train the model
                cannon_model = mini_batch(cannon_model,batch_size,X_id, X_spec, X_ivar, X_param, y_id, y_spec, y_ivar)

                ds = self.initailize_dataset(self.wl_solution,
                                                X_id, X_spec, X_ivar, X_param, 
                                                y_id, y_spec, y_ivar, self.parameters_list)

                # Evaluate model
                evaluation_list.append(self.evaluate_model(cannon_model,ds,y_param))
            
            # Store the mean evaluation score into a file 
            score = np.mean(evaluation_list)
            logging.info(f"{batch_size},{poly_order},{score}")
            return score

        else:
            # Initialize new model
            cannon_model = self.initialize_model(poly_order = poly_order)

            X_id, X_spec, X_ivar = self.train_id, self.train_spectra, self.train_ivar, 
            y_id, y_spec, y_ivar = self.test_id, self.test_spectra, self.test_ivar, 

            # Mask the spectra and ivars
            X_spec, X_ivar, y_spec, y_ivar = apply_mask(X_spec, X_ivar, y_spec, y_ivar, self.masks[mask_name])

            # [:,1:] to remove the first column which is the abundance name 
            X_param = np.array(self.train_parameter.to_numpy()[:,1:], dtype=np.float64) 
            y_param = np.array(self.test_parameter.to_numpy()[:,1:], dtype=np.float) 

            # Train the model
            cannon_model = mini_batch(cannon_model,batch_size,X_id, X_spec, X_ivar, X_param, y_id, y_spec, y_ivar)

            ds = self.initailize_dataset(self.wl_solution,
                                         X_id, X_spec, X_ivar, X_param, 
                                         y_id, y_spec,y_ivar, 
                                         self.parameters_list)
            # Save test set 
            test_set_filepath = os.path.join(self.storage_path, "y_param.joblib")  
            joblib.dump(y_param, test_set_filepath)
            # Save ds 
            ds_filepath = os.path.join(self.storage_path, "ds.joblib")
            joblib.dump(ds, ds_filepath)

            # Evaluate model
            # Store the mean evaluation score into a file 
            score = self.evaluate_model(cannon_model,ds,y_param, save = True)
            logging.info(f"Best model when trained on entire test set: {score}")
            return cannon_model
                

    @staticmethod
    def initialize_model(poly_order = 1):
        ''' Initialize The Cannon Model

        Input: poly_order (int) : A positive int, tells the model what degree polynomial to fit

        Output: TheCannon.model.CannonModel object 
        '''
        md = model.CannonModel( order = poly_order, useErrors=False )  
        return md 


    @staticmethod
    def initailize_dataset(wl_sol, X_id, X_flux, X_ivar, X_parameter, y_id, y_flux, y_ivar, parameters_names ):
        ''' Put data into data structure The Cannon will train on.

        Input: wl_sol (np.array((N,))) - wavelength solution for all spectra
               X_id (np.array((M,))) - HIRES identifers that correspond to the rows in X_flux, X_ivar, and X_parameter
               X_flux (np.array((M,N))) - flux for each star in X_id 
               X_ivar (np.array((M,N))) - ivar for each star in X_id 
               X_parameter (np.array((M,P))) - contains all the parameters for each star in X_id (in the exact same order)
               y_id (np.array((V,))) - HIRES identifers that correspond to the rows in y_flux, y_ivar
               y_flux (np.array((V,N))) - flux for each star in y_id 
               y_ivar (np.array((V,N))) - ivar for each star in y_id 
               parameters_names (np.array((P,))) - column names of X_parameter and y_parameter
              
        Output: TheCannon.dataset.Dataset object 
        '''

        ds = dataset.Dataset(wl_sol, X_id, X_flux, X_ivar, X_parameter, y_id, y_flux, y_ivar) 
        ds.set_label_names(parameters_names) 
        # wl ranges can be optimized for specific echelle ranges
        ds.ranges= [[np.min(wl_sol),np.max(wl_sol)]]
        return ds


    def cannon_splits(self, parameters_df):
        ''' Apply test and validation splits to the data. This must be ran after self.parameters_df is created.
        
        Input: parameters_df (pd.DataFrame) : contains all the parameters for each star in X_id (in the exact same order)

        Output: None
        '''
        logging.debug("CHIP.cannon_splits( )")

        # Split the parameters into a training set, a test set, and a validation set
        test_frac = self.config["Training"]["train test split"]["val"]
        self.train_parameter, self.test_parameter = train_test_split(parameters_df, test_size = test_frac, random_state= self.random_seed)

        # Split the spectra and ivars 
        stars_in_test = [name for name in self.test_parameter["HIRESID"]]
        stars_in_train = [name for name in self.train_parameter["HIRESID"]]

        self.test_spectra = np.vstack([ self.spectraDic[name] for name in  stars_in_test ])
        self.test_ivar = np.vstack([ self.ivarDic[name] for name in  stars_in_test ])
        self.test_id = np.array([name for name in  stars_in_test if name in self.spectraDic])
        self.train_spectra = np.vstack([ self.spectraDic[name] for name in  stars_in_train ])
        self.train_ivar = np.vstack([ self.ivarDic[name] for name in  stars_in_train ])
        self.train_id = np.array([name for name in  stars_in_train if name in self.spectraDic])

        logging.info("train_id" + str(self.train_id))
        # No longer needed 
        del self.spectraDic
        del self.ivarDic

        # Create a KFold object for preforming k-fold cross validation  
        num_folds = self.config["Training"]["kfolds"]["val"]
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.random_seed)
        # So repeated call doesn't need to be made to kf every time. 
        self.kfold_train_validation_splits = list(kf.split(self.train_id)) #np.vstack(kf.split(self.train_spectra))
        logging.info(f"{num_folds}-fold splits" + str(self.kfold_train_validation_splits))


    def hyperparameter_tuning(self):
        ''' Tune the hyperparameters of The Cannon model. This must be ran after self.cannon_splits is ran.

        Input: None

        Output: None
        '''
        logging.debug("CHIP.hyperparameter_tuning( )")

        # Get the hyperparameters to tune
        # batch size is the number of spectra to train on at a time (list) 
        batch_size = self.config["Training"]["batch size"]["val"]
        # poly_order is the degree of the polynomial to fit (list)
        poly_order = self.config["Training"]["poly order"]["val"]

        # Create a list of all the hyperparameters to tune
        hyperparameters = [batch_size, poly_order, self.masks]

        # Create a list of all the hyperparameter names
        hyperparameter_names = ["batch_size", "poly_order","mask"]
        
        # Create a list of all the hyperparameter combinations
        hyperparameter_combinations = list(itertools.product(*hyperparameters))
        logging.info("hyperparameter combinations" + str(hyperparameter_combinations))

        # Use joblib to parallelize the hyperparameter tuning
        num_cores = self.config["Training"]["cores"]["val"]
        results = Parallel(n_jobs=num_cores)\
                          (delayed(self.train_model)\
                          (hyperparameter_combination[0],hyperparameter_combination[1],hyperparameter_combination[2]) for hyperparameter_combination in hyperparameter_combinations)
       

        # Log the results with the hyperparameters
        for i, hyperparameter_combination in enumerate(hyperparameter_combinations):
            logging.info(f"{hyperparameter_names[0]}={hyperparameter_combination[0]}, {hyperparameter_names[1]}={hyperparameter_combination[1]}, {hyperparameter_names[2]}={hyperparameter_combination[2]}, result={results[i]}")
    

        # Get the best hyperparameters
        best_hyperparameters = hyperparameter_combinations[np.argmin(results)]
        # Log the best hyperparameters
        logging.info(f"best hyperparameters: {hyperparameter_names[0]}={best_hyperparameters[0]}, {hyperparameter_names[1]}={best_hyperparameters[1]}, {hyperparameter_names[2]}={best_hyperparameters[2]}") 

        # Train the model with the best hyperparameters
        self.train_best_model(best_hyperparameters[0], best_hyperparameters[1], best_hyperparameters[2])


    def train_best_model(self, batch_size, poly_order, mask_name):
        ''' Train the best model with the best hyperparameters. This must be ran after self.hyperparameter_tuning is ran.

        Input: batch_size (int) the 
               poly_order (int)

        Output: None
        '''
        logging.debug(f"CHIP.train_best_model( batch_size={batch_size}, poly_order={poly_order}, mask_name={mask_name})")

        # Train the model with the best hyperparameters
        cannon_model = self.train_model(batch_size, poly_order, mask_name, test_set=True)

        # Save the model
        self.save_model(cannon_model)


    def save_model(self,cannon_model):
        '''
        Save the model to a file
        '''
        logging.debug("CHIP.save_model( )")

        model_filepath = os.path.join(self.storage_path, "best_model.joblib")

        joblib.dump(cannon_model, model_filepath)

        # Save the scaler
        transformer_filepath = os.path.join(self.storage_path, "standard_scaler.joblib")
        joblib.dump(self.parameters_scaler, transformer_filepath)

        # Save the parameters names
        parameters_filepath = os.path.join(self.storage_path, "parameters_names.joblib")
        joblib.dump(self.parameters_list, parameters_filepath)




if __name__ == "__main__":

    log_filepath = 'data/CHIP.log'
    logging.basicConfig(filename= log_filepath,
                        format='%(asctime)s - %(message)s', 
                        datefmt="%Y/%m/%d %H:%M:%S",  
                        level=logging.INFO)

    # logs to file and stdout
    logging.getLogger().addHandler(logging.StreamHandler())
    # set datefmt to GMT
    logging.Formatter.converter = time.gmtime

    chip = CHIP()
    chip.run()

    # Move logging file to the location of this current run
    log_filename = os.path.basename(log_filepath)
    # Shutdown logging so the file can be put in the storage location
    logging.shutdown()

    os.rename( log_filepath, 
                os.path.join( chip.storage_path, log_filename) )