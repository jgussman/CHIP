import glob
import itertools
import json
import logging
import os
import shutil
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import specmatchemp.library
from astropy.io import fits
from hiresprv.auth import login
from hiresprv.database import Database
from hiresprv.download import Download
from hiresprv.idldriver import Idldriver
from joblib import Parallel, delayed
from pylab import *
from scipy import interpolate
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from specmatchemp import spectrum
from specmatchemp.specmatch import SpecMatch
from TheCannon import dataset, model

from alpha_shapes import contfit_alpha_hull


class CHIP:

    def __init__(self, config_file_path):
        '''

        Args: 
            config_file_path (str): Path to the config.json file

        Returns:
            None
        '''
        # Check if config file exists
        if not os.path.exists(config_file_path):
            logging.error(f"Config file {config_file_path} does not exist")
            sys.exit()
    

        logging.info(f"Using config file: {config_file_path}")
        self.config_file_path = config_file_path

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
        ''' Creates a unique subdirectory in the data directory to store the outputs of CHIP'''
        logging.debug("CHIP.create_storage_location( )")

        # If we are running preprocessing then we want a new 
        # CHIP run sub dir else we are running The Cannon
        # and want to use a previous run 
        if self.config["Pre-processing"]["run"]["val"]:
            # Create a new storage location for this pipeline
            # Greenwich Mean Time (UTC) right now 
            gmt_datetime = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

            self.storage_path = os.path.join("data","chip_runs" , gmt_datetime )


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
        ''' Run the pipeline from end to end.'''
        
        if self.config["Pre-processing"]["run"]["val"]:

            # Record results 
            self.removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], 
                                "SNR < 100":[],"NAN value in spectra":[],
                                "No Clue":[],"Nan in IVAR":[],
                                "Normalization error":[]}
            
            # Trim wl_solution 
            self.wl_solution = np.load("data/spocs/wl_solution.npy")
            if self.trim > 0:
                self.wl_solution = self.wl_solution[:, self.trim: -self.trim]

            if isinstance(self.config["Pre-processing"]["run"]["val"],bool):

                self.download_spectra()

                self.alpha_normalization()

                self.cross_correlate_spectra()

                self.interpolate()

            else:
                past_run, data_folder = self.config["Pre-processing"]["run"]["val"][0], self.config["Pre-processing"]["run"]["val"][1]
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

                        print(self.hires_filename_snr_df)
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

            self.load_the_cannon()
            self.hyperparameter_tuning()

        else:
            logging.error("Neither CHIP or The Cannon were selected to run in config.json! Please select!")


    def load_past_rv_obs(self,data_folder_path):
        ''' Load in past data to continue preprocessing

        Args: 
            data_folder_path (str): File path to rv_obs folder
        '''
        for _, row in self.hires_filename_snr_df.iterrows():
            # Save the Best Spectrum
            star_id = row["HIRESID"]
            filename = row["FILENAME"]
            self.spectraDic[filename] = self.download_spectrum(filename, 
                                                                snr=False,
                                                                past_rv_obs_path=data_folder_path)
            # Calculate ivar
            self.sigma_calculation(filename , star_id)


    def get_arguments(self):
        ''' Get arguments from src/config.json and store in self.config'''
        logging.debug("CHIP.get_arguments( )")

        with open(self.config_file_path, "r") as f:
            self.config = json.load(f)
        
        self.cores = self.config["Pre-processing"]["cores"]["val"]
        self.trim  = self.config["Pre-processing"]["trim spectrum"]["val"]
        logging.info( f"config.json : {self.config}" )
    

    @staticmethod
    def calculate_SNR(spectrum, gain = 2.09):
        ''' Calculates the SNR of spectrum using the 5th echelle orders

        Args: 
            spectrum (np.array) - spectrum with at least 5 rows representing echelle orders 
            gain (float) - gain of the detector

        Outputs: 
            (float) estimated SNR of inputed spectrum
        '''
        logging.debug(f"CHIP.calculate_SNR( {spectrum} )")

        spectrum = spectrum[4].flatten() # Using echelle orders in the middle 
        SNR = np.sqrt(np.median(spectrum * gain))
        
        return SNR 


    def delete_spectrum(self,filename):
        '''Remove spectrum from local storage

        Args: 
            filename (str): HIRES file name of spectrum you want to delete
        '''
        logging.debug(f"CHIP.delete_spectrum( {filename} )")
        file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")
        try:
            os.remove(file_path)
        except FileNotFoundError as e:
            # file was already deleted 
            pass 


    def download_spectrum(self,filename,snr = True, past_rv_obs_path = False):
        '''Download Individual Spectrum and calculate the ivar 

        Args: 
            filename (str): HIRES file name of spectrum you want to download
            snr (bool): if you want to calculate the SNR of the spectrum 
            past_rv_obs_path (str): if you want to load in old rb obs set to rb_obs dir path
        '''
        logging.debug(f"CHIP.download_spectrum( filename={filename}, snr={snr}, past_rv_obs_path={past_rv_obs_path} )")
        
        if not past_rv_obs_path:
            #Download spectra
            self.dataSpectra.spectrum(filename.replace("r",""))    
            file_path = os.path.join(self.dataSpectra.localdir, filename + ".fits")

            try: 
                temp_deblazedFlux = fits.getdata(file_path)
            except OSError:
                # There is a problem with the downloaded fits file 
                return -1
            
            if snr: # Used to find best SNR 
                snr = self.calculate_SNR(temp_deblazedFlux)

                # delete Spectrum variable so it can be delete if needed
                del temp_deblazedFlux 
                return snr
            else:
                # Trim the left and right sides of each echelle order 
                if self.trim > 0:
                    return temp_deblazedFlux[:, self.trim: -self.trim]
                else:
                    return temp_deblazedFlux
            
        else:
            # Load past data 
            file_path = os.path.join(past_rv_obs_path, filename + ".fits")
            if os.path.exists(file_path):
                temp_deblazedFlux = fits.getdata(file_path)
                if self.trim > 0:
                    return temp_deblazedFlux[:, self.trim: -self.trim]
                else:
                    return temp_deblazedFlux
            
            else:
                # Star isn't in file location
                self.download_spectrum(filename,
                                       snr,
                                       past_rv_obs_path = False)


    def update_removedstars(self):
        ''' Helper method to insure removed_stars will be updated properly accross each method.'''
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
        '''
        logging.info("CHIP.download_spectra( )")

        # login into the NExSci servers
        login('data/prv.cookies')

        start_time = time.perf_counter()

        # IDs for all the stars the user wants to download iodine imprinted spectra for
        hires_stars_ids_file_path = self.config["Pre-processing"]["HIRES stars IDs"]["val"]
        hires_stars_ids_df = pd.read_csv( hires_stars_ids_file_path, sep=" " )
        hires_names_array = hires_stars_ids_df["HIRESID"].to_numpy()

        hiresID_fileName_snr_dic = {"HIRESID": [],"FILENAME":[],"SNR":[]}  

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
                                                                                        snr=False)
                            
                            logging.debug(f"{star_ID}'s best SNR spectrum came from {best_SNR_filename} with an snr={best_SNR}")
                            hiresID_fileName_snr_dic["HIRESID"].append(star_ID)
                            hiresID_fileName_snr_dic["FILENAME"].append(best_SNR_filename)
                            hiresID_fileName_snr_dic["SNR"].append(best_SNR)
                        
                            # Calculate ivar
                            self.sigma_calculation(best_SNR_filename , star_ID)
                except Exception as e:
                    logging.debug(f"{star_ID} was removed because it recieved the following error: {e}")

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

        Args: 
            filename (str): HIRES file name of spectrum you want to calculate IVAR for
            star_ID (str): HIRES identifer
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
        ''' Rolling Continuum Normalization.'''
        logging.info("CHIP.alpha_normalization( )")
        start_time = time.perf_counter()
        
        # Create Normalized Spectra dir
        self.norm_spectra_dir_path = os.path.join( self.storage_path, "norm" )
        os.makedirs(self.norm_spectra_dir_path,exist_ok=True) 

        # for star_name in self.spectraDic: 
        #     contfit_alpha_hull(star_name,
        #                         self.spectraDic[star_name],
        #                         self.ivarDic[star_name],
        #                         self.wl_solution,
        #                         self.norm_spectra_dir_path)
        # Start parallel computing
        Parallel( n_jobs = self.cores )\
                (delayed( contfit_alpha_hull )\
                        (star_name,
                         self.spectraDic[star_name],
                         self.ivarDic[star_name],
                         self.wl_solution,
                         self.norm_spectra_dir_path) for star_name in self.spectraDic)

        # Load all the normalized files into their respective dictionaries 
        self.star_name_list = []
        for star_name in list(self.spectraDic):
            try:
                specnorm_path = os.path.join( self.norm_spectra_dir_path , f"{star_name}_specnorm.npy")
                sigmanorm_path = os.path.join( self.norm_spectra_dir_path , f"{star_name}_sigmanorm.npy")
                # self.spectraDic[star_name] = np.load(specnorm_path) 
                # self.ivarDic[star_name] = np.load(sigmanorm_path) 
                self.star_name_list.append(star_name)


            except FileNotFoundError as e:
                if isinstance(e,FileNotFoundError):
                    logging.error(f'''{star_name}'s normalization files were not found. We have removed the star.''')
                    self.removed_stars["Normalization error"].append(star_name)
        
        self.update_removedstars()
        del self.spectraDic
        
        end_time = time.perf_counter()
        logging.info(f"It took CHIP.alpha_normalization, {end_time - start_time} seconds to finish!")


    def cross_correlate_spectrum(self, filename):
            ''' Uses specmatch-emp to cross correlate a spectrum to the rest wavelength. 
            
            Args: 
                filename (str): HIRES file name of spectrum you want to cross correlate
            '''
            logging.info(f"CHIP.cross_correlate_spectrum( filename = {filename} )")

            try:
                hiresspectrum = spectrum.read_chip_spectrum(normalized_spectra_dir = self.norm_spectra_dir_path ,
                                            HIRES_id = filename, 
                                            wavelength = self.wl_solution,)
            except FileNotFoundError as e:
                if isinstance(e,FileNotFoundError):
                    logging.error(f'''{filename}'s normalization files were not found. We have removed the star.''')
                    self.removed_stars["Normalization error"].append(filename)

            specmatch_object = SpecMatch(hiresspectrum, self.lib)
            specmatch_object.shift()
            
            # Save cross-correlated spectra 
            np.save( os.path.join(self.cross_correlate_dir_path,
                                  filename + "_wavelength.npy")
                    ,specmatch_object.target.w)
            np.save( os.path.join(self.cross_correlate_dir_path,
                                  filename + "_flux.npy")
                    ,specmatch_object.target.s)
            np.save( os.path.join(self.cross_correlate_dir_path,
                            filename + "_ivar.npy")
                    ,specmatch_object.target.serr)
            
            ### NEED TO FIGURE OUT WHY MATPLOTLIB GIVES A LOCK ISSUE WHEN MULTIPROCESSING
            # # Derived from specmatch-emp quick-start tutorial
            # fig = plt.figure(figsize=(10,5))
            # specmatch_object.target_unshifted.plot(normalize=True, 
            #                                        plt_kw={'color':'forestgreen'}, 
            #                                        text='Target (unshifted)')
            # specmatch_object.target.plot(offset=0.5, 
            #                              plt_kw={'color':'royalblue'}, 
            #                              text= f'Target (shifted): {filename}')
            # specmatch_object.shift_ref.plot(offset=1, 
            #                                 plt_kw={'color':'firebrick'}, 
            #                                 text='Reference: '+specmatch_object.shift_ref.name)
            # plt.xlim(5160,5200)
            # plt.ylim(0,2.2)
            # # code-stop-plot-shifts-G
            # fig.set_tight_layout(True)
            # fig.savefig(os.path.join(self.cross_correlate_dir_path,
            #                          filename + "_comparison.png"))
            # plt.close(fig)

    def cross_correlate_spectra(self):
        ''' Shift all spectra and sigmas to the rest wavelength.'''
        logging.info("CHIP.cross_correlate_spectra( )")
        start_time = time.perf_counter()

        # Make cross correlate dir
        self.cross_correlate_dir_path = os.path.join(self.storage_path, "cr_cor")
        os.makedirs(self.cross_correlate_dir_path,exist_ok=True) 

        # Library 
        # Min wavelength for HIRES wavelength solution is 4976.64...
        # Max wavelength for HIRES wavelength solution is 6421.36...
        # So I'll use 4950 and 6450 as the wavelength limits for the library 
        self.lib = specmatchemp.library.read_hdf(wavlim=[4950,6450]) 

        Parallel( n_jobs = self.cores )\
                (delayed( self.cross_correlate_spectrum )\
                (star_name) for star_name in self.star_name_list)
        
        end_time = time.perf_counter()
        logging.info(f"It took CHIP.cross_correlate_spectra, {end_time - start_time} seconds to finish!")


    @staticmethod
    def compute_wavelength_limits(filenames):
        smallest_maxima = float("inf")
        largest_minima = float("-inf")
        for filename in filenames:
            temp_wv = np.load(filename)
            smallest_maxima = min(smallest_maxima, np.max(temp_wv))
            largest_minima = max(largest_minima, np.min(temp_wv))
        return smallest_maxima, largest_minima

    @staticmethod
    def interpolate_spectrum(filename, common_wv):
        temp_wv = np.load(filename)
        temp_flux = np.load(filename.replace("wavelength", "flux"))  # assumes a corresponding flux file exists
        temp_ivar = np.load(filename.replace("wavelength", "ivar"))  # assumes a corresponding ivar file exists

        # Replace inf and nan values in the flux with 1.0
        temp_flux = np.where(np.isfinite(temp_flux), temp_flux, 1.0)

        interp_kind = 'cubic'

        f_flux = interpolate.interp1d(temp_wv, 
                                      temp_flux, 
                                      kind=interp_kind, 
                                      bounds_error=False, 
                                      fill_value=np.nan)
        f_ivar = interpolate.interp1d(temp_wv, 
                                      temp_ivar, 
                                      kind=interp_kind, 
                                      bounds_error=False, 
                                      fill_value=np.nan)
        
        resampled_flux = f_flux(common_wv)
        resampled_ivar = f_ivar(common_wv)

        # Replace nan values with 0.0
        resampled_flux = np.nan_to_num(resampled_flux)
        resampled_ivar = np.nan_to_num(resampled_ivar)

        return temp_wv, temp_flux, temp_ivar, resampled_flux, resampled_ivar


    def interpolate(self):  
        logging.info("CHIP.interpolate( )")

        start_time = time.perf_counter()

        # Make interpolation dir
        self.interpolate_dir_path = os.path.join(self.storage_path, "inter")
        os.makedirs(self.interpolate_dir_path,exist_ok=True) 

        # Grab all the filenames that end with *_wavelength.npy in the cross_correlate_dir_path
        filenames = glob.glob(self.cross_correlate_dir_path + r"\*_wavelength.npy")

        smallest_maxima, largest_minima = self.compute_wavelength_limits(filenames)

        # Implement the algorithm
        last_numbers = self.wl_solution[:,-1]
        new_array = []

        for i in range(len(last_numbers)-1):
            next_row = self.wl_solution[i+1]
            filtered_next_row = next_row[next_row > last_numbers[i]]
            new_array.extend(filtered_next_row)

        # The new array after operation
        new_wl_solution = np.array(new_array)

        # Mask the wavelength solution
        mask = (new_wl_solution >= largest_minima) & (new_wl_solution <= smallest_maxima)
        common_wv = new_wl_solution[mask]

        # Interpolate each star's spectrum and ivar onto the common grid
        for filename in filenames:
            temp_wv, temp_flux, temp_ivar, resampled_flux, resampled_ivar = self.interpolate_spectrum(filename, common_wv)

            # Save the resampled flux and ivar in the new directory
            new_filename_flux = filename.replace(self.cross_correlate_dir_path, 
                                                 self.interpolate_dir_path).replace("wavelength.npy", 
                                                                                    "resampled_flux.npy")
            new_filename_ivar = filename.replace(self.cross_correlate_dir_path, 
                                                 self.interpolate_dir_path).replace("wavelength.npy", 
                                                                                    "resampled_ivar.npy")
            
            np.save(new_filename_flux, resampled_flux)
            np.save(new_filename_ivar, resampled_ivar)
            
            obs_name = os.path.basename(filename).split("_")[0]
            # Plot the resampled and original spectrum
            plt.figure(figsize=(10, 6))
            plt.plot(temp_wv, temp_flux, label='Original Spectrum')
            plt.plot(common_wv, resampled_flux, label='Resampled Spectrum')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.legend()
            plt.title(f"Spectrum for {obs_name}")
            plt.savefig(new_filename_flux.replace("resampled_flux.npy", "comparison.png"))
            plt.close()

        # Save the common wavelength grid in the new directory
        np.save(os.path.join(self.interpolate_dir_path, "interpolated_wl.npy"), common_wv)

        end_time = time.perf_counter()
        logging.info(f"It took CHIP.interpolate, {end_time - start_time} seconds to finish!")


    def load_the_cannon(self):
        ''' Load in the data The Cannon will use.'''
        logging.info("CHIP.load_the_cannon( )") 

        self.random_seed = self.config["Training"]["random seed"]["val"]

        interpolated_dir_path = os.path.join(self.data_dir_path, "inter")

        # Load wavelength solution
        self.wl_solution = np.load( os.path.join( interpolated_dir_path , "wl.npy") )

        # load spectra 
        # Note: the key is hiresid instead of filename 
        hiresid_filenames_array = pd.read_csv( os.path.join( self.data_dir_path,"HIRES_Filename_snr.csv"))[["HIRESID",'FILENAME']].to_numpy()
        for hiresid, filename in hiresid_filenames_array:
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
        self.parameters_df = self.parameters_df[self.parameters_df["HIRESID"].isin( hiresid_filenames_array[:,0] )]

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
        
        Args: 
            mask_path (str) - path to the mask
               
        Returns: 
            (np.array((wl_solution.shape[0],))) - masked boolean array
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

        Args: 
            md (TheCannon.model.CannonModel) - 
            ds (TheCannon.dataset.Dataset) -
            true_labels (np.array((M,))) - true values for the corresponds inferred labels 
        
        Returns:
            (float) - cost function value
        '''
        label_errors = md.infer_labels(ds)
        inferred_labels = ds.test_label_vals

        if save:
            # Save inferred labels
            joblib.dump(inferred_labels, os.path.join(self.storage_path,'inferred_labels.joblib'))


        return self.cost_function(true_labels, inferred_labels)
        

    def split_data(self, X_indecies, y_indecies):
        ''' Split data to be put in TheCannon.dataset.Dataset

        Args: 
            X_indecies (np.array((N,))) - indecies to be used for the training set
            y_indecies (np.array((M,))) - indecies to be used for the testing set
        
        Returns: 
            training id, training spectra, training ivar, training parameters, testing id, testing spectra, testing ivar, testing parameters
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

        Args: 
            batch_size (int) : number of batches to split the training set into for mini-batch training
            poly_order (int) : A positive int, tells the model what degree polynomial to fit
            mask_name (str) : name of the mask to use
            test_set (bool) : If True, the model will be evaluated on the test set, instead of the validation set
        
        Returns: 
            (float) The mean evaluation score 
        '''
        logging.info(f"train_model(batch_size = {batch_size}, poly_order={poly_order}, mask_name={mask_name}, test_set={test_set})")
        
        def mini_batch(cannon_model,batch_size,X_id, X_spec, X_ivar, X_param, y_id, y_spec, y_ivar):
            ''' Train a Cannon model using mini-batch

            Args: 
                cannon_model (TheCannon.model.CannonModel) - A Cannon model
                batch_size (int) : number of batches to split the training set into for mini-batch training
                X_id (np.array((M,))) - HIRES identifers that correspond to the rows in X_flux, X_ivar, and X_parameter
                X_flux (np.array((M,N))) - flux for each star in X_id 
                X_ivar (np.array((M,N))) - ivar for each star in X_id 
                X_parameter (np.array((M,P))) - contains all the parameters for each star in X_id (in the exact same order)
                y_id (np.array((V,))) - HIRES identifers that correspond to the rows in y_flux, y_ivar
                y_flux (np.array((V,N))) - flux for each star in y_id 
                y_ivar (np.array((V,N))) - ivar for each star in y_id 
            
            Returns: 
                (TheCannon.model.CannonModel) - A trained Cannon model

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

            Args:  
                X_spec (np.array((M,N))) - flux for each star in X_id
                X_ivar (np.array((M,N))) - ivar for each star in X_id
                y_spec (np.array((V,N))) - flux for each star in y_id
                y_ivar (np.array((V,N))) - ivar for each star in y_id

            Returns: 
                X_spec (np.array((M,N))) - mask applied flux for each star in X_id
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

        Args: 
            poly_order (int) : A positive int, tells the model what degree polynomial to fit

        Returns: 
            TheCannon.model.CannonModel object 
        '''
        md = model.CannonModel( order = poly_order, useErrors=False )  
        return md 


    @staticmethod
    def initailize_dataset(wl_sol, X_id, X_flux, X_ivar, X_parameter, y_id, y_flux, y_ivar, parameters_names ):
        ''' Put data into data structure The Cannon will train on.

        Args: 
            wl_sol (np.array((N,))) - wavelength solution for all spectra
            X_id (np.array((M,))) - HIRES identifers that correspond to the rows in X_flux, X_ivar, and X_parameter
            X_flux (np.array((M,N))) - flux for each star in X_id 
            X_ivar (np.array((M,N))) - ivar for each star in X_id 
            X_parameter (np.array((M,P))) - contains all the parameters for each star in X_id (in the exact same order)
            y_id (np.array((V,))) - HIRES identifers that correspond to the rows in y_flux, y_ivar
            y_flux (np.array((V,N))) - flux for each star in y_id 
            y_ivar (np.array((V,N))) - ivar for each star in y_id 
            parameters_names (np.array((P,))) - column names of X_parameter and y_parameter
              
        Returns: 
            (TheCannon.dataset.Dataset) containing the data in the correct format for The Cannon 
        '''

        ds = dataset.Dataset(wl_sol, X_id, X_flux, X_ivar, X_parameter, y_id, y_flux, y_ivar) 
        ds.set_label_names(parameters_names) 
        # wl ranges can be optimized for specific echelle ranges
        ds.ranges= [[np.min(wl_sol),np.max(wl_sol)]]
        return ds


    def cannon_splits(self, parameters_df):
        ''' Apply test and validation splits to the data. This must be ran after self.parameters_df is created.
        
        Args: 
            parameters_df (pd.DataFrame) : contains all the parameters for each star in X_id (in the exact same order)
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
        ''' Tune the hyperparameters of The Cannon model. This must be ran after self.cannon_splits is ran.'''
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

        Args: 
            batch_size (int) the 
            poly_order (int)
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

    chip = CHIP(config_file_path="config/config.json")
    chip.run()

    # Move logging file to the location of this current run
    log_filename = os.path.basename(log_filepath)
    # Shutdown logging so the file can be put in the storage location
    logging.shutdown()

    os.rename( log_filepath, 
                os.path.join( chip.storage_path, log_filename) )