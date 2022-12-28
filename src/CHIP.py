import logging
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
from PyAstronomy import pyasl
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from TheCannon import model




class CHIP:
    chip_version = "v1.0.0"
    thecannon_version = "v0.5.0"

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
        if self.config["CHIP"]["run"]["val"]:
            # Create a new storage location for this pipeline
            # Greenwich Mean Time (UTC) right now 
            gmt_datetime = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

            self.storage_path = os.path.join("data","chip_runs" , gmt_datetime )

            # login into the NExSci servers
            login('data/prv.cookies')

        else:
            # What chip run is going to be used
            chip_run_subdir = self.config["The Cannon"]["run"]["val"]
            self.data_dir_path = os.path.join("data/chip_runs",chip_run_subdir)

            if not(os.path.exists(self.data_dir_path)):
                logging.error(f'Your inputed sub dir name for ["The Cannon"]["run"]["val"] does not exist. {self.data_dir_path}')
            
            # Make The Cannon Results dir 
            # semi-unique subdir naming convention
            # random seed _ testing fraction _ validation fraction _ cost function name (spaces filled with -)
            rand_seed = self.config["The Cannon"]["random seed"]["val"]
            test_frac = self.config["The Cannon"]["train test split"]["val"]
            cost_fun  = self.config["The Cannon"]["cost function"]["name"].replace(" ", "-")
            semi_unique_subdir = f"{rand_seed}_{test_frac}_{cost_fun}"
            
            self.storage_path = os.path.join( self.data_dir_path, "cannon_results", semi_unique_subdir )
            
        os.makedirs( self.storage_path, exist_ok= True )

        
    def run(self):
        ''' Run the pipeline from end to end. 

        Inputs: None

        Outputs: None
        '''
        
        if self.config["CHIP"]["run"]["val"]:
            logging.info(f"CHIP {self.chip_version}")

            self.download_spectra()

            self.alpha_normalization()

            self.cross_correlate_spectra()

            self.interpolate()
        
        elif self.config["The Cannon"]["run"]["val"]:
            logging.info(f"The Cannon {self.thecannon_version}")

            self.load_the_cannon()

        else:
            logging.error("Neither CHIP or The Cannon were selected to run in config.json! Please select!")


    def get_arguments(self):
        ''' Get arguments from src/config.json and store in self.config 

        Inputs: None

        Outputs: None
        '''
        logging.debug("CHIP.get_arguments( )")

        with open("src/config.json", "r") as f:
            self.config = json.load(f)
        
        self.cores = self.config["CHIP"]["cores"]["val"]
        self.trim  = self.config["CHIP"]["trim_spectrum"]["val"]
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
                return temp_deblazedFlux[:, self.trim: -self.trim]
    

    def update_removedstars(self):
        ''' Helper method to insure removed_stars will be updated properly accross each method.
        
        Input: None

        Output: None
        '''
        pd.DataFrame(self.removed_stars).to_csv( os.path.join(self.storage_path ,"removed_stars.csv"),
                                                 index_label=False,
                                                 index=False)
        
    def download_spectra(self):
        ''' Downloads all the spectra for each star in the NExSci, calculates 
        the SNR and saves the spectrum with the highest SNR for each star.

        Inputs: None

        Outputs: None
        '''
        logging.info("CHIP.download_spectra( )")

        start_time = time.perf_counter()

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
        
        self.update_removedstars()

        # Delete unused instance attributes
        del self.dataSpectra
        del self.state

        end_time = time.perf_counter()
        logging.info(f"It took CHIP.download_spectra, {end_time - start_time} to finish!")
    

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
        os.mkdir( self.norm_spectra_dir_path )

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
        logging.info(f"It took CHIP.alpha_normalization, {end_time - start_time} to finish!")


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
        solar = np.load('data/constants/solarAtlas.npy')
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
        logging.info(f"It took CHIP.cross_correlate_spectra, {end_time - start_time} to finish!")


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
        logging.info(f"It took CHIP.interpolate, {end_time - start_time} to finish!")

    def load_the_cannon(self):
        ''' Load in the data The Cannon will use. 
        
        Input: None

        Output: None
        '''
        logging.info("CHIP.load_the_cannon( )") 

        self.random_seed = self.config["The Cannon"]["random seed"]["val"]

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
        parameters_list = self.config["The Cannon"]["stellar parameters"]["val"]
        hiresid_parameters_list = ["HIRESID"] + parameters_list
        self.parameters_df = pd.read_csv("data/spocs/stellar_parameters.csv")[ hiresid_parameters_list ]
        # Extract only the stars that were preprocessed 
        self.parameters_df = self.parameters_df[self.parameters_df["HIRESID"].isin( cross_match_array[:,0] )]

        logging.info(self.parameters_df)
        # Create a StandardScaler object
        self.parameters_scaler = StandardScaler()
        # Fit the scaler to the selected columns and transform the selected columns
        parameters_transformed = self.parameters_scaler.fit_transform( self.parameters_df[parameters_list] )
        # change values in df 
        self.parameters_df[parameters_list] = parameters_transformed
        
        logging.info( "scaled features\n" + self.parameters_df.to_string() )

        self.cannon_splits(self.parameters_df)


        # TODO: # Load masks 
        # self.masks = self.config["The Cannon"]["masks"]["val"]
        # for i in range(len(self.masks)):
        #     mask_name = self.masks[i][0]
        #     mask_path = self.masks[i][1]
        #     if os.path.exists(mask_path):
        #         wl,mask = np.load(mask_path,unpack=True)
        #         trimming_mask = np.isin(self.wl_solution, wl)



    def evaluate_model(self,X,y):
        '''
        
        '''
        pass

    def train_model(self, num_folds,batch_size, poly_order ):
        '''

        Input: num_folds (int) : number of folds for kfold cross validation
               batch_size (int) : number of batches to split the training set into for mini-batch training
               poly_order (int) : A positive int, tells the model what degree polynomial to fit
        
        Output: None
        '''
        # Initialize The Cannon model 
        cannon_model = self.initialize_model(poly_order = poly_order)
        
        
    @staticmethod
    def initialize_model(poly_order = 1):
        ''' Initialize The Cannon Model

        Input: poly_order (int) : A positive int, tells the model what degree polynomial to fit

        Output: TheCannon.model.CannonModel object 
        '''

        model = model.CannonModel( order = poly_order, useErrors=False )
        
        return model 



    def cannon_splits(self, parameters_df):
        ''' Apply test and validation splits to the data. This must be ran after self.parameters_df is created.
        
        Input: None

        Output: None
        '''
        logging.debug("CHIP.cannon_splits( )")

        # Split the parameters into a training set, a test set, and a validation set
        test_frac = self.config["The Cannon"]["train test split"]["val"]
        self.parameter_train, self.parameter_test = train_test_split(parameters_df, test_size = test_frac, random_state= self.random_seed)

        # Split the spectra and ivars 
        stars_in_test = [name for name in self.parameter_test["HIRESID"]]
        stars_in_train = [name for name in self.parameter_train["HIRESID"]]

        self.test_spectra = np.vstack([ self.spectraDic[name] for name in  stars_in_test ])
        self.test_ivar = np.vstack([ self.ivarDic[name] for name in  stars_in_test ])
        self.train_spectra = np.vstack([ self.spectraDic[name] for name in  stars_in_train ])
        self.train_ivar = np.vstack([ self.ivarDic[name] for name in  stars_in_train ])

        # No longer needed 
        del self.spectraDic
        del self.ivarDic

        # Create a KFold object for preforming k-fold cross validation  
        num_folds = self.config["The Cannon"]["kfolds"]["val"]
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.random_seed)
        self.kfold_train_validation_splits = np.vstack(kf.split(self.test_spectra))
        logging.info("kflod splits" + str(self.kfold_train_validation_splits))



        


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

    try:
           
        pass

    except Exception as e:
        logging.error(e) 

    finally:
        # Move logging file to the location of this current run
        log_filename = os.path.basename(log_filepath)
        # Shutdown logging so the file can be put in the storage location
        logging.shutdown()

        # TODO: Undo Comments below
        # os.rename( log_filepath, 
        #           os.path.join( chip.storage_path ,log_filename) )


    







