from astropy.io import fits
from hiresprv.auth import login

import logging
import json
import os
import pandas as pd
import time 



class CHIP:

    def __init__(self):
        '''
        
        '''

        logging.info(f"Instantiated : {self.__class__}")

        # Get arguments from config.json
        self.get_arguments()

        # Create a new storage location for this pipeline
        self.create_storage_location()
    
    
    def create_storage_location(self):
        ''' Creates a unique subdirectory in the data directory to store the outputs of CHIP
        
        Inputs: None

        Outputs: None 
        '''
        logging.info("create_storage_location()")

        # Greenwich Mean Time (UTC) right now 
        gmt_datetime = time.strftime("%Y-%m-%d_%H-%M",time.gmtime())

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

        logging.info("get_arguments()")

        with open("src/config.json", "r") as f:
            self.config = json.load(f)
        
        logging.info( f"config.json : {self.config}" )

        
    def download_spectra(self):
        '''

        Inputs: None

        Outputs: None
        '''

        # Cross matched names 
        cross_matched_file_path = self.config["CHIP"]["cross_match_stars"]["val"]
        cross_matched_df = pd.read_csv( cross_matched_file_path, sep=" " )
        hires_names_array = cross_matched_df["HIRES"].to_numpy()

        removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], "SNR < 100":[],"NAN value in spectra":[],"No Clue":[]}
        hiresName_fileName_snr_dic = {"HIRESName": [],"FILENAME":[],"SNR":[]}  

        
        for star_ID in hires_names_array:
            logging.info(f"Finding highest SNR spectrum for {star_ID}")
                
            try:
                # SQL query to find all the RV observations of one particular star
                search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"
                url = self.state.search(sql=search_string)
                obs_df = pd.read_html(url, header=0)[0]
                # There are no RV Observations
                if obs_df.empty: 
                    logging.info(f"{star_ID} has no RV observations")
                    removed_stars["no RV observations"].append(star_ID) 
                    continue 
                else:   

                    best_SNR = 0
                    best_SNR_filename = False 
                    for filename in obs_df["FILENAME"]:
                        temp_SNR = self.DownloadSpectrum(filename)
                        
                        # Check if the SNR is the highest out of 
                        # all the star's previous spectras 
                        if best_SNR < temp_SNR:
                            best_SNR = temp_SNR

                            best_SNR_filename = filename 

                    if best_SNR < 100: 
                        logging.info(f"{star_ID}'s best spectrum had an SNR lower than 100. Thus it was removed.")

                        removed_stars["SNR < 100"].append(star_ID)

                    else:
                        # Save the Best Spectrum
                        self.spectraDic[best_SNR_filename] = self.DownloadSpectrum(best_SNR_filename, 
                                                                                SNR=False)
                        

                        hiresName_fileName_snr_dic["HIRESName"].append(star_ID)
                        hiresName_fileName_snr_dic["FILENAME"].append(best_SNR_filename)
                        hiresName_fileName_snr_dic["SNR"].append(best_SNR)
                    
                        # Calculate ivar
                        self.SigmaCalculation(best_SNR_filename)

            except:
                logging.info(f"{star_ID} was removed for an undefined reason.")
                removed_stars["No Clue"].append(star_ID)
            finally:
                continue

        self.filename_df = pd.DataFrame(hiresName_fileName_snr_dic) 

        os.mkdir()
        self.filename_df.to_csv("HIRES_Filename_snr.csv",index_label=False,index=False)
        
        print(removed_stars)
        print("Downloading Spectra Has Finished")
        


        def download_spectrum(self,filename,SNR = True):
            '''Download Individual Spectrum and ivar 

            Input: filename (str): name of spectrum you want to download
                   SNR (bool): if you want to calculate the SNR of the spectrum 

            Output: None
            '''
            logging.info(f"download_spectrum(filename={filename},SNR={SNR})")
            
            #Download spectra
            self.dataSpectra.spectrum(filename.replace("r",""))    
            file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")

            try: 
                temp_deblazedFlux = fits.getdata(file_path)
            except OSError:
                # There is a problem with the downloaded fits file 
                return -1
            
            
            if SNR: # Used to find best SNR 
                # Delete Spectrum to save space
                SNR = self.calculate_SNR(temp_deblazedFlux)
                del temp_deblazedFlux 
                self.DeleteSpectrum(filename)
                return SNR

            else:
                return temp_deblazedFlux
        
        






if __name__ == "__main__":
    
    # logging 
    logging.basicConfig(filename='src/CHIP.log',
                        format='%(asctime)s - %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',  
                        level=logging.INFO)
    # logs to file and stdout
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(f"Running file: {__file__}")

    # logging into the NExSci servers
    login('prv.cookies')

    # Instantiation
    chip = CHIP()

    







