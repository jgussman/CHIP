from hiresprv.auth import login
import logging
import json


import numpy as np
import pandas as pd



class CHIP:

    def __init__(self):
        '''
        
        '''

        logging.info(f"Instantiated : {self.__class__}")

        # Get arguments from config.json
        self.get_arguments()

    def run(self):
        '''Run the pipeline from end to end. 

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
        # TODO: Refactor Downloading code

        # Cross matched names 
        cross_matched_file_path = self.config["CHIP"]["cross_match_stars"]["val"]
        cross_matched_df = pd.read_csv( cross_matched_file_path, sep=" " )
        hires_names_array = cross_matched_df["HIRES"].to_numpy()

         
        pass 
        






if __name__ == "__main__":
    
    # logging 
    logging.basicConfig(filename='src/CHIP.log',
                        format='%(asctime)s - %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',  
                        level=logging.INFO)

    logging.info(f"Running file: {__file__}")

    # logging into the NExSci servers
    login('prv.cookies')

    # Instantiation
    chip = CHIP()

    







