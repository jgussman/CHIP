import json
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
import unittest
 
from src.CHIP import CHIP
from hiresprv.auth import login



class TestCHIP(unittest.TestCase):

    CHIP = CHIP()


    def test_get_arguments(self):
        ''' Test CHIP's get_arguments method

        Input: None

        Output: None
        '''
        # src/config.json should be the same json file CHIP is reading from
        with open("src/config.json", "r") as f:
            test_config = json.load(f)

        # Insure the config file is being read in correctly. 
        for test_config_key in test_config:
            self.assertTrue( test_config_key in self.CHIP.config)
            self.assertTrue( test_config[test_config_key] == self.CHIP.config[test_config_key])
             
    def test_download_spectra(self):
        ''' Test CHIP's download_spectra method

        Input: None

        Output: None
        '''

        # login into the NExSci servers
        login('data/prv.cookies')

        # Change the config attribute of CHIP 
        # so it points to a smaller dataset
        with open("src/tests/config_test.json", "r") as f:
            test_config = json.load(f)

        self.CHIP.config = test_config

        self.CHIP.download_spectra()

        # Insure download_spectra downloaded the best spectra
        # Get all the filenames that should exist  
        files_that_should_exist = [ file_name for dir_path, dir_names, file_names in os.walk("src/tests/test_comparison_files") for file_name in file_names]
        files_that_do_exist = [ file_name for dir_path, dir_names, file_names in os.walk( self.CHIP.storage_path ) for file_name in file_names]
        
        self.assertEquals( len(files_that_should_exist), len(files_that_do_exist) )
        
        for file_should_exist in files_that_do_exist:
            self.assertTrue( file_should_exist in files_that_do_exist )

        # Check HIRES_Filename_snr.csv is correct 
        comparison_df = pd.read_csv(  "src/tests/test_comparison_files/HIRES_Filename_snr.csv" ) 
        new_df = pd.read_csv( os.path.join( self.CHIP.storage_path,  "HIRES_Filename_snr.csv" ) )
        
        self.assertTrue( ((comparison_df == new_df ).all()).all() )


        # spectraDic has correct shape and no nans 
        for spectrum in self.CHIP.spectraDic.values(): 
            self.assertEqual( spectrum.shape, (16, 4021) )
            self.assertFalse( np.isnan(spectrum).any() )
        
        # ivarDic has correct shape and no nans 
        for ivar in self.CHIP.ivarDic.values(): 
            self.assertEqual( ivar.shape, (16, 4021) )
            self.assertFalse( np.isnan(ivar).any() )

        



    
    def clean_up( self ):
        '''At the end of each testinf get rid of files in self.CHIP.storage_path

        Input: None 

        Output: None 
        '''
        os.remove( self.CHIP.storage_path )


if __name__ == "__main__":
    unittest.main()