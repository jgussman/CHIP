from astropy import units as u
from astropy.io import fits
from hiresprv.auth import login
from hiresprv.idldriver import Idldriver
from hiresprv.database import Database
from hiresprv.download import Download
from PyAstronomy import pyasl
from specutils.fitting import fit_generic_continuum
from specutils.spectra import Spectrum1D

import numpy as np
import pandas as pd 

class CIR:

    def __init__(self,star_ID_array,wl_solution_path = './wl_solution.csv',rvOutputPath='./RVOutput',spectraOutputPath = './SpectraOutput' ):
        '''
        star_ID_array: is an array that contains strings of the HIRES ID for stars
        wl_solution_path: defaulted to ./wl_solution.csv, the file at the end of your given path
                          needs to have the first 2 rows not contain data. skip_header is set to 2
        rvOutputPath: defaulted to ./RVOutput, the folder at the end of your path needs to have already
                      been created before you initalize this class
        spectraOutputPath: defaulted to ./SpectraOutput, the folder at the end of your path needs to have 
                            already been created before you initalize this class
        '''

        login('prv.cookies')                                               # For logging into the NExSci servers 
        self.idldriver = Idldriver('prv.cookies')                          # For creating RV scripts 
        self.state = Database('prv.cookies')                               # For retrieving data from HIRES 
        self.dataRV = Download('prv.cookies', rvOutputPath)                # For downloading RV
        self.dataSpectra = Download('prv.cookies',spectraOutputPath)       # For downloading Spectra 
        self.star_ID_array = star_ID_array                                 # HIRES ID 
        self.wl_solution = np.genfromtxt(wl_solution_path,skip_header=2)   # UNITS: Angstrom
        
    
    def Run(self):
        '''
        Runs Find_and_download_all_rv_obs -> DownloadSpectra -> ContinuumNormalize -> CrossCorrelate -> Interpolate 
        '''
        self.Find_and_download_all_rv_obs()
        self.DownloadSpectra()
        self.ContinuumNormalize()
        self.CrossCorrelate()
        self.Interpolate()


    def Find_and_download_all_rv_obs(self):
        '''            
        Description: This function downloads the rotational velocity metadata and
        returns a dictionary that makes it easy to identify what stars' max observed 
        rotational velocities as well as the filenames for which they came from.

        Note: The dataframe produced by this metho will remove all the stars that did 
              not have any RV data produced by run_script. This does not mean that 
              the star doesn't have any RV data. It could mean a few things, the ID
              is wrong, contains a character that isn't currently supported by the 
              HIRES pipeline, it could have too little RV observations to make an 
              RV curve, etc...
        
        '''
        rv_script_name_list = []
        problem_child_name = []
        problem_child_rv = []
        problem_child_filename = []
        master_script = ""
        for HIRESname in self.star_ID_array:
            try:
                    #Create script for reducing RV observations
                    temp_rv_script = self.idldriver.create_rvscript(HIRESname,self.state) 

                    length_of_name = len(HIRESname)
                    first_date = temp_rv_script[3+length_of_name:13+length_of_name].split(".")[0]

                    HIRESrvname = temp_rv_script[3:length_of_name+3].split(" ")[0]

                    temp_rv_script =  "template {0} {1}\n".format(HIRESrvname,first_date) + temp_rv_script
                    temp_rv_script += "\nrvcurve {0}\n".format(HIRESrvname)
                    rv_script_name_list.append(HIRESrvname)
                    master_script += temp_rv_script

            except AttributeError: #This is due to the idldriver.create_rvscript 
                problem_child_name = [HIRESname] + problem_child_name
                problem_child_rv += [pd.NA]
                problem_child_filename += [pd.NA]
            
        #Run script 
        self.idldriver.run_script(master_script) 
        
        #Downloading the RV data as well as getting the largest RV value for each star
        largest_rv = {"HIRESName": problem_child_name,"FILENAME":problem_child_filename, "RV":problem_child_rv}  
        localdir = data.localdir
        for name in rv_script_name_list:
            #Make sure the data is in workspace
            largest_rv["HIRESName"].append(name)
            try:
                rtn = data.rvcurve(name)
                nameLoc = '{0}/vst{1}.csv'.format(localdir,name)
                temp_df = pd.read_csv(nameLoc)
                if not temp_df.empty:
                    rv_temp = abs(temp_df['RV'])
                    row = temp_df[temp_df['RV'] == rv_temp.min()]
                    if row.empty: #The absolute max rv is negative 
                        row = temp_df[temp_df['RV'] == -rv_temp.min()]
                    largest_rv["RV"] += [row["RV"].to_numpy()[0]]
                    largest_rv["FILENAME"] += [row["FILENAME"].to_numpy()[0]]
                else:
                    largest_rv["RV"] += [pd.NA]
                    largest_rv["FILENAME"] += [pd.NA]
            except OSError: #This error occurs because for some reason the star's rvcurve wasn't created
                    largest_rv["RV"] += [pd.NA]
                    largest_rv["FILENAME"] += [pd.NA]
        df = pd.DataFrame(largest_rv)
        df.dropna()   #If you don't drop the na's then other methods will break
        self.name_filename_rv_df = df
        self.name_filename_rv_df.to_csv("HIRES_Filename_rv.csv",index_label=False,index=False)
        

    def DownloadSpectra(self):
        '''
        Description: This method downloads all the deblazed spectra from HIRES that are in self.filename_rv_df["FILENAME"]
                     to self.dataSpectra.localdir
        '''
        self.spectraDic = {} 
        download_Location = self.dataSpectra.localdir #This is the second parameter of hiresprv.download.Download
        for filename in self.filename_rv_df["FILENAME"]:
            #I tried to use the , seperation and new line seperation 
            #for the different file names but it doesn't seem to work.
            #Thus, a for-loop was used!
            self.dataSpectra.spectrum(filename.replace("r",""))  #Download spectra 
            
            temp_deblazedFlux = fits.getdata("{0}/{1}.fits".format(download_Location,filename))
            self.spectraDic[filename] = np.append(temp_deblazedFlux[0],[temp_deblazedFlux[i] for i in range(1,16)])     
        
        self.spectraDic = spectraDic    

    def ContinuumNormalize(self):
        '''        
        Description: This method uses specutils' Spectrum1D function to fit a function
                     to each echelle order spectra then subtracts the function from the 
                     echelle order. 
                     The normalized spectra are put in the {} called self.spectraDic with
                     the keys being the HIRES filename and the values being a continuum 
                     normalized 1-D array.   
        '''
        #This is the same for all HIRES data 
    
        spectral_axis_wl_solution = self.wl_solution*u.um

        download_Location = self.dataSpectra.localdir 
        #Continumm Normalize
        for filename in self.spectraDic:
            deblazedFlux = self.spectraDic[filename]*u.Jy
            normalized_array = np.array([])
            for j in range(0,16): #Normalize each individual echelle order 
                i  = 4021 * j
                temp_echelle_flux = deblazedFlux[i:i+4021]
                temp_echelle_wl = spectral_axis_wl_solution[i:i+4021]
                spectrum = Spectrum1D(flux=temp_echelle_flux, spectral_axis=temp_echelle_wl )
                g1_fit = fit_generic_continuum(spectrum)
                flux_fit = g1_fit(temp_echelle_wl)

                normalized_echelle = temp_echelle_flux / flux_fit
                #Converting to a float like this removes 2 decimal places from normalized_echelle
                normalized_echelle = np.array(list(map(np.float,normalized_echelle)))  
                #Make all the echelle spectra into a 1-D array again
                normalized_array = np.append(normalized_array,normalized_echelle)
                
            self.spectraDic[filename] = normalized_array
            
        self.spectraDic
    

    def CrossCorrelate(self,numOfEdgesToSkip = 25):
        '''
        Description: Uses Pyastronomy's crosscorrRV function to compute the cross correlation.

                     The updated spectra are put into self.spectraDic.   
        '''
        wvnum, wvlen, crf, tel, c, n = np.genfromtxt("../Atlases/solarAtlas.txt",skip_header=1,unpack=True)
        wvnum, wvlen, crf, tel, c, n = wvnum[::-1], wvlen[::-1], crf[::-1], tel[::-1], c[::-1], n[::-1] 
        
        wl_solution = np.genfromtxt('./wl_solution.csv',skip_header=2) #UNITS: Angstrom
        
        crossCorrelatedspectra = {} #Key: FILENAME Values: (correlated wavelength, normalized flux)
        for i in range(self.filename_rv_df.shape[0]):
            row = self.filename_rv_df.iloc[i]
            filename = row[1]
            RV = abs(row[2])
            
            normalizedFlux = self.spectraDic[filename]
            
            rv, cc = pyasl.crosscorrRV(wl_solution, normalizedFlux, wvlen,c, -1*RV, RV, RV/100., skipedge=numOfEdgesToSkip)
            maxind = np.argmax(cc)
            argRV = rv[maxind]  #UNITS: km/s 
            
            # z = v_0/c    
            z = (argRV/299_792.458) #UNITS: None 
            computeShiftedWavelength = lambda wl: wl + wl*z #UNITS: Angstroms 
            shifted_wl = np.array(list(map(computeShiftedWavelength,wl_solution)))
            
            #Making the key the HIRES ID so I easily convert it back to the Spoc ID later
            crossCorrelatedspectra[row[0]] = (shifted_wl,normalizedFlux) 
        
        self.spectraDic  = crossCorrelatedspectra

    def Interpolate(self):
        '''        
        Description: This method downloads the interpolated wavelength to interpolated_wl.csv 
                     and downloads the fluxes to fluxes_for_HIRES.csv. 
        '''
        
        #Interpolate the spectra with each other to get the same wavelength scale for all of them.
        firstKey = next(iter(self.spectraDic))
        first_spectra = self.spectraDic[firstKey][0]
        wl_length = len(first_spectra)
        
        
        maxMinVal = float('-inf')
        minMaxVal = float('inf')
        #Finds the max minimum wavelength val & finds the min maximum wavelenght val 
        for spectra_flux_tuple in self.spectraDic.values(): 
            #Assumption: wavelength is sorted from the 0th index being min,
            #            the len(wavelength array)-1 is the max wavelength val,
            #            all the wavelength arrays are the same length.
            temp_spectra = spectra_flux_tuple[0]
            temp_min_wl = temp_spectra[0]
            temp_max_wl = temp_spectra[wl_length-1]
            
            if maxMinVal < temp_min_wl:
                maxMinVal = temp_min_wl
            if minMaxVal > temp_max_wl:
                minMaxVal = temp_max_wl
        
        #wavelength range 
        interpolate_over = [wl for wl in first_spectra if wl > maxMinVal and wl<=minMaxVal]    #I think this is where the 1 off wl array error was coming from, fixed now.
        
        fluxDic = {}
        for HIRES_ID in self.spectraDic:
            wl = self.spectraDic[HIRES_ID][0]
            flux_norm = self.spectraDic[HIRES_ID][1]
            interpolated_flux = np.interp(interpolate_over,x,flux_norm)
            fluxDic[HIRES_ID] = interpolated_flux
        
        #Saving 
        np.savetxt("interpolated_wl.csv",interpolate_over,delimiter=",")
        fluxDF = pd.DataFrame(fluxDic)
        fluxDF.to_csv("fluxes_for_HIRES.csv",index_label=False,index=False)
        self.fluxDF  = fluxDF
        self.interpolate_wl = interpolate_over

    def IvarCalculation(self):
        '''

        '''
        pass

if __name__ == '__main__':
    crossMatchedNames = pd.read_csv("../spocData/starnames_crossmatch_SPOCS_NEXSCI.txt",sep=" ")
    cirObject = CIR(crossMatchedNames,'./wl_solution.csv','./RVOutput','./SpectraOutput')
    cirObject.Run()