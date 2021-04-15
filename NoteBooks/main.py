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

class CIR:

    def __init__(self,star_ID_array,wl_solution_path = './wl_solution.csv',rvOutputPath='./RVOutput',spectraOutputPath = './SpectraOutput' ):
        '''
        Used for reducing HIRES data to get it ready for being fed into The Cannon. 

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
        self.Ivar = {}                                                     # For storing sigma valeus 
        self.filename_rv_df = pd.DataFrame()                               # For storing meta Data
        
    
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
        returns a dictionary that makes it easy to identify what stars' rotational 
        velocities nearest 0 as well as the filenames for which they came from.

        Note: The dataframe produced by this method will remove all the stars that did 
              not have any RV data produced by rvcurve. This does not mean that 
              the star doesn't have any RV data. It could mean a few things, the ID
              is wrong, contains a character that isn't currently supported by the 
              HIRES pipeline, it could have too few RV observations to make an 
              RV curve, etc...
        '''
        print("Downloading RV Data Has Began")
        #Downloading the RV data as well as getting the largest RV value for each star
        hiresName_fileName_rv_dic = {"HIRESName": [],"FILENAME":[], "RV":[]}  
        rvDownloadLocation = self.dataRV.localdir
        for name in self.star_ID_array:
            #Make sure the data is in workspace
            hiresName_fileName_rv_dic["HIRESName"].append(name)
            try:
                rtn = self.dataRV.rvcurve(name)
                nameLoc = '{0}/vst{1}.csv'.format(rvDownloadLocation,name)
                temp_df = pd.read_csv(nameLoc)
                if not temp_df.empty:
                    rv_temp = abs(temp_df['RV'])
                    row = temp_df[temp_df['RV'] == rv_temp.max()]
                    if row.empty: #The absolute max rv is negative 
                        row = temp_df[temp_df['RV'] == -rv_temp.max()]
                    hiresName_fileName_rv_dic["RV"] += [row["RV"].to_numpy()[0]]
                    hiresName_fileName_rv_dic["FILENAME"] += [row["FILENAME"].to_numpy()[0]]
                else:
                    hiresName_fileName_rv_dic["RV"] += [pd.NA]
                    hiresName_fileName_rv_dic["FILENAME"] += [pd.NA]
            except OSError: #This error occurs because for some reason the star's rvcurve wasn't created
                    hiresName_fileName_rv_dic["RV"] += [pd.NA]
                    hiresName_fileName_rv_dic["FILENAME"] += [pd.NA]
        df = pd.DataFrame(hiresName_fileName_rv_dic) 
        self.filename_rv_df = df.dropna() #If you don't drop the na's then other methods will break
        self.filename_rv_df.to_csv("HIRES_Filename_rv.csv",index_label=False,index=False)
        print("Downloading RV Data Has Ended")
        

    def DownloadSpectra(self):
        '''
        Description: This method downloads all the deblazed spectra from HIRES that are in self.filename_rv_df["FILENAME"]
                     to self.dataSpectra.localdir
        '''
        print("Downloading Spectra Has Began")
        self.spectraDic = {} 
        download_Location = self.dataSpectra.localdir #This is the second parameter of hiresprv.download.Download
        for filename in self.filename_rv_df["FILENAME"]:
            #I tried to use the , seperation and new line seperation 
            #for the different file names but it doesn't seem to work.
            #Thus, a for-loop was used!
            self.dataSpectra.spectrum(filename.replace("r",""))  #Download spectra 
            file_path = "{0}/{1}.fits".format(download_Location,filename)
            temp_deblazedFlux = fits.getdata(file_path)
            self.spectraDic[filename] = np.append(temp_deblazedFlux[0],[temp_deblazedFlux[i] for i in range(1,16)])     
            self.SigmaCalculation(filename)
        print("Downloading Spectra Has Ended")

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
        print("Continuum Normalization Began")
    
        spectral_axis_wl_solution = self.wl_solution*u.um
        length = len(spectral_axis_wl_solution)
        download_Location = self.dataSpectra.localdir 
        #Continumm Normalize
        for filename in self.spectraDic:
            deblazedFlux = self.spectraDic[filename]*u.Jy
            ivarSigma = self.Ivar[filename]*u.Jy
            normalized_array = np.array([]) #Name should be changed too normalized_spectra
            ivar_array = np.array([])
            for j in range(0,16): #Normalize each individual echelle order 
                i  = 4021 * j
                temp_echelle_flux = deblazedFlux[i:i+4021]
                temp_sigma  =  ivarSigma[i:i+4021]
                temp_echelle_wl = spectral_axis_wl_solution[i:i+4021]
                
                spectrum = Spectrum1D(flux=temp_echelle_flux, spectral_axis=temp_echelle_wl )
                g1_fit = fit_generic_continuum(spectrum)
                flux_fit = g1_fit(temp_echelle_wl)
                
                #Dividing the continuum 
                normalized_echelle = temp_echelle_flux / flux_fit
                normalized_ivar = temp_sigma / flux_fit
                #Converting to a float like this removes 2 decimal places from normalized_echelle
                normalized_echelle = np.array(list(map(np.float,normalized_echelle))) 
                normalized_ivar = np.array(list(map(np.float,normalized_ivar))) 
                #Make all the echelle spectra into a 1-D array again
                
                
                normalized_array = np.append(normalized_array,normalized_echelle)
                ivar_array = np.append(ivar_array,normalized_ivar)
            spec_norm_scaled = normalized_array/np.repeat(np.percentile(normalized_array, 95, axis=0)[ np.newaxis], length, axis=0)
            self.spectraDic[filename] = spec_norm_scaled
            self.Ivar[filename] = ivar_array
        print("Continuum Normalization Ended")
            
    def CrossCorrelate(self,numOfEdgesToSkip = 25):
        '''
        Description: Uses Pyastronomy's crosscorrRV function to compute the cross correlation.

                     The updated spectra are put into self.spectraDic.   

        numOfEdgesToSkip: the amount of pixels to cut off both ends of the spectra.
        '''
        print("Cross Correlate Has Began")
        wvlen,c = np.genfromtxt("../Atlases/solarAtlas.txt",skip_header=1,usecols=(1,4),unpack=True) 
        wvlen, c = wvlen[::-1], c[::-1]
        
        wl_solution = np.genfromtxt('./wl_solution.csv',skip_header=2) #UNITS: Angstrom
        
        crossCorrelatedspectra = {} #Key: FILENAME Values: (correlated wavelength, normalized flux)
        for i in range(self.filename_rv_df.shape[0]):
            row = self.filename_rv_df.iloc[i]
            filename = row[1]
            #RV = abs(row[2])
            #Uncomment above if you want to use the observed RV
            RV = 50 
            
            normalizedFlux = self.spectraDic[filename]
            #We don't need to cross correlate the entire spectra, only a small subset.
            start = 290_000
            stop = 340_500
            begin = 4021*2
            end = 4021*3
            rv, cc = pyasl.crosscorrRV(wl_solution[begin:end], normalizedFlux[begin:end],wvlen[start:stop],c[start:stop], -1*RV, RV, RV/600., skipedge=100)
            maxind = np.argmax(cc)
            argRV = rv[maxind]  #UNITS: km/s 
            
#           #Uncomment Below in-order to plot the correlation
#             plt.plot(rv, cc, 'bp-')
#             plt.title(f"Max RV of {argRV} km/s with a shift {argRV/299_792.458}")
#             plt.plot(rv[maxind], cc[maxind], 'ro')
#             plt.show()
            # z = v_0/c    
            z = (argRV/299_792.458) #UNITS: None 
            computeShiftedWavelength = lambda wl: wl/ (1 + z)  #UNITS: Angstroms 
            shifted_wl = np.array(list(map(computeShiftedWavelength,wl_solution)))
            
            self.spectraDic[filename] = (shifted_wl,normalizedFlux)     
        print("Cross Correlate Has Ended")

    def Interpolate(self):
        '''        
        Description: This method downloads the interpolated wavelength to interpolated_wl.csv 
                     and downloads the fluxes to fluxes_for_HIRES.csv. 
        '''
        print("Interpolation Has Began")
        #Interpolate the spectra with each other to get the same wavelength scale for all of them.
        maxMinVal = float('-inf')  
        minMaxVal = float('inf')
        #Finds the max minimum wavelength val & finds the min maximum wavelenght val 
        for spectra_flux_tuple in self.spectraDic.values(): 
            #Assumption: wavelength is sorted from the 0th index being min,
            #            the len(wavelength array)-1 is the max wavelength val,
            #            all the wavelength arrays are the same length.
            temp_spectra = spectra_flux_tuple[0]
            temp_min_wl = temp_spectra[0]
            temp_max_wl = temp_spectra[-1]
            
            if maxMinVal < temp_min_wl:
                maxMinVal = temp_min_wl
            if minMaxVal > temp_max_wl:
                minMaxVal = temp_max_wl
        
        #wavelength range 
        #Here I am assuming that all of the spectra increment their wavelengths by the same amount
        firstKey = next(iter(self.spectraDic))
        first_spectra = self.spectraDic[firstKey][0]
        interpolate_over = [wl for wl in first_spectra if wl >= maxMinVal and wl<= minMaxVal]    #I think this is where the 1 off wl array error was coming from, fixed now. ****DELETE IF COMMENT IF NOT VALID ANYMORE**
        length_interpolate = len(interpolate_over)
        #Actual interpolation, using np.interp 
        fluxDic = {}
        for HIRES_ID in self.spectraDic:
            wl = self.spectraDic[HIRES_ID][0]
            flux_norm = self.spectraDic[HIRES_ID][1]
            flux_func = interpolate.interp1d(wl, flux_norm)
            ivar_func = interpolate.interp1d(wl,self.Ivar[HIRES_ID])
            
            fluxDic[HIRES_ID] = flux_func(interpolate_over)
            self.Ivar[HIRES_ID] = ivar_func(interpolate_over)
 
        #Saving Data
        np.savetxt("interpolated_wl.csv",interpolate_over,delimiter=",")
        fluxDF = pd.DataFrame(fluxDic)
        fluxDF.to_csv("fluxes_for_HIRES.csv",index_label=False,index=False)
        ivarDF = pd.DataFrame(self.Ivar)
        ivarDF.to_csv("ivar_for_HIRES.csv",index_label=False,index=False)

        #This might be confusing to now make self.Ivar a dataframe when it was 
        #just a dictionary, but thats okay. Don't want to make too many variables.
        self.Ivar = ivarDF     
        self.fluxDF  = fluxDF
        self.interpolate_wl = interpolate_over
        print("Interpolation Has Ended")

    def SigmaCalculation(self,filename,fits_path=False):
        '''
        Calculates sigma for ivar 
        '''
        gain = 1.2 #electrons/ADU
        readn = 2.0 #electrons RMS
        xwid = 5.0 #pixels, extraction width
        sigma = np.sqrt((gain*self.spectraDic[filename]) + (xwid*readn**2))/gain
        self.Ivar[filename] = 1/(sigma**2)

import time 
if __name__ == '__main__':
    start_time = time.time()
    crossMatchedNames = pd.read_csv("../spocData/starnames_crossmatch_SPOCS_NEXSCI.txt",sep=" ")
    cirObject = CIR(crossMatchedNames["HIRES"].to_numpy(),'./wl_solution.csv','./RVOutput','./SpectraOutput')
    cirObject.Run()
    time_elap = time.time() - start_time 
    print(f"This took {time_elap/60} minutes!")