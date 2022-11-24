from distutils.dir_util import copy_tree
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


np.random.seed(3)
class CHIP:

    def __init__(self,star_ID_array,wl_solution_path = '../SPOCSdata/wl_solution.csv',
                 rvOutputPath='./RVOutput',spectraOutputPath = './SpectraOutput'):
        '''
        Used for reducing deblazed Keck HIRES spectra for the purpose of putting the output 
        inot The Cannon. 

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
        if not os.path.exists(rvOutputPath):
            os.mkdir(rvOutputPath)
        self.dataSpectra = Download('prv.cookies',spectraOutputPath)       # For downloading Spectra 
        if not os.path.exists(spectraOutputPath):
            os.mkdir(spectraOutputPath)
        self.star_ID_array = star_ID_array                                 # HIRES ID 
        self.wl_solution = np.genfromtxt(wl_solution_path,skip_header=2)   # UNITS: Angstrom 
        self.Ivar = {}                                                     # For storing sigma valeus 
        self.filename_df = pd.DataFrame()                                  # For storing meta Data
        self.spectraDic = {}                                               # For storing spectra filenames
        
        
        
    
    def Run(self,use_past_data=False,alpha_normalize = False ):
        '''
        Description: This method will run all of the following
            Unless parameter is set to True
             find_rv_obs_with_highest_snr -> 
             Continuum Normalize ->
             CrossCorrelate -> 
             Interpolate 
        
        use_past_data:  Set to True if you have HIRES_Filename_rv.csv 
                  populated with the stars you want from a previous run. 
        alpha_normalize: Set to True if you have a ton of time or a really fast computer.
                        700 stars will take ~48 hours. Plus shipping and handling.
        '''
        self.use_past_data = use_past_data
        if not self.use_past_data:
            self.find_rv_obs_with_highest_snr()
            # self.DownloadSpectra()
        else:
            self.NoDownload()
        if not alpha_normalize:
            self.ContinuumNormalize()
        else:
            self.AlphaNormalization()
        self.CrossCorrelate()
        self.Interpolate()

    @staticmethod
    def calculate_SNR(spectrum):
        spectrum = spectrum[7:10].flatten() # Using echelle orders in the middle 
        SNR = np.mean(np.sqrt(spectrum))
        return SNR 

    def DeleteSpectrum(self,filename):

        #print(f"Deleting {filename}.fits")
        file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")
        try:
            os.remove(file_path)
        except FileNotFoundError as e:
            # file was already deleted 
            pass 

    def DownloadSpectrum(self,filename,SNR = True):
        '''Download Individual Spectrum and ivar 

        Input: filename (str): name of spectrum you want to download
            SNR (bool): if you want to calculate the SNR of the spectrum 

        Output: None
        '''
        #print(f"Downloading {filename}")
        self.dataSpectra.spectrum(filename.replace("r",""))  #Download spectra  
        file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")
        try: 
            temp_deblazedFlux = fits.getdata(file_path)
        except OSError: 
            return -1
        
        if SNR: # Used to find best SNR 
            # Delete Spectrum to save space
            SNR = self.calculate_SNR(temp_deblazedFlux)
            del temp_deblazedFlux 
            self.DeleteSpectrum(filename)
            return SNR

        else:
            return temp_deblazedFlux

    def use_path_spectrum(self):
        print("use_path_spectrum")
        self.star_snr_dic = dict()
        try:
            with open("stars_RV_found.txt",'r') as f: 
                star_meta_list = f.readlines()

                for star_meta_data in star_meta_list:

                    star_meta_data = star_meta_data.replace("\n","")

                    star_ID, star_filename, star_SNR = star_meta_data.split(" ")

                    self.star_snr_dic[star_ID] = (star_filename, star_SNR)
                    print(star_ID, star_filename, star_SNR)
        except:
            print("No file found called stars_RV_found.txt")

    def find_rv_obs_with_highest_snr(self):
        removed_stars = {"no RV observations":[],"rvcurve wasn't created":[], "SNR < 100":[],"NAN value in spectra":[],"No Clue":[]}
        hiresName_fileName_snr_dic = {"HIRESName": [],"FILENAME":[],"SNR":[]}  

        if self.use_past_data:
            self.use_path_spectrum()


        counter = 0
        for star_ID in self.star_ID_array:
            print(f"Stars completed: {counter}",end='\r')

            try:
                best_SNR_filename, best_SNR = self.star_snr_dic[star_ID]
                hiresName_fileName_snr_dic["HIRESName"].append(star_ID)
                hiresName_fileName_snr_dic["FILENAME"].append(best_SNR_filename)
                hiresName_fileName_snr_dic["SNR"].append(best_SNR)

                file_path = os.path.join(self.dataSpectra.localdir,filename + ".fits")
                temp_deblazedFlux = fits.getdata(file_path)
                self.spectraDic[best_SNR_filename] = temp_deblazedFlux
                continue

            except:
                pass 
                

            try:
                search_string = f"select OBTYPE,FILENAME from FILES where TARGET like '{star_ID}' and OBTYPE like 'RV observation';"

                url = self.state.search(sql=search_string)
                obs_df = pd.read_html(url, header=0)[0]

                if obs_df.empty: # There are no RV Observations
                    removed_stars["no RV observations"].append(star_ID) 
                    continue 
                
                else:   

                    best_SNR = 0
                    best_SNR_filename = False 
                    for filename in obs_df["FILENAME"]:
                        temp_SNR = self.DownloadSpectrum(filename)
                        
                        # Check if the SNR is the highest out of 
                        # all the star's spectras 
                        if best_SNR < temp_SNR:
                            best_SNR = temp_SNR

                            best_SNR_filename = filename 

                    if best_SNR < 100:
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
                removed_stars["No Clue"].append(star_ID)
            finally:
                counter += 1 
                continue
        print(f"Stars completed: {counter}")
        self.filename_df = pd.DataFrame(hiresName_fileName_snr_dic) 
        self.filename_df.to_csv("HIRES_Filename_snr.csv",index_label=False,index=False)
        
        print(removed_stars)
        print("Downloading Spectra Has Finished")


    def find_and_download_all_rv_obs(self):
        '''            
        Description: This method downloads the rotational velocity metadata and
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
                    #This is for removing these RV-less stars 
                    hiresName_fileName_rv_dic["RV"] += [pd.NA]
                    hiresName_fileName_rv_dic["FILENAME"] += [pd.NA]
            except OSError: #This error occurs because for some reason the star's rvcurve wasn't created
                    #This is for removing these stars that have no RV metadata 
                     
                    hiresName_fileName_rv_dic["RV"] += [pd.NA]
                    hiresName_fileName_rv_dic["FILENAME"] += [pd.NA]
        df = pd.DataFrame(hiresName_fileName_rv_dic) 
        self.filename_df = df.dropna() #If you don't drop the na's then other methods will break
        self.filename_df.to_csv("HIRES_Filename_rv.csv",index_label=False,index=False)
        print("Downloading RV Data Has Finished")

            
    def CrossCorrelate(self,numOfEdgesToSkip = 100):
        '''
        Description: Uses Pyastronomy's crosscorrRV function to compute the cross correlation.

                     The updated spectra are put into self.spectraDic.   

        numOfEdgesToSkip: the amount of pixels to cut off both ends of the spectra.
        '''
        print("Cross Correlate Has Began")
        solar = np.load('../Constants/solarAtlas.npy')
        wvlen = solar[:,1][::-1]
        c = solar[:,4][::-1]
        
        solar_echelle_list = []
        for echelle_num in range(16):
            begin,end = 4021*echelle_num, 4021*(echelle_num+1)
            temp_echelle_wv = self.wl_solution[begin:end] #UNITS: Angstrom
            temp_lower_bound = wvlen >= temp_echelle_wv[0]
            temp_upper_bound = wvlen[temp_lower_bound] <= temp_echelle_wv[-1]
            temp_wv = wvlen[temp_lower_bound][temp_upper_bound]
            temp_c = c[temp_lower_bound][temp_upper_bound]
            solar_echelle_list.append((temp_wv,temp_c))
        
        
        RV = 80 #60 gave good results 
        crossCorrelatedspectra = {} #Key: FILENAME Values: (correlated wavelength, normalized flux)
        for i in range(self.filename_df.shape[0]):
            try:
                row = self.filename_df.iloc[i]
                filename = row[1]
                normalizedFlux = self.spectraDic[filename]

                z_list = [] #Going to take the average of all the echelle shifts 
                for echelle_num in range(15,16): #echelle orders
    #             for echelle_num in range(16):
                    begin,end = 4021*echelle_num, 4021*(echelle_num+1)
                    e_wv = self.wl_solution[begin:end]  #hires 
                    e_flux = normalizedFlux[begin:end]
                    s_wv = solar_echelle_list[echelle_num][0]    #solar     
                    s_flux = solar_echelle_list[echelle_num][1]  

                    rv, cc = pyasl.crosscorrRV(e_wv, e_flux,s_wv,s_flux, -1*RV, RV, RV/200., skipedge=numOfEdgesToSkip)
                    argRV = rv[np.argmax(cc)]  #UNITS: km/s 
                    z = (argRV/299_792.458) #UNITS: None 
                    z_list.append(z)

                avg_z = np.mean(z_list)    
                computeShiftedWavelength = lambda wl: wl/ (1 + avg_z)  #UNITS: Angstroms
                #There has to be a better way to convert to a numpy array
                shifted_wl = np.array(list(map(computeShiftedWavelength,self.wl_solution.copy())))
                self.spectraDic[filename] = (shifted_wl,normalizedFlux)    
            except:
                pass 
        print("Cross Correlate Has Finished")
        

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
            
            #TODO: Add remove print statement
            print("Should be two arrays: wavelengths, flux",spectra_flux_tuple)
            
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
        
        #Wavelength range 
        firstKey = next(iter(self.spectraDic))
        first_spectra = self.spectraDic[firstKey][0]
        interpolate_over = [wl for wl in first_spectra if wl >= maxMinVal and wl<= minMaxVal]   
        length_interpolate = len(interpolate_over)
        
        spocs_wl = np.genfromtxt("../SPOCSdata/wavelengths_flat.txt")
        
        spocs_wl = spocs_wl[spocs_wl >= interpolate_over[0]]
        spocs_wl = spocs_wl[spocs_wl <= interpolate_over[-1]]
        
        interpolate_over = spocs_wl[::-1]
        interpolate_over.sort()
        
        length_interpolate = len(interpolate_over)
        
        #Interpolation         
        replacementSpectraDic = {}
        replacementIvarDic = {}
        
        for HIRESname,filename,rv in self.filename_df.to_numpy():
            try:
                wl = self.spectraDic[filename][0]
                flux_norm = self.spectraDic[filename][1]
                flux_func = interpolate.interp1d(wl, flux_norm)
                ivar_func = interpolate.interp1d(wl,self.Ivar[filename])

                replacementSpectraDic[HIRESname] = flux_func(interpolate_over)
                #Now ivar will actually become ivar 
                replacementIvarDic[HIRESname] = 1/ivar_func(interpolate_over)**2 
            except:
                pass
        if np.isnan(replacementIvarDic[HIRESname]).any(): #Happens in the ivar
                print(f"****{HIRESname} HAS BEEN REMOVED BECAUSE IT CONTAINS NAN VALUES")
                del replacementSpectraDic[HIRESname]
                del replacementIvarDic[HIRESname]
            
            
        self.spectraDic = replacementSpectraDic
        self.Ivar = replacementIvarDic
        self.interpolation_range = interpolate_over
        
        
        #Saving Data
        np.savetxt("interpolated_wl.csv",interpolate_over,delimiter=",",header='wavelength(Angstrom)')
        fluxDF = pd.DataFrame(self.spectraDic)
        np.save("stellar_names_for_flux_and_ivar",np.array(fluxDF.columns))
        np.save("fluxes_for_HIRES",fluxDF.to_numpy())
        ivarDF = pd.DataFrame(self.Ivar)
        np.save("ivars_for_HIRES",ivarDF.to_numpy())

        #This might be confusing to now make self.Ivar a dataframe when it was 
        #just a dictionary, but thats okay. Don't want to make too many variables.
        self.Ivar = ivarDF     
        self.fluxDF  = fluxDF
        self.interpolate_wl = interpolate_over
        print("Interpolation Has Finished")

    def SigmaCalculation(self,star_name):
        '''
        Description: Calculates sigma for inverse variance (ivar) 
        '''
        gain = 1.2 #electrons/ADU
        readn = 2.0 #electrons RMS
        xwid = 5.0 #pixels, extraction width

        sigma = np.sqrt((gain*self.spectraDic[star_name]) + (xwid*readn**2))/gain 
        #Checkinng for a division by zeros 
        if not np.isnan(sigma).any(): #Happens in the ivar
                self.Ivar[star_name] = sigma
        else:
            print(f"****{star_name} HAS BEEN REMOVED BECAUSE IT CONTAINS NAN VALUES") 
            del self.spectraDic[star_name]
        
    
    def AlphaNormalization(self):
        '''
        Description: Rolling Continuum Normalization. Takes forever and it mega worth it. 
        '''
        print("Alpha Normalization has Begun")

        wl_sixteen_echelle = np.ones((16,4021))
        for row in range(16):
            begin,end = 4021*row, 4021*(row+1)
            wl_sixteen_echelle[row] = self.wl_solution[begin:end] 

        # Create or keep Normalized_Spectra dir
        if not os.path.exists("Normalized_Spectra"):
            os.mkdir("Normalized_Spectra")

        counter = 0
        for star_name in self.spectraDic:
            print(f"Star Normalized Count: {counter}",end="\r")
            try:
                if os.path.exists(f"Normalized_Spectra/{star_name}_specnorm.npy") and os.path.exists(f"Normalized_Spectra/{star_name}_sigmanorm.npy"):
                    continue
                else:
                    raise FileExistsError
            except FileExistsError:
                contfit_alpha_hull(star_name,
                                    self.spectraDic[star_name],
                                    self.Ivar[star_name],
                                    wl_sixteen_echelle,"./Normalized_Spectra/") 

            counter += 1 
            

        for star_name in self.spectraDic:
            try:
                self.spectraDic[star_name] = np.load(f"Normalized_Spectra/{star_name}_specnorm.npy").flatten()
                self.Ivar[star_name] = np.load(f"Normalized_Spectra/{star_name}_sigmanorm.npy").flatten()
            except:
                print(f'''Something is wrong with {star_name}'s normalization.
                            We have removed it...''')
                del self.spectraDic[star_name]
                del self.Ivar[star_name]
                    
        print("Alpha Normalization has Ended")
         

    def NoDownload(self):
        '''
        Description: Need to have spectra already downloaded and HIRES_Filename_rv.csv needs to be made already.
        Only made this method because the internet at my parents house is extremely poor. 1.1 mbs 
        '''
        hires,file = np.genfromtxt("HIRES_Filename_snr.csv",delimiter=',',usecols=(0,1),skip_header=1,dtype='str',unpack = True)
        rv = np.genfromtxt("HIRES_Filename_snr.csv",delimiter=',',usecols=(2),skip_header=1)
        self.filename_df = pd.DataFrame({"HIRESName":hires,'FILENAME':file,"RV":rv})
        download_Location = self.dataSpectra.localdir #This is the second parameter of hiresprv.download.Download
        self.spectraDic = {}
        for filename in self.filename_df["FILENAME"]:
            #I tried to use the , seperation and new line seperation 
            #for the different file names but it doesn't seem to work.
            #Thus, a for-loop was used!
            #self.dataSpectra.spectrum(filename.replace("r",""))  #Download spectra 
            file_path = "{0}/{1}.fits".format(download_Location,filename)
            try:
                temp_deblazedFlux = fits.getdata(file_path).flatten()
                self.spectraDic[filename] = temp_deblazedFlux
                self.SigmaCalculation(filename)
            except OSError: #More of a problem with fits but that is okay
                print(f"{filename} has a problem with it's spectra")
        print("Completed Retrieving past data")        
             
if __name__ == '__main__':
     
    
    config_values_array = np.loadtxt("config.txt", dtype=str, delimiter='=', usecols=(1))
    crossMatchedNames = pd.read_csv(config_values_array[1].replace(" ",""),sep=" ")
    hiresNames = crossMatchedNames["HIRES"].to_numpy()
    chipObject = CHIP(hiresNames)
    
    past_q = (config_values_array[3].replace(" ","") == "True")
    alpha_q = (config_values_array[2].replace(" ","") == "True")

    import timeit

    exec_time = timeit.timeit(chipObject.Run(use_past_data=past_q,alpha_normalize = alpha_q))


    print(f"CHIP took {exec_time/60 :.2f} minutes!")