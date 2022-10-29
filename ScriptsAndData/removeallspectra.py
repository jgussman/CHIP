import os 

dir = "./SpectraOutput/"
spectra = os.listdir(dir)

for spectrum in spectra: 
    os.remove(os.path.join(dir, spectrum) )