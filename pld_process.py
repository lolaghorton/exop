import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
import numpy as np
from lightkurve.correctors import PLDCorrector

#need to run thru tic list from csv file
#also need to somehow only pick one sector 
#also need to somehow only pick one that has the 30 min cadence (1426 exp time)

tic = 220435095
sector = 9

#find the star tic with sector chosen
sr = lk.search_tesscut(f'TIC {tic}', sector=sector)
#print(sr)

#download search result as a target pixel file, if there are multiple it takes the first in the list
tpf = sr.download(cutout_size=10) #pixel x pixel size of image since you cant grab full FFIs from mast LOOK INTO WHAT SIZE IS BEST did 25 then 10 and 10 looked like less noise

#apply the preset corrector to the tpf and turn into lc
pld = PLDCorrector(tpf)
corrected_lc = pld.correct() #mask type LOOK INTO MORE FOR CHOICES TBH

#make all of them 1000 points only (should i try and take points from only ends?)
final_lc = corrected_lc[0:1000]

#flatten (normalize doesnt do the name thing, look into that)
#add rolling?
ax = final_lc.flatten().plot()

show()
