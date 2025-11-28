import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
import numpy as np

#test singular example from the paper
#pc
#tic = 258920431 
#sector = 18

#cp
#tic = 394346745
#sector = 12

#kp
#tic = 336732616
#sector = 1

#eb
#tic = 220435095
#sector = 9

#fa (fp for tfopwg is mostly ebsaccording to tess disp)
#tic = 29781292 #this one is instrument noise according tess disp
#sector = 13

#nothingness
tic = 27564782 #did random numbers, checked to be sure it wasnt in list from exofop
sector = 12

sr = lk.search_tesscut(f'TIC {tic}', sector=sector)
print(sr)

tpf = sr.download(cutout_size=25)
tpf.plot()

uncorrected_lc = tpf.to_lightcurve(aperture_mask='threshold')
uncorrected_lc.plot();



from lightkurve.correctors import PLDCorrector
pld = PLDCorrector(tpf)
corrected_lc = pld.correct()
pld.diagnose();

pld.diagnose_masks();

corr_len_lc = corrected_lc#[0:1000]

#rolled_lc = np.roll(corr_len_lc, np.random.randint(corr_len_lc.shape[0]), axis=0)

ax = corr_len_lc.flatten().plot()

show()

print(len(corrected_lc), len(corr_len_lc))
