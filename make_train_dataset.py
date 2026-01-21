# making training set with tois.csv from EXOFOP-TESS
# for infer dataset ... well ill deal with that later
import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from lightkurve.correctors import PLDCorrector
import os

#tfopwg dispositions: apc, cp, fa, fp (aka ebs), kp, pc --> cp positives, fa and fp negatives
#based on playing with toi.csv, there should be 708 CPs, and 1232 FPs, and 99 FAs
#however there are some that are too short, or come back empty so itll prob be like 90% of those numbers ish?

#load in exofop csv
df = pd.read_csv("tois.csv")

#filter to only keep CP, FA, FP 
df = df[df["TFOPWG Disposition"].isin(["CP", "FP"])] #removed FA's bc they were throwing errors like no ones business

#loop row by row so we get tic id and disp per
df = df[["TIC ID", "TFOPWG Disposition"]].drop_duplicates()

#output folders
os.makedirs("processed_lcs/train/positive", exist_ok=True)
os.makedirs("processed_lcs/train/negative", exist_ok=True)

#loop thru tic ids
for _, row in df.iterrows():
    tic = int(row["TIC ID"])
    disp = row["TFOPWG Disposition"]
    
    label = "positive" if disp == "CP" else "negative"
    
    print(f"\nprocessing TIC {tic} ({disp} -> {label})")
    
    #now get into og processing
    try:
        #search for FFI cutouts (no sector specificed, will show all, well take first one, should be 30 min cadence
        sr = lk.search_tesscut(f"TIC {tic}")
        
        if len(sr) == 0:
            print(f"no tesscut data for TIC {tic}")
            continue
        
        #take first available cutout
        tpf = sr[0].download(cutout_size=10)
        
        #apply PLD correction
        pld = PLDCorrector(tpf)
        corrected_lc = pld.correct()
        
        #take first 1000 data points CONSIDER ROLLING AND REMOVING RANDOM POINTS???
        final_lc = corrected_lc[:1000]
        
        #flatten (detrend, not normalize)
        flat = final_lc.flatten()
        
        #check to see if lightkurve actually returned a proper lc or if it was too faint and aperture failed
        if flat is None or len(flat) == 0:
            print("empty lc after flatten, skipping")
            continue
        
        #convert to numpy array and normalize (mean 0, std 1)? if needed?
        flux = flat.flux.value
        #flux = (flux - np.mean(flux)) / (np.std(flux)
        
        #more checks
        if flux is None or len(flux) == 0:
            print("empty flux array, skipping")
            continue
        
        if not np.isfinite(flux).all():
            print("flux has nans/infs, skipping")
            continue
        
        if len(flux)<1000:
            print("flux too short, skipping")
            continue
        
        flux = flux[:1000]
        
        #save it 
        outpath = f"processed_lcs/train/{label}/TIC_{tic}.npy"
        np.save(outpath, flux)
        print(f"saved processed LC to {outpath}")
    
    except Exception as e:
        print(f"error processing TIC {tic}: {e}")
        continue

'''
processing TIC 150030205 (CP -> positive)
Warning: aperture mask contains zero pixels.
/home/lola/exop/.venv/lib64/python3.13/site-packages/lightkurve/utils.py:118: RuntimeWarning: invalid value encountered in scalar divide
  percent_masked = 100.0 * n_cadences_masked / n_cadences
/home/lola/exop/.venv/lib64/python3.13/site-packages/numpy/_core/fromnumeric.py:3860: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/home/lola/exop/.venv/lib64/python3.13/site-packages/numpy/_core/_methods.py:144: RuntimeWarning: invalid value encountered in divide
  ret = ret.dtype.type(ret / rcount)
/home/lola/exop/.venv/lib64/python3.13/site-packages/astropy/units/quantity.py:1889: RuntimeWarning: All-NaN slice encountered
  result = super().__array_function__(function, types, args, kwargs)
/home/lola/exop/.venv/lib64/python3.13/site-packages/numpy/lib/_nanfunctions_impl.py:2015: RuntimeWarning: Degrees of freedom <= 0 for slice.
  var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
error processing TIC 150030205: cannot reshape array of size 0 into shape (0,newaxis)
'''
#still got this error but i though the checks i had wouldve made this just send a skipping over print statement, few diff ones do this too
#theres also A LOT of these that are skipping because the flux is too short, and so maybe ill rerun it without trimming it and use the autosizing in torch






# now for making a test/validation set
# now time to make an inference set
# maybe i should be putting these into a def function and call instead of commenting out w ''' ''' lol
