# making training set with tois.csv from EXOFOP-TESS

import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from lightkurve.correctors import PLDCorrector
import os

#use pandas to load in the csv from EXOFOP
df = pd.read_csv("tois.csv")
tic_list = df["TIC ID"].astype(int).unique()

#make a folder to save results
os.makedirs("processed_lcs", exist_ok=True)

#loop thru TICs
for tic in tic_list:
    print(f"\nProcessing TIC {tic} ...")

    try:
        #search for FFI cutouts (no sector specified, will show all and well take the first one that shows (should be 30 min cadence))
        sr = lk.search_tesscut(f"TIC {tic}")

        if len(sr) == 0:
            print(f"No TESSCut data for TIC {tic}, skipping.")
            continue

        #automatically take the FIRST search result (usually primary mission FFI)
        tpf = sr[0].download(cutout_size=10)

        #apply PLD correction
        pld = PLDCorrector(tpf)
        corrected_lc = pld.correct()

        #take first 1000 data points CONSIDER ROLLING AND TAKING RANDOM POINTS???
        final_lc = corrected_lc[0:1000]

        #flatten LC (detrend, not normalize)
        flat = final_lc.flatten()
        #flat.plot()
        #show()

        #convert to numpy array bc idk what format i need yet so .npy for now it is
        flux = flat.flux.value

        #normalize (mean 0, std 1) idk abt this tbh, the flatten looks good alone
        #flux = (flux - np.mean(flux)) / np.std(flux)

        outpath = f"processed_lcs/TIC_{tic}.npy"
        np.save(outpath, flux)

        print(f"Saved processed LC to {outpath}")

    except Exception as e:
        print(f"Error processing TIC {tic}: {e}")
        continue

#ugh forgot that i need to specify which is positives and negatives from the disposition columns, ill get back to that
