#this is to make the inference LCs, basically same script as noisy_lcs.py, just switch some saving stuff

import os
import numpy as np
import pandas as pd
import random
import lightkurve as lk
from lightkurve.correctors import PLDCorrector

#set up stuff
SECTOR_CSV_DIR = "tics"  #folder containing all_targets_S#_v1.csv for # is sectors 1-26 (only sectors with 30 minute cadence)
N_RANDOM_TOTAL = 150  #total number of LCs i want to infer on
LC_LENGTH = 1000
OUTPUT_DIR = "processed_lcs/infer"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#read all sector csv's (1-26) and get TICs
csv_sectors = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
all_tics = []
for sector_num in csv_sectors:
    csv_path = os.path.join(SECTOR_CSV_DIR, f"all_targets_S0{sector_num}_v1.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path, skiprows=5)
    tics = df["TICID"].astype(int).tolist()
    # attach sector info
    all_tics.extend([(tic, sector_num) for tic in tics])

print(f"Total TICs available across all sectors: {len(all_tics)}")

#randomly sample from all available tic-sector pairs for N_RANDOM_TOTAL amount
random_tics = random.sample(all_tics, min(N_RANDOM_TOTAL, len(all_tics)))

#now the usual processing for the LCs
def process_tic(tic, sector):
    print(f"\nProcessing TIC {tic}, Sector {sector}")

    try:
        sr = lk.search_tesscut(f"TIC {tic}", sector=int(sector))
        if len(sr) == 0:
            print("No tesscut data for this TIC-sector")
            return False

        #filter 30-min cadence (exptime = 1426 s, so just grab any over 1000)
        table = sr.table
        filtered_indices = np.where(table["exptime"] > 1000)[0]

        if len(filtered_indices) == 0:
            print("No 30-min cadence for this TIC-sector")
            return False

        #loop over all matching indices for safety (usually one)
        for i in filtered_indices:
            mission_str = table[i]["mission"]
            sec_num = int(mission_str.split()[-1])

            res = sr[i]
            tpf = res.download(cutout_size=13)
            if tpf is None:
                continue

            pld = PLDCorrector(tpf)
            corrected_lc = pld.correct()
            
            if corrected_lc is None or len(corrected_lc) == 0:
            	continue

            #random 1000-point window to avoid positional bias
            if len(corrected_lc) > LC_LENGTH:
                start = np.random.randint(0, len(corrected_lc) - LC_LENGTH)
                segment = corrected_lc[start:start + LC_LENGTH]
            else:
                segment = corrected_lc
            
            norm = segment.normalize()
            flux = norm.flux.value
            
            if not np.isfinite(flux).all() or len(flux) == 0:
                continue

            #wrap short LCs
            if len(flux) < LC_LENGTH:
                repeats = int(np.ceil(LC_LENGTH / len(flux)))
                flux = np.tile(flux, repeats)
            
            #hardcode 1000 points to be sure
            flux = flux[:LC_LENGTH]

            #save it
            filename = f"TIC_{tic}_s{sec_num}.npy"
            outpath = os.path.join(OUTPUT_DIR, filename)
            np.save(outpath, flux)
            print(f"Saved: {outpath}")
            return True

    except Exception as e:
        print(f"Error processing TIC {tic}, Sector {sector}: {e}")
        return False

#main loop
saved = 0
for tic, sector in random_tics:
    success = process_tic(tic, sector)
    if success:
        saved += 1

print(f"\nFinished. Saved {saved} noisy-negative light curves.")
