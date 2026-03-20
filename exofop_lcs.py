# making training set with tois.csv from EXOFOP-TESS

import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from lightkurve.correctors import PLDCorrector
import os
import random 

#tfopwg dispositions: apc, cp, fa, fp (aka ebs), kp, pc --> cp positives, fp negatives
#based on playing with toi.csv, there should be 708 CPs and 1232 FPs (only take 400)

LC_LENGTH = 1000

#load in exofop csv
df = pd.read_csv("tics/tois.csv")

#filter to only keep CP, KP, FP
df = df[df["TFOPWG Disposition"].isin(["CP", "FP", "KP"])] 

#loop row by row so we get tic id and disp per
df = df[["TIC ID", "TFOPWG Disposition"]].drop_duplicates()

#split function
def get_split():
	r = random.random()
	if r < 0.8:
		return "train"
	elif r< 0.9:
		return "validation"
	else:
		return "test"

#label mapping
def get_label_folder(disp):
	if disp == "CP":
		return "positive/CP"
	elif disp == "KP":
		return "positive/KP"
	else:
		return "negative/FP"

#create directories 
splits = ["train", "validation", "test"]
for split in splits:
	os.makedirs(f"processed_lcs/{split}/positive/CP", exist_ok=True)
	os.makedirs(f"processed_lcs/{split}/positive/KP", exist_ok=True)
	os.makedirs(f"processed_lcs/{split}/negative/FP", exist_ok=True)

#main loop
for _, row in df.iterrows():
	
	tic = int(row["TIC ID"])
	disp = row["TFOPWG Disposition"]
	
	print(f"\nProcessing TIC {tic} ({disp})")
	
	try:
		sr = lk.search_tesscut(f"TIC {tic}")
		
		if len(sr) == 0:
			continue
		
		#filter to 30-min cadence only (exptime=1426s) and grab any sector available
		table = sr.table
		
		filtered_indices = np.where(table["exptime"] > 1000)[0]
		
		if len(filtered_indices) == 0:
			print("no 30-min cadence sectors for this tic")
			continue 
		
		for i in filtered_indices:
			mission_str = table[i]["mission"]
			sector = int(mission_str.split()[-1])
			
			print(f"sector {sector}")
			
			try:
				res = sr[i]
				tpf = res.download(cutout_size=13)
				
				if tpf is None:
					continue 
			
				pld = PLDCorrector(tpf)
				corrected_lc = pld.correct()
						
				if corrected_lc is None or len(corrected_lc) == 0:
					continue 
				
				#random window to avoid positional bias 
				if len(corrected_lc) > LC_LENGTH:
					start = np.random.randint(0, len(corrected_lc) - LC_LENGTH)
					segment = corrected_lc[start:start + LC_LENGTH]
				else:
					segment = corrected_lc
		
				norm = segment.normalize()
				flux = norm.flux.value
				
				if not np.isfinite(flux).all():
					continue
			
				#wrap short lcs w numpy bc faster, for if short of 1000 points
				if len(flux) < LC_LENGTH:
					repeats = int(np.ceil(LC_LENGTH / len(flux)))
					flux = np.tile(flux, repeats)
			
				#hardcode 1000 points again to be sure
				flux = flux[:LC_LENGTH]
			
				#get file path of where its getting saved 
				split = get_split()
				label_folder = get_label_folder(disp)
		
				outdir = f"processed_lcs/{split}/{label_folder}"
				filename = f"TIC_{tic}_s{sector}.npy"
				outpath = os.path.join(outdir, filename)
			
				np.save(outpath, flux)
			
				print(f"Saved: {outpath}")
		
			except Exception as e:
				print(f"error_1: {e}")
				continue
	except Exception as e:
		print(f"error_2: {e}")
		continue

