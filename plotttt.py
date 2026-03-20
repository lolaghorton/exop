#just a wee script to check a single lc

import numpy as np
import matplotlib.pyplot as plt


tic = 224294697
sec = 21
disp = "noisy"
pn = "negative"
tvt = "train"

lc_path = f"processed_lcs/{tvt}/{pn}/{disp}/TIC_{tic}_s{sec}.npy" 

flux = np.load(lc_path)

plt.figure(figsize=(10, 4))
plt.plot(flux, lw=0.8)
plt.title(lc_path)
plt.tight_layout()
plt.show()


'''
#version for inference lcs if i need to peep

tic = 140578024
sec = 12

lc_path = f"processed_lcs/infer/TIC_{tic}_s{sec}.npy"

flux = np.load(lc_path)

plt.figure(figsize=(10, 4))
plt.plot(flux, lw=0.8)
plt.title(lc_path)
plt.tight_layout()
plt.show()
'''


