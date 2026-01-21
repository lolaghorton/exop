import numpy as np
import matplotlib.pyplot as plt

tic = 234523599 #good CP example
ticn = 389525208 #good FP (aka EB) example

lc_path = f"processed_lcs/train/positive/TIC_{tic}.npy" 
lc_pathn = f"processed_lcs/train/negative/TIC_{ticn}.npy"

flux = np.load(lc_path)
fluxn = np.load(lc_pathn)

plt.figure(figsize=(10, 4))
plt.plot(flux, lw=0.8)
plt.ylabel("Flux")
plt.title(lc_path)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(fluxn, lw=0.8)
plt.ylabel("Flux")
plt.title(lc_path)
plt.tight_layout()

plt.show()

