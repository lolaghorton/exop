import numpy as np
import matplotlib.pyplot as plt

tic_pos = 234523599 #good CP example
tic_neg = 389525208 #good FP (aka EB) example

lc_path_pos = f"processed_lcs/train/positive/TIC_{tic_pos}.npy" 
lc_path_neg = f"processed_lcs/train/negative/TIC_{tic_neg}.npy"

flux_pos = np.load(lc_path_pos)
flux_neg = np.load(lc_path_neg)

plt.figure(figsize=(10, 4))
plt.plot(flux_pos, lw=0.8)
plt.ylabel("Flux")
plt.title(lc_path_pos)
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(flux_neg, lw=0.8)
plt.ylabel("Flux")
plt.title(lc_path_neg)
plt.tight_layout()

plt.show()

