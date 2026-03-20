#this loops through all the noisy negatives as I need to vet them by eye quickly to be sure there isnt transits since im just grabbing random TIC ids

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

direc = "train" #train or validation or test

noisy_dir = Path(f"processed_lcs/{direc}/negative/noisy")

files = sorted(noisy_dir.glob("TIC_*.npy"))

for file in files:

    flux = np.load(file)

    plt.figure(figsize=(10, 4))
    plt.plot(flux, lw=0.8)
    plt.ylabel("Normalized Flux")
    plt.title(f"{file.name} in {direc}/negative/noisy")
    plt.tight_layout()
    plt.show()

