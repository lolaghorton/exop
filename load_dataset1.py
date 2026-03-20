#this is what will be called in the training script to load in the processed LCs and give them labels for train/val

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from pathlib import Path
import random 

#this one didnt consider the positive and negative organization
class lcDataset(Dataset):
    def __init__(self, data_folder, label_dict):
        self.files = glob.glob(f"{data_folder}/TIC_*.npy")
        self.label_dict = label_dict  #dict: {tic_id: 0/1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        tic = int(path.split("_")[-1].split(".")[0])

        lc = np.load(path).astype(np.float32)
        lc = torch.tensor(lc).unsqueeze(0)  #shape is (1, 1000)

        label = torch.tensor(self.label_dict[tic]).float()

        return lc, label


class LightCurveDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir options:
            processed_lcs/train -> has /positive (/CP and /KP), /negative (/FP and /noisy)
            processed_lcs/test -> same orgaization as train
            processed_lcs/validation -> again same as train
        """
        self.data = []

        root_dir = Path(root_dir)

		#explicitly define file structure
		class_map = {"positive/CP": 1, "positive/KP": 1, "negative/FP": 0, "negative/noisy": 0}
		
		for subpath, label in class_map.items():
			class_dir = root_dir / subpath
			
			if not class_dir.exists():
				continue
			
			for npy_file in class_dir.glob("TIC_*.npy"):
				self.data.append((npy_file, label))
			
		print(f"loaded {len(self.data)} samples from {root_dir}")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        lc = np.load(path).astype(np.float32)

        #shape of (1, 1000)
        lc = torch.from_numpy(lc).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float32)

        return lc, label



'''
if the pos to neg is super imbalanced:

can do weighted loss func: 
pos_weight = torch.tensor([neg_count / pos_count])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

or balanced sampling:
use WeightedRandomSampler
'''
