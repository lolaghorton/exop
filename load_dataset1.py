import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from pathlib import Path

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
            processed_lcs/train
            processed_lcs/test
            processed_lcs/validation
        """
        self.data = []

        root_dir = Path(root_dir)

        for label_name, label_value in [("negative", 0), ("positive", 1)]:
            class_dir = root_dir/label_name
            for npy_file in class_dir.glob("TIC_*.npy"):
                self.data.append((npy_file, label_value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        lc = np.load(path).astype(np.float32)

        #shape of (1, 1000)
        lc = torch.from_numpy(lc).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float32)

        return lc, label

