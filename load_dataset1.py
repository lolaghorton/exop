from torch.utils.data import Dataset
import numpy as np
import glob

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

