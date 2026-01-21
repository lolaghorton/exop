import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from load_dataset1 import LightCurveDataset
from cnn1 import decentCNN
from pathlib import Path

#ok again this was elementary and for simpleCNN, can do better
'''
#test inference on one lc
lc = np.load("processed_lcs/TIC_123456789.npy").astype(np.float32)
lc = torch.tensor(lc).unsqueeze(0).unsqueeze(0)  #shape (1, 1, 1000)

model = simpleCNN()
model.eval()
with torch.no_grad():
    p = model(lc)
    print("Transit probability:", float(p))
'''


#get the inference dataset, not using LightCurveDataset bc that has pos/neg labels and dont have those here
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("TIC_*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lc = np.load(self.files[idx]).astype(np.float32)
        lc = torch.from_numpy(lc).unsqueeze(0)  #shape of lcs (1, 1000)
        return lc


#do the actual inference
def infer(data_dir: str, model_path: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = InferenceDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = decentCNN(input_length=1000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            probs = model(X) #sigmoid already inside model
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)





#ok so for deugging i will also still have infer on single lc
def infer_single_lc(npy_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lc = np.load(npy_path).astype(np.float32)
    lc = torch.from_numpy(lc).unsqueeze(0).unsqueeze(0)  #(1, 1, 1000)

    model = decentCNN(input_length=1000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        prob = model(lc.to(device)).item()

    return prob





if __name__ == "__main__":
#for inference on directory of lcs
'''
    probs = infer(data_dir="processed_lcs/test", model_path="decentCNN.pt")
    #print(probs[:10])  #values in 0 to 1
    
    lcs = sorted(Path("processed_lcs/test").glob("TIC_*.npy")) 
    for f, p in zip(lcs, probs):
        print(f.name, p) #now for each prob we have the corresponding file
'''

#for inference on single lc
'''
    tic = 1
    p = infer_single_lc(f"processed_lcs/test/TIC_{tic}.npy", "decentCNN.pt")
    print("Exoplanet probability:", p)
'''


