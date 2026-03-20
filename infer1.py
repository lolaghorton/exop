import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from cnn1 import smallerCNN
from pathlib import Path


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

    model = smallerCNN(input_length=1000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            probs = model(X) #sigmoid already inside model, not needed here
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    
    return dataset.files, probs





#ok so for deugging i will also still have this to infer on single lc
def infer_single_lc(npy_path, model_path): 
	#model_path is to the trained model file that has extension .pt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lc = np.load(npy_path).astype(np.float32)
    lc = torch.from_numpy(lc).unsqueeze(0).unsqueeze(0)  #(1, 1, 1000)

    model = smallerCNN(input_length=1000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        prob = model(lc.to(device)).item()

    return prob






if __name__ == "__main__":
#for inference on directory of lcs
    
    files, probs = infer(data_dir="processed_lcs/test", model_path="decentCNN.pt")
    #print(probs[:10])  #first 10 probs, values in 0 to 1
    
    #for each prob we get the corresponding file
    for f, p in zip(files, probs):
        name = f.stem #TIC_#_s#
        tic, sec = name.split("_s")
        print(f"TIC {tic} | Sector {sec} | Prob {p:.3f}")
    
    #save this to a txt
	with open("infer.txt", "w") as f:
		f.write("TIC\tSector\tProbability\n")
		for lc, p in zip(files, probs):
			name = lc.stem
			tic, sec = name.split("_s")
			f.write(f"{tic}\t{sec}\t{p}\n")
    

'''
#for inference on single lc
    tic = 234523599
    sec = 1
    #p = infer_single_lc(f"processed_lcs/infer/TIC_{tic}_s{sec}.npy", "smallerCNN_2.pt")
    p = infer_single_lc(f"processed_lcs/train/positive/CP/TIC_{tic}_s{sec}.npy", "trainedmodel.pt")
    print("Exoplanet probability:", p)
'''



'''
- THEN MAKE ANALYSIS SCRIPT W FUNC TO GRAB PROBS AND TIC THAT ARE OVER A CERTAIN THRESHOLD SO I CAN LOOK AT THEM MORE AND SEE IF THESE ARE ACTUAL GOOD CANDIDATES 
- LAST THEN MAKE PLOT FUNC TO TAKE THOSE CANDIDATES FOR ME TO LOOK AT BY EYE LIKE THE CHECK NOISY SCRIPT AND HOPEFULLY ITS OBVIOUS 
'''







