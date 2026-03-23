To install packages needed, used UV (set up instructions used: https://emily.space/posts/251023-uv). With pyproject.toml and README.md in a directory run uv sync to install all dependencies. uv.lock file will appear. If updating the pyproject.toml to add a dependency, can just re-run uv sync and everything will update in the venv. To run scripts use uv run filename.extension, no need for python3 filename.py or anything, uv does it all. 

BREAKDOWN OF FILES:

Set up files:
- uv.lock
- README.md
- pyproject.toml

LIGHT CURVE RELATED SCRIPTS:

used for these:
- tics/tois.csv -> CP, KP, FP
- tics/all_targets_S#_v1.csv -> noisy and inference LCs, # is place holder for sector
- processed_lcs -> directory stores all labeled light curves 
- plotttt.py -> plotting script to do some checks and debugging 

Positives: 
- CP TFOPWG disposition from EXOFOP tois.csv (710) *confirmed vetted planets

Go through the CP TIC ids search through lightkurve for the FFI. There is also a column for sectors this TIC is in, use that to grab all sector options, then filter to be sure its for exptime of 1426 s (30 min cadence). Process those, save as np arrays, name TIC_#_s#.npy where its the TIC id then the sector number, to processed_lcs/train/positives/CP. Script: exofop_lcs.py

- KP TFOPWG disposition from EXOFOP tois.csv (587) *known planet from different survey

Same as CP positives just saved to processed_lcs/train/positives/KP. Script: exofop_lcs.py

Negatives:

- FP TFOPWG disposition from EXOFOP tois.csv (1234) *false positives, mostly eclipsing binaries 

Same as CP positives just saved to processed_lcs/train/negatives/FP. Script: exofop_lcs.py

- Noisy non-transit stars 
Pulled randomly from 2 minute cadence TESS target lists in csv's per sector. Random pulls from various sectors csv's to avoid bias, for a certain amount pre determined of LCs to be produced. Same processing and labeling of files, saved to processed_lcs/train/negatives/noisy. Vetted by eye with script: check_noisy.py. Script: noisy_lcs.py

*FA disposition is mixture of TESS dispo. IS, O, V, PC, only 99 should I include? -> I think I have enough negatives since there are more than CPs alone for FPs, so ignore this tbh. 

Inference:

Basically the same script as for noisy lcs, just changes where things are saved. Script: infer_lcs.py




NEURAL NETWORK RELATED SCRIPTS:
- load_dataset1.py -> function used to gather all training/validation light curves and attach labels (positive/negative) when being run through training script 
- cnn1.py -> the actual neural network, current model: smallerCNN
- train1.py -> training/validation script to prep the model to be unleashed on data
- infer1.py -> the script to unleash the model on random data and use it 
- overfit_train.py -> a training script that specifically overfits the model, for debugging, ensures the model can actually learn (to the point of memorization)
- metrics_plots.py -> from training and overfit script get back loss function wrt epochs and predictions vs truth, use this to plot up loss curves and confusion matrices
- analysis.py -> various things to look at the data received from inference
