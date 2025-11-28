#test inference on one lc
lc = np.load("processed_lcs/TIC_123456789.npy").astype(np.float32)
lc = torch.tensor(lc).unsqueeze(0).unsqueeze(0)  #shape (1, 1, 1000)

model.eval()
with torch.no_grad():
    p = model(lc)
    print("Transit probability:", float(p))

