from torch.utils.data import DataLoader
import torch.optim as optim

model = simpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_dataset = lcDataset("processed_lcs", label_dict)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for X, y in train_loader:
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss = {loss.item():.4f}")

