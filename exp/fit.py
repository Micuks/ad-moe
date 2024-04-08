import torch
from torch.utils.data import DataLoader, Dataset


class OBDataset(Dataset):
    def __init__(self, X_train_tensor, y_train):
        self.X = X_train_tensor
        self.y = y_train

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def fit(
    X_train_tensor,
    y_train,
    model,
    optimizer,
    epochs,
    batch_num,
    batch_size,
    clip_value=1.0,  # paramater for gradient clipping
    device=None,
):
    model.train()
    for epoch in range(epochs):
        dataloader = DataLoader(
            OBDataset(X_train_tensor, y_train), batch_size=batch_size, shuffle=True
        )
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= batch_num:
                break

            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                output = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        assert not torch.isnan(
                            param.grad
                        ).any(), f"NaN gradient in {name}"

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
