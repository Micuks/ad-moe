import torch
import higher
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class THEDataset(Dataset):
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
            THEDataset(X_train_tensor, y_train), batch_size=batch_size, shuffle=True
        )
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= batch_num:
                break

            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            output = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, y)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()


def meta_fit(
    train_datasets,
    model,
    optimizer,
    meta_optimizer,
    epochs,
    batch_num,
    batch_size,
    meta_batch_size,
    inner_update_steps,
    inner_lr,
    clip_value=1.0,
    device=None,
):
    model.train()
    meta_train_dataloader = [
        DataLoader(dataset, batch_size=meta_batch_size, shuffle=True)
        for dataset in train_datasets
    ]

    for epoch in range(epochs):
        for meta_batch_idx, meta_batch in enumerate(meta_train_dataloader):
            meta_optimizer.zero_grad()

            meta_loss = 0
            for x_task, y_task in meta_batch:
                x_task, y_task = x_task.to(device), y_task.to(device).float()

                with higher.innerloop_ctx(
                    model, optimizer, copy_initial_weights=False
                ) as (fmodel, diffopt):
                    for _ in range(inner_update_steps):
                        output = fmodel(x_task)
                        task_loss = F.binary_cross_entropy_with_logits(output, y_task)
                        diffopt.step(task_loss)

                    with torch.no_grad():
                        output = fmodel(x_task)
                        val_loss = F.binary_cross_entropy_with_logits(output, y_task)
                    meta_loss += val_loss

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.patameters(), clip_value)
            meta_loss.backword()
            meta_optimizer.step()

            if meta_batch_idx >= batch_num:
                break

    return model
