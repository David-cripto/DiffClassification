from collections import defaultdict
import torch
import torch.optim as optim
from tqdm import tqdm



def train_epoch(model, train_loader, optimizer, use_cuda, loss_key='total', device='cuda'):
    model.train()

    stats = defaultdict(list)
    for batch_idx, (x, _) in enumerate(train_loader):
        if use_cuda:
            x = x.to(device)
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model, train_loader, use_cuda, device='cuda'):
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(train_loader):
            if use_cuda:
                x = x.to(device)
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(train_loader.dataset)
    return stats


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    use_tqdm=False,
    use_cuda=False,
    loss_key='total_loss',
    device='cuda' 
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.to(device)

    k = 0
    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key, device)
        test_loss = eval_model(model, test_loader, use_cuda, device)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
        print(f"{test_losses['elbo_loss']=}")
        print(f"{test_losses['kl_loss']=}")
        print(f"{test_losses['recon_loss']=}")

    return dict(train_losses), dict(test_losses)