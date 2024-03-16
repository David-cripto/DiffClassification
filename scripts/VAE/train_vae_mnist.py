import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F


# Loss function for VAE
def vae_loss(reconstructed_x, x, mu, logvar):
    x = x.view(-1, 1, 28, 28)
    reconstructed_x = reconstructed_x.view(-1, 1, 28, 28)

    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train_model(model, epochs, train_loader, lr=1e-3, device='cuda'):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_arr = []

    for epoch in tqdm(range(epochs)):
        loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            # Move data to the device
            data = data.to(device)

            # Forward pass
            reconstructed_images, mu, logvar = model(data)

            # Compute loss
            loss = vae_loss(reconstructed_images, data, mu, logvar)
            loss_arr.append(loss)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(mnist_loader.dataset)} "
            #           f"({100. * batch_idx / len(mnist_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")
        print(f'Epoch {epoch}, Loss: {loss}')
    return loss_arr