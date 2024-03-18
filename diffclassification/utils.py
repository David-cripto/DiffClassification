import torch as th
import numpy as np

def get_beta_schedule(num_diffusion_timesteps: int) -> th.Tensor:
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    betas = th.from_numpy(betas).double()
    return betas

def train_model(
    ddpm,
    dataloader,
    lr: float,
    weight_decay: float,
    n_epoch: int,
    device: str = "cpu",
    log_every: int = 500
):
    ddpm = ddpm.to(device)

    optimizer = th.optim.AdamW(
        ddpm.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    step = 0
    curr_loss_gauss = 0.0
    curr_count = 0
    from tqdm import trange
    tqdm_epoch = trange(n_epoch)
    for epoch in tqdm_epoch:
          avg_loss = 0.
          num_items = 0
          for x, y in dataloader:
              x = x.to(device)    
              loss = ddpm.train_loss(x, None)
              loss.backward()    
              optimizer.step()
              optimizer.zero_grad()
              avg_loss += loss.item() * x.shape[0]
              num_items += x.shape[0]
          tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))