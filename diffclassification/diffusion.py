import re
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
from copy import deepcopy
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(
        self,
        betas: th.Tensor,
        model: nn.Module,
        clip_x0: Optional[bool] = False,
        shape: Optional[th.Tensor] = None,
        update_ema_after: Optional[int] = 2000,
    ) -> None:
        super().__init__()

        self.forward_diffusion = ForwardDiffusion(betas=betas)
        self.reverse_diffusion = ReverseDiffusion(betas=betas, clip_x0=clip_x0)
        self.model = model
        self.num_timesteps = len(betas)

        self.ema = deepcopy(model)
        self.update_ema_after = update_ema_after
        self.ema_counter = 0

        self.register_buffer("betas", betas)
        self.register_buffer("clip_x0", th.tensor(clip_x0, dtype=bool))
        self.register_buffer("shape", shape)


    @property
    def device(self) -> None:
        return next(self.parameters()).device


    @th.no_grad()
    def sample(self, y: th.Tensor, num_samples) -> th.Tensor:
        assert self.shape is not None
        if self.ema_counter < self.update_ema_after:
            model = self.model
        else:
            model = self.ema
        if y is not None:
            assert num_samples == y.shape[0]
        x = th.randn((num_samples, *self.shape), device=self.device, dtype=th.float32) # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
        indices = list(range(self.num_timesteps))[::-1]

        for i in tqdm(indices):
            t = th.tensor([i] * num_samples, device=x.device)
            eps = self.model(x,t,y)   # predicted noise
            x = self.reverse_diffusion.p_sample(x, eps ,t)  # sampling
        return x, y


    @th.no_grad()
    def sample_up_to_t(self, y: th.Tensor, t_up) -> th.Tensor:
        assert self.shape is not None
        if self.ema_counter < self.update_ema_after:
            model = self.model
        else:
            model = self.ema

        num_samples = y.shape[0]
        x = th.randn((num_samples, *self.shape), device=self.device, dtype=th.float32)
        indices = list(range(self.num_timesteps))[:t_up-1:-1] 

        for i in tqdm(indices):  
            t = th.tensor([i] * num_samples, device=x.device) 
            eps = self.model(x,t,y)  
            x = self.reverse_diffusion.p_sample(x, eps ,t)
        return x, y


    def train_loss(self, x0: th.Tensor, y: th.Tensor) -> th.Tensor:
        self._update_ema()
        if self.shape is None:
            self.shape = th.tensor(list(x0.shape)[1:], device="cpu")
        t = th.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = th.randn_like(x0)     # true noise
        x_t = self.forward_diffusion.q_sample(x0,t,noise)
        eps = self.model(x_t,t,y) 
        loss = F.mse_loss(eps, noise)
        return loss


    def _update_ema(self):
        self.ema_counter += 1
        if self.ema_counter < self.update_ema_after:
            return
        if self.ema_counter == self.update_ema_after:
            self.ema.load_state_dict(self.model.state_dict())

        ema_weight = 0.99
        new_ema_state_dict = self.ema.state_dict()
        model_state_dict = self.model.state_dict()
        for key, val in new_ema_state_dict.items():
            if isinstance(val, th.Tensor):
                new_ema_state_dict[key] = (
                    ema_weight * new_ema_state_dict[key] +
                    (1 - ema_weight) * model_state_dict[key]
                )
        self.ema.load_state_dict(new_ema_state_dict)


    @classmethod
    def from_pretrained(cls: "DDPM", model: nn.Module, ckpt_path: str) -> "DDPM":
        ckpt = th.load(ckpt_path)
        model_state_dict = {
            re.sub("ema.", "", re.sub("model.", "", key)):
            val for key, val in ckpt.items() if "ema." in key
        }
        model.load_state_dict(model_state_dict)
        return cls(
            betas=ckpt["betas"],
            model=model,
            clip_x0=ckpt["clip_x0"],
            shape=ckpt["shape"],
        )



class BaseDiffusion:
    def __init__(self, betas: th.Tensor) -> None:
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=-1)
        self.num_timesteps = len(self.betas)



class ForwardDiffusion(BaseDiffusion):
    def q_mean_variance(self, x0: th.Tensor, t: th.Tensor) -> th.Tensor:
        sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        mean = _extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape)*x0
        #mean = th.sqrt(self.alphas_cumprod[t])*x0
        variance =  1 -  _extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape)
        return mean, variance


    def q_sample(self, x0: th.Tensor, t: th.Tensor, noise: Optional[th.Tensor]=None) -> th.Tensor:
        # sample from the distribution q(x_t | x_0) (use equation (1))
        if noise is None:
          noise = th.randn_like(x0) 
        mean, variance = self.q_mean_variance(x0,t)
        samples = mean + variance.sqrt()*noise
        return samples



class ReverseDiffusion(BaseDiffusion):
    def __init__(self, *args, clip_x0: Optional[bool]=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas_cumprod_prev = th.cat(
            [th.tensor([1.0], device=self.betas.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.variance = (1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)*self.betas
        self.xt_coef = self.alphas.sqrt() * (1 - self.alphas_cumprod_prev)/ (1-self.alphas_cumprod)
        self.x0_coef = self.alphas_cumprod_prev.sqrt()*(1-self.alphas)/(1-self.alphas_cumprod)
        self.clip_x0 = clip_x0


    def get_x0(self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor) -> th.Tensor:
        alphas_cumprod = _extract_into_tensor(self.alphas_cumprod,t,xt.shape)
        x0 = (xt - (1-alphas_cumprod).sqrt() * eps) / alphas_cumprod.sqrt()
        if self.clip_x0:
            x0 = x0.clamp(-1., 1.)
        return x0


    def q_posterior_mean_variance(
        self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        variance = _extract_into_tensor(self.variance , t, xt.shape)
        xt_coef = _extract_into_tensor(self.xt_coef, t,xt.shape)
        x0_coef =  _extract_into_tensor(self.x0_coef, t,xt.shape)
        mean = xt_coef * xt + x0_coef * self.get_x0(xt,eps,t)
        return mean, variance


    def p_sample(self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor) -> th.Tensor:
        mean, variance = self.q_posterior_mean_variance(xt=xt, eps=eps, t=t)
        noise = th.randn_like(xt, device=xt.device)

        nonzero_mask = th.ones_like(t)  # to not add any noise while predicting x0
        nonzero_mask[t == 0] = 0
        nonzero_mask = _extract_into_tensor(
            nonzero_mask, th.arange(nonzero_mask.shape[0]), xt.shape
        )
        nonzero_mask = nonzero_mask.to(xt.device)
        sample = mean + nonzero_mask * variance.sqrt() * noise
        return sample.float()


def get_beta_schedule(num_diffusion_timesteps: int) -> th.Tensor:
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    betas = th.from_numpy(betas).double()
    return betas


# it's just an utility function. basically, returns arr[timesteps], where timesteps are indices. (look at class Diffusion)
def _extract_into_tensor(arr: th.Tensor, timesteps: th.Tensor, broadcast_shape: Tuple):
    """
    Extract values from a 1-D torch tensor for a batch of indices.
    :param arr: 1-D torch tensor.
    :param timesteps: a tensor of indices to extract from arr.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)




