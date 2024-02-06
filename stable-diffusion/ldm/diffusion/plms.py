"""SAMPLING ONLY."""
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from ldm.diffusion.diffuse_util import (make_beta_schedule,
                                        make_ddim_sampling_parameters,
                                        make_ddim_timesteps, noise_like)


class PLMSSampler(object):
    def __init__(self, model, 
                 schedule="linear", 
                 **kwargs):
        super().__init__()

        self.model = model

        # params for sampler
        self.schedule = schedule                        # linear
        self.ddpm_num_timesteps = kwargs.pop('timesteps', None)         # 1000
        self.linear_start = kwargs.pop('linear_start', None)            # 0.00085
        self.linear_end = kwargs.pop('linear_end', None)                # 0.0120
        self.cosine_s = kwargs.pop('cosine_s', None)                    # 8e-3
        
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        
        betas = make_beta_schedule(self.schedule, 
                                   self.ddpm_num_timesteps, 
                                   linear_start=self.linear_start, 
                                   linear_end=self.linear_end, 
                                   cosine_s=self.cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        # to_cuda = lambda x: to_torch(x).clone().detach().to(self.model.device)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
                                            alphacums = self.alphas_cumprod.cpu(),
                                            ddim_timesteps = self.ddim_timesteps,
                                            eta = ddim_eta, verbose = verbose)
        
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(conditioning, size,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    temperature=temperature,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False, timesteps=None, mask=None, 
                      x0=None, log_every_t=100, temperature=1., 
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.betas.device                          # get compute device
        b = shape[0]
        if x_T is None:         # init latent code, [bs, 4, 64, 64]
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # timesteps:
        # array([  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
        #    261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
        #    521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
        #    781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981])
        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # time_range:
        # array([981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        #    721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481,
        #    461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221,
        #    201, 181, 161, 141, 121, 101,  81,  61,  41,  21,   1])
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):         # sample inter, sample steps
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:                        # remain for inpainting
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, cond, ts, index=index,
                                      temperature=temperature,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)

            if len(old_eps) >= 4:
                old_eps.pop(0)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, 
                      temperature=1., unconditional_guidance_scale=1., 
                      unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            # w/o negative prompt condition
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                # w negative prompt condition
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device = device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device = device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device = device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device = device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            return x_prev, pred_x0

        e_t = get_model_output(x, t)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
