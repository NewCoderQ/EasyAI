import argparse
import os

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast

# sampler
from ldm.diffusion.plms import PLMSSampler
# model
from ldm.models.latentdiffusion_model import LatentDiffusion


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        help="the negative prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",             # for latent feature
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")

    # load sd models, including text-encoder, AutoEncoderKL and Denoising Unet
    sd_ckpt = config.model_config.sd_ckpt
    model = LatentDiffusion(**config.get("model_config", dict()))
    print(f"Loading model from {sd_ckpt}")
    model_state_dict = torch.load(sd_ckpt, map_location="cpu")["state_dict"]
    
    model_state_dict = {state_key.replace('model.diffusion_model', 'unet_model') 
                        if state_key.startswith('model.diffusion_model') else state_key: model_state_dict[state_key]
                        for state_key in model_state_dict.keys()}

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # init Sampler
    sampler = PLMSSampler(model, **config['sampler_config'])

    # output folder
    sample_path = opt.outdir
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # sample size, batch_size
    batch_size = opt.n_samples

    # pormpt, pos and neg
    prompt = opt.prompt
    negative_prompt = opt.negative_prompt
    assert prompt is not None
    prompts = batch_size * [prompt]

    start_code = None
    with torch.no_grad():
        with autocast("cuda"):
            # text embedding
            uc = None               # for negative prompt
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if negative_prompt:
                uc = model.get_learned_conditioning(len(prompts) * [negative_prompt])

            # positive prompt
            # [bs, 77, 768]
            c = model.get_learned_conditioning(prompts)             # prompts: list
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]         # shape: [4, 64, 64]

            # import pdb; pdb.set_trace()
            samples_ddim, _ = sampler.sample(S=opt.sample_steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                x_T=start_code)

            x_samples_ddim = model.decode_first_stage(samples_ddim)             # autoEncoder decode to image
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1

    print(f"Your samples are ready and waiting for you here: \n{sample_path} ")


if __name__ == "__main__":
    main()
