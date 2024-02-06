import pytorch_lightning as pl
import torch

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.clip_embedder import FrozenCLIPEmbedder
from ldm.models.diffuse_unet import UNetModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LatentDiffusion(pl.LightningModule):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 unet_config, 
                 scale_factor=1.0,
                 *args, **kwargs):
        super().__init__()

        self.scale_factor = scale_factor

        # init AutoEncoder
        autoencoderKL_model = AutoencoderKL(**first_stage_config['params'])
        self.first_stage_model = autoencoderKL_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        
        # init text_encoder
        textencoder_model = FrozenCLIPEmbedder(version=cond_stage_config['ckpt'])
        self.cond_stage_model = textencoder_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        # UNetModel
        self.unet_model = UNetModel(**unet_config['params'])
        self.unet_model.eval()

    def get_first_stage_encoding(self, encoder_posterior):           # for img2img transform
        z = encoder_posterior
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        c = self.cond_stage_model.encode(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z):                        # decoder: decode latent feature to image
        z = 1. / self.scale_factor * z                      # scale_factor: 0.18215
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def apply_model(self, x_noisy, t, cond):
        # forward
        x_recon = self.unet_model(x_noisy, t, context=cond)
        return x_recon
