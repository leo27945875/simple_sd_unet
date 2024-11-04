import unittest

import torch
from torchinfo import summary

from diffusers import StableDiffusionPipeline

from simple_sd_unet.unet import Cond_UNet2DModel
from simple_sd_unet.utils import load_json


class TestUNet(unittest.TestCase):
    def test_unet_arch(self):
        cfg = load_json("sd_unet_v15/unet.json")
        wgt = torch.load("sd_unet_v15/unet.pth", map_location="cpu")
        unet = Cond_UNet2DModel(cfg)
        unet.load_state_dict(wgt)
        summary(unet)

    def test_load_pipeline(self):
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        unet = Cond_UNet2DModel(pipe.unet.config)
        unet.load_state_dict(pipe.unet.state_dict())
        summary(unet)

