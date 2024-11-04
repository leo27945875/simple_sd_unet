import torch
from torchinfo import summary

from simple_sd_unet.unet import Cond_UNet2DModel
from simple_sd_unet.utils import load_json


def test_unet_arch():
    cfg = load_json("sd_unet_v15/unet.json")
    wgt = torch.load("sd_unet_v15/unet.pth", map_location="cpu")
    unet = Cond_UNet2DModel(cfg)
    unet.load_state_dict(wgt)
    summary(unet)


if __name__ == "__main__":

    test_unet_arch()