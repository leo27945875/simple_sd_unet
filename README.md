## A simple implementation of the UNet of StableDiffusion

This is a simpler implementation of the UNet of SD than [diffusers](https://github.com/huggingface/diffusers). Researchers can test their new ideas on this implementation more quickly and conveniently.

It can directly load the SD model config and weights in the [diffusers](https://github.com/huggingface/diffusers):

```python
from diffusers import StableDiffusionPipeline
from simple_sd_unet.unet import Cond_UNet2DModel

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = Cond_UNet2DModel(pipe.unet.config)
unet.load_state_dict(pipe.unet.state_dict())
```

![](figures/unet.png)

## Reference

* https://arxiv.org/abs/2112.10752
* https://github.com/huggingface/diffusers
