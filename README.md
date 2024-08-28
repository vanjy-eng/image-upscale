# image-upscale
service to upscale an image.


Fixed an issue with importing 
```
from basicsr.archs.rrdbnet_arch import RRDBNet
```

using this [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985)



To change the inference device (GPU, CPU or MPS) directly in the code, follow the path below.
/image-upscale/venv/lib/python3.12/site-packages/realesrgan/utils.py:63