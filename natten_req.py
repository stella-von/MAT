import natten
import torch

natten.use_fused_na(True)

print(natten.has_cuda())
print(natten.is_fused_na_enabled())

major, minor = torch.cuda.get_device_capability(0)
sm = major * 10 + minor
print(sm)
