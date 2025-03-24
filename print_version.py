import torch
import sys

# Get Python version
python_version = sys.version

# Get CUDA version
cuda_version = torch.version.cuda

# Get PyTorch version
pytorch_version = torch.__version__

# Print the versions
print(f"Python version: {python_version}")
print(f"CUDA version: {cuda_version}")
print(f"PyTorch version: {pytorch_version}")
