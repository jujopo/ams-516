import torch

if torch.cuda.is_available():
    print("CUDA is available! GPU detected.")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not available.")