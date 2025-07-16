# TEST FOR GPU

import torch
import onnxruntime

print("PyTorch", torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

print("\nonnxruntime", onnxruntime.__version__)
print(f"Available providers: {onnxruntime.get_available_providers()}")