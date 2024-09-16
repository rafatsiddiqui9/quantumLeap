import torch
import gc

# Your training code

# After training loop
del tensor_variable  # Replace with your tensor variable names
gc.collect()
torch.cuda.empty_cache()