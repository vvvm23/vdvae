import torch

info = lambda s: print(f"\33[92m> {s}\33[0m")
error = lambda s: print(f"\33[31m! {s}\33[0m")
warning = lambda s: print(f"\33[94m$ {s}\33[0m")

def get_device(try_cuda):
    if try_cuda == False:
        info("CUDA disabled by hyperparameters.")
        return torch.device('cpu')
    if torch.cuda.is_available():
        info("CUDA is available.")
        return torch.device('cuda')
    error("CUDA is unavailable but selected in hyperparameters.")
    error("Falling back to default device.")
    return torch.device('cpu')
