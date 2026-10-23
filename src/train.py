
import random
import torch
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(cfg) -> tuple:
    """(device, warning or None). A visible warning on CPU: the full run would be very slow."""
   
    want_cuda = cfg.DEVICE == "cuda" or (cfg.DEVICE == "auto" and torch.cuda.is_available())
    if cfg.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but no GPU is available")
    device = torch.device("cuda" if want_cuda else "cpu")
    warning = None if want_cuda else "NO GPU: 'Esegui tutto' would be very slow. Use a GPU runtime."
    return device, warning