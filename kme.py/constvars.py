import torch
import logging


# --- public interface ---

__all__ = ['device', 'dtype']


# --- private interface ---

# set pytorch device (to gpu if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug("PyTorch device set to: ", device)

# set pytorch precision
dtype = torch.float32
logging.debug("PyTorch dtype set to: ", dtype)
