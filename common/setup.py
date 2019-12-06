import os
import torch
from torch.utils.data import DataLoader
import torchaudio

def torch_version():
    return torch.__version__

def cuda_is_available():
    return torch.cuda.is_available()

def cuda_device_count():
    return torch.cuda.device_count()

def cpu():
    return torch.device('cpu')

def gpu(device=0):
    return torch.device('cuda:{}'.format(device))

def device():
    return gpu() if torch.cuda.is_available() else cpu()

def load(dataset, batch_size, num_workers=0):
    return DataLoader(dataset, batch_size, True, num_workers=num_workers)

def allow_kmp_duplicate_lib():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def init_audio():
    torchaudio.initialize_sox()

def shutdown_audio():
    torchaudio.shutdown_sox()
