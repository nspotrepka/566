import torch

def torch_version():
    return torch.__version__

def cuda_is_available():
    return torch.cuda.is_available()

def cuda_device_count():
    return torch.cuda.device_count()

def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parallel(model):
    return nn.DataParallel(model) if cuda_device_count() > 1 else model
