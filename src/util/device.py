import os
from util.config import get_config

def setup_computing_device(device_type, gpu_id="0"):
    """ To select computing device

    Args:
        device_type (str): "gpu" or "cpu"
        gpu_id (str, optional): "7" or "6, 7". Defaults to "0".

    Returns:
        str: string with selected device value as cpu or cuda
    """    
    if device_type == "gpu":  # select gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # gpu id
        device = "cuda"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # select cpu
        device = "cpu"
    return device

params = get_config("configs/run.yaml")
device = setup_computing_device(params.device_type, params.gpu_id)
from torch.cuda import device_count
print("##__NOTE___")
print(f"Device used: {device}")
print(f"torch cuda device count: {device_count()}\n")