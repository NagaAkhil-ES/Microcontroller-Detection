import torch
import os
import random
import numpy as np

def setup_deterministic_env(seed=137):
    """ To set random seed for deterministic training to reproduce same results

    Args:
        seed (int, optional): seed value. Defaults to 3.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
