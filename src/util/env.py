import torch
import os
import random
import numpy as np

def seed_everything(seed=137, use_deterministic_algorithms=False):
    """ To set random seed for deterministic training to reproduce same results

    Args:
        seed (int, optional): seed value. Defaults to 3.
    """
    # seed everything
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # set deterministic algo
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)