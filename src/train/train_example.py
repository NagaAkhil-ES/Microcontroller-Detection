from torch_utils.engine import train_one_epoch, evaluate
from torch import optim

from util.config import get_configs
from data.loader import get_data_loader
from model.loader import get_model
from util.env import seed_everything
from utils.general import Averager

if __name__ == "__main__":
    # load parameters and run args
    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=True)
    seed_everything(params.random_seed, use_deterministic_algorithms=False)
    device = "cuda:0"

    # Dataloader
    train_loader = get_data_loader(params, mode="train")
    val_loader = get_data_loader(params, mode="val")

    # Model
    model = get_model(params.model_arch, params.num_classes, params.lr).model
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=params.lr)

    # train one epoch
    # import pdb; pdb.set_trace()
    # train_loss_hist = Averager()
    # train_one_epoch(model, optimizer, train_loader, device, epoch=1, train_loss_hist=train_loss_hist, print_freq=1)
    evaluate(model, val_loader, device=device)