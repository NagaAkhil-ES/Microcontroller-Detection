from util.config import get_configs
from data.loader import get_data_loader
from model.loader import get_model
from util.env import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # load parameters and run args
    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=True)
    seed_everything(params.random_seed, use_deterministic_algorithms=True)
    
    # Dataloader
    train_loader = get_data_loader(params, mode="train")
    val_loader = get_data_loader(params, mode="val")

    # Model
    model = get_model(params.model_arch, params.num_classes, params.lr)

    # logger
    logger = TensorBoardLogger("models", name=params.exp_name)

    # Trainer
    trainer = pl.Trainer(accelerator=params.accelerator, devices=params.devices, 
                        max_epochs=params.max_epochs, logger=logger, 
                        num_sanity_val_steps=0, log_every_n_steps=10)
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)