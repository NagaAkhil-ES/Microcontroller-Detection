from util.config import get_configs
from data.loader import get_data_loader
from model.loader import get_model
from util.env import seed_everything
import pytorch_lightning as pl

if __name__ == '__main__':
    # load parameters and run args
    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=True)
    seed_everything(params.random_seed)
    
    # Dataloader
    train_loader = get_data_loader(params, mode="train")
    val_loader = get_data_loader(params, mode="val")

    # Model
    model = get_model(params.model_arch, params.num_classes, params.lr)

    # Trainer
    trainer = pl.Trainer(accelerator=params.accelerator, devices=params.devices, 
                        max_epochs=params.max_epochs)
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)