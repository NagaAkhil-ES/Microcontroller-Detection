from util.config import get_configs
from data.loader import get_data_loader
from model.loader import get_model
from util.env import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
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

    # call backs
    on_best_val_loss = ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min",
                                        save_last=True, filename="{epoch}-{val_loss:.4f}")

    on_best_val_map = ModelCheckpoint(monitor="val_map", save_top_k=3, mode="max",
                                     filename="{epoch}-{val_map:.3f}")

    on_best_val_map_small = ModelCheckpoint(monitor="val_map_small", save_top_k=3, mode="max",
                                            filename="{epoch}-{val_map_small:.3f}")

    on_best_val_map_large = ModelCheckpoint(monitor="val_map_large", save_top_k=3, mode="max",
                                            filename="{epoch}-{val_map_large:.3f}") 

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")  

    l_callbacks = [on_best_val_loss, on_best_val_map, on_best_val_map_small, on_best_val_map_large, early_stop_callback]                                                   

    # Trainer
    trainer = pl.Trainer(accelerator=params.accelerator, devices=params.devices, 
                        max_epochs=params.max_epochs, logger=logger, 
                        num_sanity_val_steps=0, log_every_n_steps=10,
                        callbacks=l_callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)