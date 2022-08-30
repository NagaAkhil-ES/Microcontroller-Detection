import torch

class Trainer:
    def __init__(self, model, train_dl, valid_dl, device):
        self.model = model.to(device)
        self.optimizer = model.configure_optimizers()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = device
    
    def _run_epoch(self, is_train):
        # setup parameters
        model = self.model.model
        optimizer = self.optimizer
        model.train(is_train)
        loader = self.train_dl if is_train else self.valid_dl

        for b_idx, batch in enumerate(loader):
            # import pdb; pdb.set_trace()            
            # load and transfer the batch to gpu
            b_image, b_target = batch
            b_image = b_image.to(self.device)
            b_target =  [{k: v.to(self.device) for k, v in t.items()} for t in b_target]

            # forward a batch of data to model
            with torch.set_grad_enabled(is_train):
                loss_dict = model(b_image, b_target)
                loss = sum(loss for loss in loss_dict.values())

            # backprop and update the parameters
            if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            print(f"batch {b_idx} loss: {loss}")
            break

    def fit(self, epochs):
        print("Model Training started..........")
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            self._run_epoch(is_train=True)
            if self.valid_dl is not None:
                valid_loss = self._run_epoch(is_train=False)
        print(f"\nModel Training completed!")

# unit testing
if __name__ == '__main__':
    from util.config import get_configs
    from data.loader import get_data_loader
    from model.loader import get_model
    from util.env import seed_everything

    # load parameters and run args
    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=False)
    seed_everything(params.random_seed)
    
    # Dataloader
    train_loader = get_data_loader(params, mode="train")
    val_loader = get_data_loader(params, mode="val")

    # Model
    model = get_model(params.model_arch, params.num_classes, params.lr)

    # Trainer
    tr = Trainer(model, train_loader, val_loader, "cuda:0")
    tr.fit(params.max_epochs)