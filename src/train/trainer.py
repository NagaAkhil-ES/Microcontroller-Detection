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