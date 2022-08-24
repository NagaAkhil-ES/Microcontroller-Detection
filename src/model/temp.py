import torch
import torchvision
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pytorch_lightning as pl

from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import RPNHead, MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor

from torch import nn, Tensor
from typing  import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, asdict 
from pathlib import Path, PosixPath
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

print(f"lightning: {pl.__version__}")
print(f"torch: {torch.__version__}")



class GDRTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**asdict(ModelParams()))

    def training_step(self, batch, batch_idx):
        images, targets = batch 
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log(f"train_loss", losses, prog_bar=True)
        return {"loss": losses, "outputs": {k:v.detach() for k, v in loss_dict.items()}} 

    def validation_step(self, batch, batch_idx):
        images, targets = batch 
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        ## we are using image size of 1 
        ## Example: {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0')}
        preds = self.model(images, targets)[0]
        return {"preds": preds, "targets": targets[0]} 

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val") 

    def epoch_end(self, outputs, phase):
        if phase == "val":
            preds = [i["preds"] for i in outputs]
            pred_bboxes = [torch.hstack([i["scores"].reshape(-1, 1), i
            ["boxes"]]).cpu().numpy() for i in preds]
            target_bboxes = [i["targets"]["boxes"].cpu().numpy() for i in outputs]
            pred_count = sum([i.shape[0] for i in pred_bboxes])
            f2_score = calc_f2_score(target_bboxes, pred_bboxes, False)
            self.log("val_f2_score", f2_score, prog_bar=True)
            self.log("pred_count", torch.as_tensor(pred_count).float(), prog_bar=True)

        elif phase == "train":
            final_train_loss = torch.mean(torch.stack([i["loss"] for i in outputs]))
            self.log("train_epoch_loss", final_train_loss, prog_bar=True)

            for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
                final_loss = torch.mean(torch.stack([i["outputs"][loss_type] for i in outputs]))
                self.log(f"train_{loss_type}", final_train_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"], momentum=self.cfg["momentum"], weight_decay=self.cfg["weight_decay"])
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": { "scheduler": lr_scheduler, "interval": "epoch",},} 