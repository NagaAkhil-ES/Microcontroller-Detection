import pytorch_lightning as pl
from torch import optim
import torchvision.models.detection as tv_detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, backbone, num_classes, lr):
        super().__init__()
        self.model = self._get_model(backbone, num_classes) # model
        self.lr = lr #learning rate
        
    def _get_model(self, backbone, num_classes):
        if backbone == "mobilenet":
            # load Faster RCNN pre-trained model
            model = tv_detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # define a new head for the detector with required number of classes
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    def forward(self, x):
        return self.model(x)
  
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        b_image, b_target = batch
        # targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(b_image, b_target)
        loss = sum(loss for loss in loss_dict.values())
        # self.log_dict(loss_dict)  # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer