import pytorch_lightning as pl
from torch import optim
import torchvision.models.detection as tv_detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from typing import Tuple, List, Dict
import torch
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from torchvision.ops import nms as tv_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, backbone, num_classes, lr):
        super().__init__()
        self.model = self._get_model(backbone, num_classes) # model
        self.lr = lr #learning rate
        self.map_score = MeanAveragePrecision()
           
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
        loss_dict = self.model(b_image, b_target)
        loss = sum(loss for loss in loss_dict.values())
        return loss 
    
    def training_epoch_end(self, outputs):
        l_loss = torch.stack([i["loss"] for i in outputs])
        train_loss =  l_loss[-10:].mean()
        self.logger.experiment.add_scalars('loss', {'train': train_loss}, 
                                            self.current_epoch+1) 
        
    def eval_forward(self, model, images, targets):
        """
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # model.eval()

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training=True
        #model.roi_heads.training=True


        #####proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        model.rpn.training=False
        model.roi_heads.training=False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections

    def _evaluate_iou(self,target, pred):
        """Evaluate intersection over union (IOU) for target from dataset and output prediction from
        model."""

        if pred["boxes"].shape[0] == 0:
            # no box detected, 0 IOU
            return torch.tensor(0.0, device=pred["boxes"].device)
        return box_iou(target["boxes"], pred["boxes"]).diag().mean()

    def validation_step(self, batch, batch_idx):
        b_image, b_target = batch
        loss_dict, b_pred = self.eval_forward(self.model, b_image, b_target)
        loss = sum(loss for loss in loss_dict.values())
        self.map_score.update(b_pred, b_target)
        iou = torch.stack([self._evaluate_iou(t, o) for t, o in zip(b_target, b_pred)]).mean()
        return {'loss': loss, "iou":iou} 

    def validation_epoch_end(self, outputs):
        l_loss = torch.stack([i["loss"] for i in outputs])
        val_loss =  l_loss.mean()
        self.logger.experiment.add_scalars('loss', {'val': val_loss}, 
                                            self.current_epoch+1)
        self.log("val_loss", val_loss, prog_bar=False, logger=False)

        map_dict = self.map_score.compute()
        val_map = map_dict["map"]
        val_map_small = map_dict["map_small"]
        val_map_large = map_dict["map_large"]

        self.logger.experiment.add_scalar("val_map", val_map, self.current_epoch+1)
        self.logger.experiment.add_scalar("val_map_small", val_map_small, self.current_epoch+1)
        self.logger.experiment.add_scalar("val_map_large", val_map_large, self.current_epoch+1)
        
        self.log_dict({"val_map": val_map, "val_map_small": map_dict["map_small"],
                       "val_map_large": map_dict["map_large"] }, logger=False)

        l_iou = torch.stack([i["iou"] for i in outputs])
        val_iou = l_iou.mean()
        self.logger.experiment.add_scalar("val_iou", val_iou, self.current_epoch+1)

    def apply_nms(self, targets):
        for target in targets:
            keep = tv_nms(target["boxes"], target["scores"], 0.3)
            target["boxes"] = target["boxes"][keep]
            target["labels"] = target["labels"][keep]
            target["scores"] = target["scores"][keep]
            
        ''' detections = self.apply_nms(detections)
        from data.loader import save_transformed_images
        save_transformed_images(b_image, detections, "data/testing/with_nms1")
        import pdb; pdb.set_trace() '''
        return targets

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer