




## Option 1 coco evaluator code reference
coco = get_coco_api_from_dataset(data_loader.dataset)
iou_types = _get_iou_types(model)
coco_evaluator = CocoEvaluator(coco, iou_types)

for images, targets in data_loader:
    outputs = model(images)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
stats = coco_evaluator.summarize()


def validation_step(self, batch, batch_idx):
    images, targets = batch
    outputs = self.model(images, targets)
    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    self.coco_evaluator.update(res)
    return {}

def validation_epoch_end(self, outputs):
    self.coco_evaluator.accumulate()
    self.coco_evaluator.summarize()
    coco main metric
    metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
    metric = 0
    tensorboard_logs = {'main_score': metric}
    return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}