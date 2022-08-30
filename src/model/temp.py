

## Apply validation evaluation
from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
    def evaluate_preds_1(self, b_target, b_pred):
        metric = MeanAveragePrecision()
        metric.update(b_pred, b_target)
        print(metric.compute())



