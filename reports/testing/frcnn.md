
# val_step output with train loader

{'loss_classifier': tensor(1.8934, device='cuda:0'), 'loss_box_reg': tensor(0.2323, device='cuda:0'), 'loss_objectness': tensor(0.0805, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0169, device='cuda:0')}

loss 2.22306489944458


# train_step_output with train loader

{'loss_classifier': tensor(1.8934, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2323, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0805, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0169, device='cuda:0', grad_fn=<DivBackward0>)}

2.22306489944458


# train and val step output using train loader


## Run 1

batch 0 val loss_dict: {'loss_classifier': tensor(1.8084, device='cuda:0'), 'loss_box_reg': tensor(0.2199, device='cuda:0'), 'loss_objectness': tensor(0.2822, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0199, device='cuda:0')}
batch 0 val loss: 2.330350875854492


batch 1 val loss_dict: {'loss_classifier': tensor(1.9270, device='cuda:0'), 'loss_box_reg': tensor(0.3350, device='cuda:0'), 'loss_objectness': tensor(0.2615, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0195, device='cuda:0')}
batch 1 val loss: 2.5430662631988525


batch 0 train loss_dict: {'loss_classifier': tensor(1.8084, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2199, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2908, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0199, device='cuda:0', grad_fn=<DivBackward0>)}
batch 0 train loss: 2.338897943496704


batch 1 train loss_dict: {'loss_classifier': tensor(1.9102, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.3411, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2545, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0194, device='cuda:0', grad_fn=<DivBackward0>)}
batch 1 train loss: 2.525148391723633


batch 2 train loss_dict: {'loss_classifier': tensor(1.9809, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1806, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2327, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0074, device='cuda:0', grad_fn=<DivBackward0>)}
batch 2 train loss: 2.401648759841919

batch 0 val loss_dict: {'loss_classifier': tensor(1.7615, device='cuda:0'), 'loss_box_reg': tensor(0.2217, device='cuda:0'), 'loss_objectness': tensor(0.2662, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0192, device='cuda:0')}
batch 0 val loss: 2.26857328414917

batch 1 val loss_dict: {'loss_classifier': tensor(1.8764, device='cuda:0'), 'loss_box_reg': tensor(0.3464, device='cuda:0'), 'loss_objectness': tensor(0.2440, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0191, device='cuda:0')}
batch 1 val loss: 2.48594069480896

batch 2 val loss_dict: {'loss_classifier': tensor(1.9610, device='cuda:0'), 'loss_box_reg': tensor(0.1789, device='cuda:0'), 'loss_objectness': tensor(0.2323, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0074, device='cuda:0')}
batch 2 val loss: 2.3796775341033936



## Run 2

batch 0 val loss_dict: {'loss_classifier': tensor(1.8084, device='cuda:0'), 'loss_box_reg': tensor(0.2199, device='cuda:0'), 'loss_objectness': tensor(0.2822, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0199, device='cuda:0')}
batch 0 val loss: 2.330350875854492

batch 1 val loss_dict: {'loss_classifier': tensor(1.9270, device='cuda:0'), 'loss_box_reg': tensor(0.3350, device='cuda:0'), 'loss_objectness': tensor(0.2615, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0195, device='cuda:0')}
batch 1 val loss: 2.5430662631988525


batch 0 train loss_dict: {'loss_classifier': tensor(1.8084, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2199, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2908, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0199, device='cuda:0', grad_fn=<DivBackward0>)}
batch 0 train loss: 2.338897943496704

batch 1 train loss_dict: {'loss_classifier': tensor(1.9100, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.3412, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2545, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0194, device='cuda:0', grad_fn=<DivBackward0>)}
batch 1 train loss: 2.5250892639160156

batch 2 train loss_dict: {'loss_classifier': tensor(1.9809, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1806, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.2337, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0074, device='cuda:0', grad_fn=<DivBackward0>)}
batch 2 train loss: 2.402658462524414

batch 0 val loss_dict: {'loss_classifier': tensor(1.7615, device='cuda:0'), 'loss_box_reg': tensor(0.2221, device='cuda:0'), 'loss_objectness': tensor(0.2703, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0192, device='cuda:0')}
batch 0 val loss: 2.273129940032959

batch 1 val loss_dict: {'loss_classifier': tensor(1.8766, device='cuda:0'), 'loss_box_reg': tensor(0.3448, device='cuda:0'), 'loss_objectness': tensor(0.2456, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0191, device='cuda:0')}
batch 1 val loss: 2.4860901832580566

batch 2 val loss_dict: {'loss_classifier': tensor(1.9610, device='cuda:0'), 'loss_box_reg': tensor(0.1789, device='cuda:0'), 'loss_objectness': tensor(0.2297, device='cuda:0'), 'loss_rpn_box_reg': tensor(0.0074, device='cuda:0')}
batch 2 val loss: 2.37705397605896

# Detections

## from eval_forward
(Pdb) detections[0]["scores"][0:5]
tensor([0.4036, 0.3916, 0.3476, 0.3458, 0.3443], device='cuda:0')

## from self model
(Pdb) t1 = self.model(b_image)
(Pdb) t1[0]["scores"][0:5]
tensor([0.4036, 0.3916, 0.3476, 0.3458, 0.3443], device='cuda:0')

# conclusion
- code is deteministic
- val_step loss with train_loader is equal to train_step loss with train_loader
- eval_forward's detection output is same as self.model output in val_step
- therefore, eval_forward's outputs are correct