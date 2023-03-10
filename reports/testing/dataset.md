# Dataset testing logs
## Sample output without transforms
```
img
<class 'torch.Tensor'> torch.float32 torch.Size([3, 600, 800]) tensor(0.0118) tensor(0.9882)

boxes
<class 'torch.Tensor'> torch.float32 torch.Size([1, 4]) tensor([[317., 265., 556., 342.]])

labels
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([1])

area
<class 'torch.Tensor'> torch.float32 torch.Size([1]) tensor([18403.])

iscrowd
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([0])

image_id
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([101826]) 
```

## transforms
```
train transforms 
 Compose([
  Resize(always_apply=False, p=1, height=200, width=200, interpolation=1),
  Normalize(always_apply=False, p=1.0, mean=[0.2947157025337219, 0.21146990358829498, 0.2231939733028412], 
            std=[0.29502108693122864, 0.23328842222690582, 0.24320556223392487], max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels'], 
'min_area': 0.0, 'min_visibility': 0.0, 'check_each_transform': True}, keypoint_params=None, additional_targets={})

test transforms 
 Compose([
  Resize(always_apply=False, p=1, height=200, width=200, interpolation=1),
  Normalize(always_apply=False, p=1.0, mean=[0.2947157025337219, 0.21146990358829498, 0.2231939733028412], 
  std=[0.29502108693122864, 0.23328842222690582, 0.24320556223392487], max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels'], 
'min_area': 0.0, 'min_visibility': 0.0, 'check_each_transform': True}, keypoint_params=None, additional_targets={})
```

## Sample output with transforms h=200, w=200 without norm
```
img
<class 'torch.Tensor'> torch.float64 torch.Size([3, 200, 200]) tensor(0.0196, dtype=torch.float64) tensor(0.9137, dtype=torch.float64) 

boxes
<class 'torch.Tensor'> torch.float32 torch.Size([1, 4]) tensor([[ 79.2500,  88.3333, 139.0000, 114.0000]]) 

labels
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([1]) 

area
<class 'torch.Tensor'> torch.float32 torch.Size([1]) tensor([1533.5831]) 

iscrowd
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([0]) 

image_id
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([101826]) 
```


## Sample output with transforms h=600, w=800 without norm
```
(env) ps40012941@dell-mys-gpu:~/mc_detection$ python src/data/dataset.py
img
<class 'torch.Tensor'> torch.float64 torch.Size([3, 600, 800]) tensor(0.0118, dtype=torch.float64) tensor(0.9882, dtype=torch.float64) 

boxes
<class 'torch.Tensor'> torch.float32 torch.Size([1, 4]) tensor([[317., 265., 556., 342.]]) 

labels
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([1]) 

area
<class 'torch.Tensor'> torch.float32 torch.Size([1]) tensor([18403.]) 

iscrowd
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([0]) 

image_id
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([101826]) 
```

## Sample output with transforms frame shape = 200,200 and with norm
```
img
<class 'torch.Tensor'> torch.float32 torch.Size([3, 200, 200]) tensor(-0.8793) tensor(2.7581) 

boxes
<class 'torch.Tensor'> torch.float32 torch.Size([1, 4]) tensor([[ 79.2500,  88.3333, 139.0000, 114.0000]]) 

labels
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([1]) 

area
<class 'torch.Tensor'> torch.float32 torch.Size([1]) tensor([1533.5831]) 

iscrowd
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([0]) 

image_id
<class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([101826]) 
```


## DataLoader without collate function
```
dataloader item: 

img
<class 'torch.Tensor'> torch.float32 torch.Size([4, 3, 200, 200]) tensor(0.) tensor(1.) 

{'boxes': tensor([[[ 79.2500,  88.3333, 139.0000, 114.0000]],

        [[ 72.5000,  79.6667, 128.5000, 129.0000]],

        [[ 48.5000,  66.6667, 167.2500, 177.0000]],

        [[ 95.7500, 112.0000, 147.7500, 148.0000]]]), 'labels': tensor([[1],
        [2],
        [3],
        [1]]), 'area': tensor([[ 1533.5831],
        [ 2762.6667],
        [13102.0840],
        [ 1872.0000]]), 'iscrowd': tensor([[0],
        [0],
        [0],
        [0]]), 'image_id': tensor([[101826],
        [101903],
        [101915],
        [102013]])}
<class 'dict'>
```

## Dataloader with collate function
```
dataloader item: 

img
<class 'torch.Tensor'> torch.float32 torch.Size([4, 3, 200, 200]) tensor(0.) tensor(1.) 

target <class 'tuple'>
({'boxes': tensor([[ 79.2500,  88.3333, 139.0000, 114.0000]]), 'labels': tensor([1]), 'area': tensor([1533.5831]), 'iscrowd': tensor([0]), 'image_id': tensor([101826])}, {'boxes': tensor([[ 72.5000,  79.6667, 128.5000, 129.0000]]), 'labels': tensor([2]), 'area': tensor([2762.6667]), 'iscrowd': tensor([0]), 'image_id': tensor([101903])}, {'boxes': tensor([[ 48.5000,  66.6667, 167.2500, 177.0000]]), 'labels': tensor([3]), 'area': tensor([13102.0840]), 'iscrowd': tensor([0]), 'image_id': tensor([101915])}, {'boxes': tensor([[ 95.7500, 112.0000, 147.7500, 148.0000]]), 'labels': tensor([1]), 'area': tensor([1872.]), 'iscrowd': tensor([0]), 'image_id': tensor([102013])})
```