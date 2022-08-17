"""Script which contains pytorch's custom dataset classes"""

# Import dependencies
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from pathlib import Path

# Custom dataset as data structure
class CustomDataset_df(Dataset):
    def __init__(self, meta_df, l_classes, transforms=None):
        """ Custom Dataset as pytorch dataset subclass to handle data reading

        Args:
            meta_df (DataFrame): Object which holds Set's meta data
            l_classes (list): list classes to consider in label tensor
            transforms (transforms.Compose, optional): Object which 
            contains all transform operations callable instances. Defaults to None.
        """
        self.meta_df = meta_df
        self.l_classes = l_classes
        self.transforms = transforms
    
    def __len__(self):
        """attribute to get count of samples in meta_df

        Returns:
            int: length of input dataframe
        """
        return len(self.meta_df)
    
    def __getitem__(self, index):
        """ Attribute to return sample's features & label tensors at given index

        Args:
            index (int): sample index in meta_df

        Returns:
            torch.Tensor, dict: image tensor, target dict of tensors
        """
        row = self.meta_df.iloc[index]

        # read image
        img = Image.open(row.image_path).convert("RGB")
        img = ToTensor()(img)
        
        # read target
        boxes = [[row.xmin, row.ymin, row.xmax, row.ymax]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = [self.l_classes.index(row.type)]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        filename = Path(row.image_path).stem
        image_id = int(filename.split("_")[-1])
        image_id = torch.tensor([image_id])

        target = {"boxes": boxes, "labels": labels, "area":area, 
                 "iscrowd": iscrowd, "image_id": image_id}
        
        # apply transformation
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_mean_std(self, n_samples=1000):
        l_image = []
        
        pbar = tqdm(desc="Reading samples", total=n_samples)
        for i in range(n_samples):
            image, _ = self[i]
            l_image.append(image)
            pbar.update(1)
        
        l_image = torch.stack(l_image)
        mean = l_image.mean(dim=(0,2,3))
        std = l_image.std(dim=(0,2,3))
        return mean, std

# Testing block
if __name__ == "__main__":
    import pandas as pd
    from util.config import get_config
    
    params = get_config("configs/params.yaml")
    meta_df = pd.read_csv(params.train_csv_path)
    d_set = CustomDataset_df(meta_df, params.classes)
    
    # import pdb; pdb.set_trace()
    img, target = d_set[0] # get a item
    print("img")
    print(type(img), img.dtype, img.shape, img.min(), img.max(), "\n")
    for key, value in target.items():
        print(key)
        print(type(value), value.dtype, value.shape, value, "\n")

    # Sample output
    # img
    # <class 'torch.Tensor'> torch.float32 torch.Size([3, 600, 800]) tensor(0.0118) tensor(0.9882) 

    # boxes
    # <class 'torch.Tensor'> torch.float32 torch.Size([1, 4]) tensor([[317., 265., 556., 342.]]) 

    # labels
    # <class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([1]) 

    # area
    # <class 'torch.Tensor'> torch.float32 torch.Size([1]) tensor([18403.]) 

    # iscrowd
    # <class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([0]) 

    # image_id
    # <class 'torch.Tensor'> torch.int64 torch.Size([1]) tensor([101826]) 