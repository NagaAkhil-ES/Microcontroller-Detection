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
    def __init__(self, meta_df, l_classes, frame_shape, tv_transform=None):
        """ Custom Dataset as pytorch dataset subclass to handle data reading

        Args:
            meta_df (DataFrame): Object which holds Set's meta data
            l_classes (list): list classes to consider in label tensor
            tv_transform (torchvision.transforms.Compose, optional): Object which 
            contains all transform operations callable instances. Defaults to None.
        """
        self.meta_df = meta_df
        self.l_classes = l_classes
        self.frame_shape = frame_shape
        self.tv_transform = tv_transform
    
    def __len__(self):
        """attribute to get count of samples in meta_df

        Returns:
            int: length of input dataframe
        """
        return len(self.meta_df)
    
    def _read_frame(self, frame_path):
        """ Method to read a frame using PIL and return it as tensor after 
        applying torchvision transforms

        Args:
            frame_path (str): path of frame to read

        Returns:
            torch.Tensor: processed frame
        """
        frame = Image.open(frame_path).convert("RGB")
        frame = ToTensor()(frame)
        if self.tv_transform is not None:
            frame = self.tv_transform(frame)
        return frame
    
    def _read_boxes(self, data):
        width, height = self.frame_shape
        xmin = (data.xmin/data.width)*width
        xmax = (data.xmax/data.width)*width
        ymin = (data.ymin/data.height)*height
        ymax = (data.ymax/data.height)*height
        boxes = [[xmin, ymin, xmax, ymax]]
        # boxes = [[data.xmin, data.ymin, data.xmax, data.ymax]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return boxes, area

    def __getitem__(self, index):
        """ Attribute to return sample's features & label tensors at given index

        Args:
            index (int): sample index in meta_df

        Returns:
            str(optional), torch.Tensor, torch.Tensor: frame_path, frame's 
            features and label tensors
        """
        row = self.meta_df.iloc[index]
        image = self._read_frame(row.image_path)
        labels = [self.l_classes.index(row.type)]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes, area = self._read_boxes(row)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        filename = Path(row.image_path).stem
        image_id = int(filename.split("_")[-1])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([image_id])

        return image, target

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

# Main block for unit testing
if __name__ == "__main__":
    import pandas as pd
    from util.config import get_config
    from data.transforms import get_train_transforms

    config_path = "configs/params.yaml"
    params = get_config(config_path, f_show=False)
    
    # Compose transforms with normalization
    trf = get_train_transforms(params)
    # Compose transforms with no normalization
    trf_nonorm = get_train_transforms(params, normalize=False)

    # Create dataset
    meta_df = pd.read_csv(params.train_csv_path)
    d_set = CustomDataset_df(meta_df, params.classes, params.frame_shape, trf)
    d_set_nonorm = CustomDataset_df(meta_df, params.classes, params.frame_shape, trf_nonorm)

    # Find mean and std
    mean, std = d_set.get_mean_std(n_samples=1000)
    mean_nonorm, std_nonorm = d_set_nonorm.get_mean_std(n_samples=1000)
    print(f"After normalization using the mean and std present in {config_path}")
    print(f"---\nmean: {mean.tolist()}\nstd: {std.tolist()}\n---")
    print("If you don't see mean ~ 0, std ~ 1,")
    print(f"place the following configuration in {config_path}:")
    print(f"---\nmean: {mean_nonorm.tolist()}\nstd: {std_nonorm.tolist()}\n---")

    # Debug
    # import pdb;pdb.set_trace()
    # data = d_set[0]