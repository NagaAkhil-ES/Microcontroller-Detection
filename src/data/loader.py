""" Script of Get API to setup and return pytorch data loader from params.yml"""
import os
import pandas as pd
from torch.utils.data import DataLoader
from PIL import ImageDraw
from torchvision.transforms import ToPILImage
from torch import stack as t_stack

from data.dataset import CustomDataset_df
from data.transforms import get_transforms
from util.utils import setup_save_dir

def get_data_loader(params, mode, normalize=True):
    """ Get API to setup and return pytorch data loader from params.yml for both
    train and val modes.

    Args:
        params (DotDict): parameters dictionary
        mode (str): train/val to select csv and set of transforms
        r_name (bool, optional): flag to return frame_path. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: data loader to iterate over dataset
    """
    if mode == "train":
        csv_path = params.train_csv_path
        transforms = get_transforms(params, is_train=True, normalize=normalize)
    elif mode == "val":
        csv_path = params.val_csv_path
        transforms = get_transforms(params, is_train=False, normalize=normalize)
    # data loader pipeline
    meta_df = pd.read_csv(csv_path)
    d_set = CustomDataset_df(meta_df, params.classes, transforms)
    d_loader = DataLoader(d_set, batch_size=params.batch_size, shuffle=False, 
                            num_workers=params.num_workers, collate_fn=collate_fn)
    return d_loader          

def collate_fn(batch):
    sample = list(zip(*batch))
    sample[0] = t_stack(sample[0])
    return sample

def save_transformed_images(images, targets, save_dir):
    # import pdb; pdb.set_trace()
    for i, target in enumerate(targets):
        image = ToPILImage()(images[i])
        img_draw = ImageDraw.Draw(image)
        for box in target["boxes"]:
            img_draw.rectangle(box.tolist(), outline="red")
        
        setup_save_dir(save_dir)
        image_id = int(target["image_id"])
        save_path = os.path.join(save_dir, f"{image_id}.png")
        image.save(save_path)    
    
# Main block
if __name__ == "__main__":
    from util.config import get_configs

    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=False)
    dl = get_data_loader(params, mode="train", normalize=False)

    # Show properties of an item
    images, targets = next(iter(dl))
    print("dataloader item:","\n")
    print("img")
    print(type(images), images.dtype, images.shape, images.min(), images.max(), "\n")
    print("target", type(targets))
    print(targets)

    save_transformed_images(images, targets, "reports/data_loader/album/test1")