""" Script of Get API to setup and return pytorch data loader from params.yml"""
import os
import pandas as pd
from torch.utils.data import DataLoader
from PIL import ImageDraw
from torchvision.transforms import ToPILImage

from data.dataset import CustomDataset_df
from data.transforms import get_train_transforms, get_test_transforms
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
        tv_transforms = get_train_transforms(params, normalize)
    elif mode == "val":
        csv_path = params.val_csv_path
        tv_transforms = get_test_transforms(params, normalize)
    # data loader pipeline
    meta_df = pd.read_csv(csv_path)
    d_set = CustomDataset_df(meta_df, params.classes, params.frame_shape, tv_transforms)
    d_loader = DataLoader(d_set, batch_size=params.batch_size, shuffle=False, 
                            num_workers=params.num_workers) #, collate_fn=collate_fn)
    return d_loader          

def collate_fn(batch):
    return tuple(zip(*batch))

def save_transformed_images(d_loader, save_dir):
    # import pdb; pdb.set_trace()

    images, targets = next(iter(d_loader))
    print("dataloader item:")
    print(images.shape)
    print(targets)
    print(f"Images Min: {images.min()}, Max: {images.max()}")
    
    # save transformed image
    boxes = targets["boxes"].int().tolist()
    image_ids = targets["image_id"].squeeze().tolist()
    setup_save_dir(save_dir)
    for i, image in enumerate(images):
        image = ToPILImage()(image)
        img1 = ImageDraw.Draw(image)
        for box in boxes[i]:
            img1.rectangle(box, outline="red")
        save_path = os.path.join(save_dir, f"{image_ids[i]}.png")
        image.save(save_path)
    
# Main block
if __name__ == "__main__":
    from util.config import get_configs
    params = get_configs("configs/run.yaml", "configs/params.yaml", f_show=True)
    dl = get_data_loader(params, mode="train", normalize=False)
    save_transformed_images(dl, "reports/data_loader/test6")