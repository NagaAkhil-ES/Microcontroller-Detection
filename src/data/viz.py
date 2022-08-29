import os
from PIL import ImageDraw
from torchvision.transforms import ToPILImage

from util.utils import setup_save_dir

def save_transformed_images(images, targets, save_dir):
    # import pdb; pdb.set_trace()
    for i, target in enumerate(targets):
        image = ToPILImage()(images[i])
        img_draw = ImageDraw.Draw(image)
        for box in target["boxes"]:
            img_draw.rectangle(box.tolist(), outline="red")
        
        setup_save_dir(save_dir)
        try:
            image_id = int(target["image_id"])
        except:
            image_id = i
        save_path = os.path.join(save_dir, f"{image_id}.png")
        image.save(save_path)    