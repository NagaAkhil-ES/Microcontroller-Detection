""" Script of Get APIs to read params.yml & return train & test transforms """
import albumentations as A

def get_transforms(params, is_train=False, normalize=True):
    """ Get API to read params.yml and return transforms 

    Args:
        params (DotDict): parameters dictionary
        is_train (bool, optional): flag to return train transforms. Default to 
        Flase
        normalize (bool, optional): flag to consider data normalization. Default
        to True

    Returns:
        transforms.Compose: Composed object of transforms
    """
    transforms = []

    # Append spatial transformations to boost training
    if is_train:
        pass

    # Append basic transformation operations
    transforms.append(A.Resize(height=params.img_height, width=params.img_width))
    if normalize:
        transforms.append(A.Normalize(mean=params.mean, std=params.std, 
                                      max_pixel_value=1))

    # return composed transform
    transforms = A.Compose(transforms, bbox_params={'format': 'pascal_voc',
                                                    'label_fields': ['labels']})
    return transforms

def resize_bounding_box(data, frame_shape):
    width, height = frame_shape
    xmin = (data.xmin/data.width)*width
    xmax = (data.xmax/data.width)*width
    ymin = (data.ymin/data.height)*height
    ymax = (data.ymax/data.height)*height
    boxes = [[xmin, ymin, xmax, ymax]]
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return boxes, area


# testing block
if __name__ == "__main__":
    from util.config import get_config

    # read parameters
    params = get_config("configs/params.yaml", f_show=False)

    # import pdb; pdb.set_trace()
    tf = get_transforms(params, is_train=True)
    print("train transforms", "\n", tf)
    print("")
    tf = get_transforms(params, is_train=False)
    print("test transforms", "\n", tf)