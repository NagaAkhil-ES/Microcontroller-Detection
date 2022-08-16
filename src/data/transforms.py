""" Script of Get APIs to read params.yml & return train & test transforms """
import torchvision.transforms as tf

def get_train_transforms(params, normalize=True):
    """ Get API to read params.yml and return train transforms using APIs of 
    torchvision transforms(tf). Each key in `params.train_transforms` as API in 
    `tf` and its value as API's kwargs. A transform operation is applied only 
    when 'kwargs.apply' is True. `kwargs.apply` is dropped before parsing API's 
    args. Any key-value pair in `params.train_transforms` can be invoked as `tf`
    APIs without modeification to this API for newly added operations

    Args:
        params (DotDict): parameters dictionary
        normalize (bool): Whether to normalize. Should be turned off if you
            want to compute mean and std from a sample. Defaults to True.

    Returns:
        torchvision.transforms.Compose: Composed object of train transforms
    """
    opt = params.train_transforms
    train_transforms = []
    for key in opt:
        kwargs = getattr(opt, key) # opt.<key> to get nested dot dict
        if kwargs.apply:
            kwargs.pop("apply")
            tf_op = getattr(tf, key) # get torchvision operation
            train_transforms.append(tf_op(**kwargs))
    train_transforms.append(tf.Resize(params.frame_shape))
    
    if normalize:
        train_transforms.append(tf.Normalize(mean=params.mean, std=params.std))
    
    train_transforms = tf.Compose(train_transforms)
    return train_transforms

def get_train_transforms_old(params):
    """ Get API to read params.yml and return train transforms using APIs of 
    torchvision transforms. Each key in `params.train_transforms` as API in 
    `tf` and its value as API's kwargs. Each transform is checked using .apply.
    For new addition of transform operations it is required to alter this API.

    Args:
        params (DotDict): parameters dictionary

    Returns:
        torchvision.transforms.Compose: Composed object of train transforms
    """
    opt = params.train_transforms
    train_transforms = []
    if opt.RandomRotation.apply:
        degrees = opt.RandomRotation.degrees
        train_transforms.append(tf.RandomRotation(degrees,expand=True))
    
    if opt.RandomHorizontalFlip.apply:
        prob = opt.RandomHorizontalFlip.p
        train_transforms.append(tf.RandomHorizontalFlip(prob))

    if opt.RandomVerticalFlip.apply:
        prob = opt.RandomVerticalFlip.p
        train_transforms.append(tf.RandomVerticalFlip(prob))

    if opt.ColorJitter.apply:
        kwargs = opt.ColorJitter
        kwargs.pop("apply")
        train_transforms.append(tf.ColorJitter(**kwargs))
    
    train_transforms.append(tf.Resize(params.frame_shape))
    train_transforms.append(tf.Normalize(mean=params.mean, std=params.std))
    train_transforms = tf.Compose(train_transforms)
    return train_transforms

def get_test_transforms(params, normalize=True):
    """ Get API to read params.yml and return test transforms using APIs of 
    torchvision transforms(tf). 

    Args:
        params (DotDict): parameters dictionary
        f_norm (bool, optional): flag to consider data normalization. Defaults 
        to True.

    Returns:
        torchvision.transforms.Compose: Composed object of test transforms
    """
    test_transforms = [tf.Resize(params.frame_shape)]
    if normalize:
        test_transforms.append(tf.Normalize(mean=params.mean, std=params.std))
    test_transforms = tf.Compose(test_transforms)
    return test_transforms

if __name__ == "__main__":
    from util.config import get_config
    config_path = "configs/params.yaml"
    params = get_config(config_path, f_show=False)
    # import pdb; pdb.set_trace()
    ''' trf = get_train_transforms(params)
    print(trf)
    trf = get_train_transforms_old(params)
    print(trf) '''
    trf = get_test_transforms(params)
    print(trf)
    # import pdb; pdb.set_trace()