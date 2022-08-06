""" Script of yaml config file apis to read and return parameters as DotDict"""
import yaml
from util.dot_dict import DotDict
import os

def get_config(config_path, f_show=False):
    """ Get API to read yaml config file and return parameters as DotDict

    Args:
        config_path (str): path of yaml config file like params.yml
        f_show (bool, optional): flag to display parameters. Defaults to False.

    Returns:
        DotDict: parameters dictionary
    """
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)
    if f_show:
        print(f"Input parameters from {config_path}")
        print(yaml.dump(params, default_flow_style=None, sort_keys=False))
    params = DotDict(params)
    return params

def get_configs(path1, path2, f_show=False):
    params = get_config(path1, f_show)
    params.update(get_config(path2, f_show))
    return params

def save_config(params, save_dir):
    """ To save yaml config file to the given dir

    Args:
        params (dict): Dictionary that contains parameters and its values 
        save_dir (str): dir path to save config file
    """
    save_path = os.path.join(save_dir, "config.yaml")
    params = dict(params)  # type cast to dict
    # write to config file
    with open(save_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=None, sort_keys=False)
    print(f"saved experiment's config at: {save_path}")

if __name__ == "__main__":
    # config_path = "configs/params.yaml"
    # params = get_config(config_path, f_show=True)
    params = get_configs("configs/run.yaml", "configs/params.yaml", True)
    save_config(params, "models/exp3_3")
    # import pdb; pdb.set_trace()