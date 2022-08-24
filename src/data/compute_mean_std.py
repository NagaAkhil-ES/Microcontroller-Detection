import pandas as pd

from util.config import get_config
from data.dataset import CustomDataset_df
from data.transforms import get_transforms

# Main block for unit testing
if __name__ == "__main__":
    config_path = "configs/params.yaml"
    params = get_config(config_path, f_show=False)
    
    # Compose transforms with normalization
    trf = get_transforms(params, is_train=True, normalize=True)
    # Compose transforms with no normalization
    trf_nonorm = get_transforms(params, is_train=True, normalize=False)

    # Create dataset
    meta_df = pd.read_csv(params.train_csv_path)
    d_set = CustomDataset_df(meta_df, params.classes, trf)
    d_set_nonorm = CustomDataset_df(meta_df, params.classes, trf_nonorm)

    # Find mean and std
    mean, std = d_set.get_mean_std(n_samples=100)
    mean_nonorm, std_nonorm = d_set_nonorm.get_mean_std(n_samples=100)
    print(f"After normalization using the mean and std present in {config_path}")
    print(f"---\nmean: {mean.tolist()}\nstd: {std.tolist()}\n---")
    print("If you don't see mean ~ 0, std ~ 1,")
    print(f"place the following configuration in {config_path}:")
    print(f"---\nmean: {mean_nonorm.tolist()}\nstd: {std_nonorm.tolist()}\n---")