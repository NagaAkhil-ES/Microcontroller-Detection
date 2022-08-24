import pandas as pd
import os


# Main block
if __name__ == "__main__":
    # import pdb; pdb.set_trace()

    # parameter
    input_csv = "data/csvs/microcontroller_detection.csv"
    val_ratio = 0.1
    seed = 137
    output_dir = "data/csvs"

    # load meta_df
    meta_df = pd.read_csv(input_csv)

    # generate test_df
    test_df = meta_df[meta_df.tag == "test"].copy()
    test_df.drop("tag", axis=1, inplace=True)

    # generate train-val df
    train_df = meta_df[meta_df.tag == "train"].copy()
    train_df.drop("tag", axis=1, inplace=True)
    val_df = train_df.sample(frac=val_ratio, random_state=seed)
    train_df.drop(val_df.index, inplace=True)

    train_df.to_csv(os.path.join(output_dir, f"train_s{seed}.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"val_s{seed}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test.csv"), index=False)