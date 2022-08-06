""" Script to combine csvs
requirement: There should be a csv for each folder"""
import pandas as pd
import os

from util.utils import listdir

# Main block
if __name__ == "__main__":
    # import pdb; pdb.set_trace()

    # Parameters
    csvs_dir = "data/original/Microcontroller Detection"
    images_dir = "data/processed/Images"
    out_csv_name = "data/csvs/microcontroller_detection.csv"

    # Read all csvs
    l_csvs = listdir(csvs_dir, ".csv")

    # Combine csvs
    combined_df = []
    for csv_name in l_csvs:
        meta_df = pd.read_csv(os.path.join(csvs_dir, csv_name))
        meta_df["tag"] = csv_name.split("_")[0]  # add tag to csv 
        s_image_path = images_dir + "/" + meta_df.filename # expand filename with relative path
        meta_df.insert(loc=0, column="image_path", value=s_image_path) # append image_path
        meta_df.drop("filename", axis=1, inplace=True) # remove filename column
        combined_df.append(meta_df)
        
    # save combined csv
    combined_df = pd.concat(combined_df, ignore_index=True)
    combined_df.to_csv(out_csv_name, index=False)