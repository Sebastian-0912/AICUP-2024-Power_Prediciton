import pandas as pd
import os 

data_frames = []

base_path = "./dataset/36_TrainingData_for_test"
for file_name in os.listdir(base_path):
  file_path = os.path.join(base_path, file_name)
  df = pd.read_csv(file_path)
  data_frames.append(df)
  
merge_df = pd.concat(data_frames, ignore_index=True)

output_path = os.path.join(base_path, "combined_test_dataset.csv")

merge_df.sort_values(by=['LocationCode','DateTime'],inplace=True)


df_one_hot = pd.get_dummies(merge_df, columns=["LocationCode"], prefix="Location", dtype=float)

# df_one_hot.to_csv(output_path, index=False)

print(df_one_hot.columns)