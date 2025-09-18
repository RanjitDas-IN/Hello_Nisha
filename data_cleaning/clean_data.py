import pandas as pd
df = pd.read_csv(r"/home/ranjit/Desktop/projects/Hello_Nisha/data/long_wake_up_dataset.csv", sep = "|")
# print(df.head())

dff = df["text"]

print(dff)