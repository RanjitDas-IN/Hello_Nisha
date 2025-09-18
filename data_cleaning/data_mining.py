import pandas as pd

df = pd.read_csv(r"/home/ranjit/Desktop/projects/Hello_Nisha/data/1_Main_Dataset.csv", sep="|")

# Filter utterances containing "Nisha" or "nisha" (case-insensitive)
utterance = df[df["utterance"].str.contains("nisha", case=False, na=False)]["utterance"]

# print(type(utterance))
print(utterance)

# with open(r"/home/ranjit/Desktop/projects/Hello_Nisha/data/temp.csv", "w") as file:
#     file.write("\n".join(utterance.astype(str).tolist()))