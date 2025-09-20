import os
import pandas as pd


en_US_AdamMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AdamMultilingualNeural"))
en_US_AlloyTurboMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AlloyTurboMultilingualNeural"))
en_US_AmberNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AmberNeural"))
en_US_AnaNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AnaNeural"))
en_US_AndrewMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AndrewMultilingualNeural"))
en_US_AndrewNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AndrewNeural"))
en_US_AvaMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-AvaMultilingualNeural"))
en_US_BrandonMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-BrandonMultilingualNeural"))
en_US_BrianMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-BrianMultilingualNeural"))
en_US_ChristopherMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-ChristopherMultilingualNeural"))
en_US_CoraMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-CoraMultilingualNeural"))
en_US_DavisMultilingualNeural = sorted(os.listdir(r"/home/ranjit/Desktop/projects/Hello_Nisha/long_voices/en-US-DavisMultilingualNeural"))


df = pd.read_csv(r"/home/ranjit/Desktop/projects/Hello_Nisha/data/long_wake_up_dataset.csv", sep = "|")
# print(df.head())

text_data = df["text"]

# print(type(file_names))
# print(type(text_data))


data = {
    "text": text_data,
    "en-US-AdamMultilingualNeural": en_US_AdamMultilingualNeural,
    "en-US-AlloyTurboMultilingualNeural": en_US_AlloyTurboMultilingualNeural,
    "en-US-AmberNeural": en_US_AmberNeural,
    "en-US-AnaNeural": en_US_AnaNeural,
    "en-US-AndrewMultilingualNeural": en_US_AndrewMultilingualNeural,
    "en-US-AndrewNeural": en_US_AndrewNeural,
    "en-US-AvaMultilingualNeural": en_US_AvaMultilingualNeural,
    "en-US-BrandonMultilingualNeural": en_US_BrandonMultilingualNeural,
    "en-US-BrianMultilingualNeural": en_US_BrianMultilingualNeural,
    "en-US-ChristopherMultilingualNeural": en_US_ChristopherMultilingualNeural,
    "en-US-CoraMultilingualNeural": en_US_CoraMultilingualNeural,
    "en-US-DavisMultilingualNeural": en_US_DavisMultilingualNeural,
}
new_df = pd.DataFrame(data)
print(new_df.columns)
# new_df.to_csv(r'/home/ranjit/Desktop/projects/Hello_Nisha/data/temporary.csv', sep= "|" ,index=False)