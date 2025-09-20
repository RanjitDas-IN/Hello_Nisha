import pandas as pd
import matplotlib.pyplot as plt

file_path = r"Hello_Nisha/data/long_wake_up_dataset.psv"
# file_path = r"Hello_Nisha/data/mapping.psv"
df= pd.read_csv(file_path, sep="|")

print("Shape of the CSV:",df.shape)
# print(df.describe(include="all"))






def find_duplicates():
    """Find and print duplicated rows."""
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    print("\nNo of duplicate values:",duplicate_rows.shape[0])
    print("Duplicate rows:\n", duplicate_rows)



def drop_duplicates():
    """Drop duplicates in-place and save the updated CSV."""
    print("Before removing the duplicates shape:",df.shape)
    df.drop_duplicates(inplace=True)
    print("Shape after dropping duplicates:", df.shape)
    df.to_csv(file_path, sep="|", index=False)
    print("Cleaned data saved to CSV.")



def intent_count():
    intent_counts = df['intent'].value_counts()
    # Print the result
    print(intent_counts)
    print("The unique intents:",df["intent"].unique())




def check_word_count(padding_or_length):
    # assume df is already in scope
    df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

    # filter rows with exactly `padding_or_length` words
    matching_rows = df[df['word_count'] == padding_or_length]

    if not matching_rows.empty:
        print(f"Found {len(matching_rows)} utterance(s) with exactly {padding_or_length} words:\n")
        for idx, row in matching_rows.iterrows():
            print(f"- Row {idx + 1}: {row['word_count']} words, intent: {row['intent']}")
            print(f"  Utterance: \"{row['utterance']}\"\n")
    else:
        print(f"No utterances found with exactly {padding_or_length} words.")



def appearing_a_word_in_utterance(df, word="time"):
    total = 0
    print(f"\n🔍 Searching for the word '{word}' across all intents...\n")

    for intent in df["intent"].unique():
        gdf = df[df["intent"] == intent]
        count = 0

        for utterance in gdf["utterance"]:
            if word in utterance.lower():
                if count == 0:
                    print(f"👉 Intent: {intent}")
                print("   ", utterance)
                count += 1
                total += 1
        
        if count > 0:
            print(f"   ⟶ {count} matches in intent '{intent}'\n")

    print(f"\n✅ Total '{word}' occurrences across all intents: {total}")



def search_word_in_intent(df, intent: str, word: str):
    
    filtered_df = df[df["intent"] == intent]
    count = 0
    print(f"\n🔍 Searching for word '{word}' in intent '{intent}'...\n")

    for utterance in filtered_df["utterance"]:
        if word in utterance.lower():
            if count == 0:
                print(f"👉 Intent: {intent}")
            print("   ", utterance)
            count += 1

    if count > 0:
        print(f"\n   ⟶ {count} matches found in intent '{intent}'")
    else:
        print(f"❌ No matches found for '{word}' in intent '{intent}'")


    
def convert_weather_containing_utterance_to_google_search_intent_if_needed(csv_path: str, delimiter: str = '|'):
        
            df = pd.read_csv(csv_path, sep=delimiter)

            mask = df['utterance'].str.contains('weather', case=False) & (df['intent'] != 'google search')
            changes = df[mask]

            if not changes.empty:
                print("Changes to be made (intent changed to 'google search'):\n")
                print(changes)
            else:
                print("✅ No changes needed. No 'weather' entries outside of 'google search' intent.")

            df.loc[mask, 'intent'] = 'google search'
            # df.to_csv(csv_path, sep=delimiter, index=False)
            print(f"\n✔ Total entries modified: {len(changes)}")



def remove_outlier(
    csv_path: str,
    max_word_count: int,
    cleaned_csv_path: str = "No_outlier_dataset.psv",
    outliers_txt_path: str = "outliers.txt",
):
    """
    Removes utterances with word count >= max_word_count + 1
    Saves cleaned CSV and writes removed utterances to a .txt file.
    """
    # 1. Load the dataset
    df = pd.read_csv(csv_path, sep="|")

    # 2. Detect outliers (>= 22 words by default)
    is_outlier = df["utterance"].str.split().apply(len) > max_word_count

    # 3. Save outliers to .txt
    df.loc[is_outlier, "utterance"].to_csv(outliers_txt_path, index=False, header=False)

    # 4. Save cleaned dataset (non-outliers only)
    df_cleaned = df.loc[~is_outlier]
    df_cleaned.to_csv(cleaned_csv_path, sep="|", index=False)

    print(f"✅ Removed {is_outlier.sum()} utterances with > {max_word_count} words.")
    print(f"📁 Saved: {outliers_txt_path}")
    print(f"📁 Saved: {cleaned_csv_path}")






# print(df.describe(include="all"))
# intent_count()
# view_outliers(file_path)
# remove_outlier(df,221)
# find_duplicates()
# drop_duplicates()
# appearing_a_word_in_utterance(df)
# search_word_in_intent(df, intent="general", word="turn on")
# check_word_count(padding_or_length=21)
# convert_weather_containing_utterance_to_google_search_intent_if_needed(file_path)