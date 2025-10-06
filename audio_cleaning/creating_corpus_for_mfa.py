"""
creating the .lab
creating the .wav
whic is needed by the MFA (conda + mfa environment) 
"""

import os
import pandas as pd
import string

# ----------------------
# Paths
# ----------------------
negative_psv = r"/home/ranjit/Desktop/projects/Hello_Nisha/data/negative_mapping.psv"
positive_psv = r"/home/ranjit/Desktop/projects/Hello_Nisha/data/positive_mapping.psv"
corpus_dir = r"/home/ranjit/Desktop/projects/Hello_Nisha/corpus_for_mfa"

# ----------------------
# Voice columns
# ----------------------
voice_columns = [
    'en-US-AlloyTurboMultilingualNeural',
    'en-US-AndrewMultilingualNeural',
    'en-US-AvaMultilingualNeural',
    'en-US-BrandonMultilingualNeural',
    'en-US-BrianMultilingualNeural',
    'en-US-ChristopherMultilingualNeural',
    'en-US-CoraMultilingualNeural',
    'en-US-DavisMultilingualNeural'
]

# ----------------------
# Text normalization function
# ----------------------
def normalize_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Uppercase
    text = text.upper()
    # Collapse multiple spaces
    text = ' '.join(text.split())
    return text

# ----------------------
# Process a dataset
# ----------------------
def create_lab_files(psv_path, is_positive=True):
    df = pd.read_csv(psv_path, sep='|')
    
    for idx, row in df.iterrows():
        transcript = normalize_text(row['text'])
        
        for voice_col in voice_columns:
            wav_file = row[voice_col].strip()
            if not wav_file:
                continue  # skip if empty

            # Construct full path
            if is_positive:
                wav_path = os.path.join(corpus_dir, "wav_16k_positive_voices", voice_col, wav_file)
            else:
                wav_path = os.path.join(corpus_dir, "wav_16k_negative_voices", voice_col, wav_file)
            
            if not os.path.exists(wav_path):
                print(f"[Warning] WAV file not found: {wav_path}")
                continue

            # LAB path
            lab_path = os.path.splitext(wav_path)[0] + ".lab"

            # Write transcript
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write(transcript)

# ----------------------
# Run for both datasets
# ----------------------
print("Processing negative dataset...")
create_lab_files(negative_psv, is_positive=False)

print("Processing positive dataset...")
create_lab_files(positive_psv, is_positive=True)

print("All .lab files created successfully!")
