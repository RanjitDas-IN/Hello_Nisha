import pandas as pd
import edge_tts
import asyncio
import os
import re
import time

# Load dataset (choose long or single)
df = pd.read_csv("data/Negative_long_weak)up_dataset.psv", sep="|")  # or "single_wake_up_dataset.psv"

# Load voices list
voices_df = pd.read_csv("Audio_generatoin/egde_tts.psv")
voices_list = voices_df["voices"].str.strip().tolist()

def sanitize_filename(text):
    # Remove non-alphanumeric characters for filenames
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.replace(" ", "_")
    return text

async def generate_voices_for_voice(voice_name):
    output_dir = os.path.join("negative_voices", voice_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        text = row["text"]
        file_name = f"{str(idx+1).zfill(3)}_{sanitize_filename(text)}.mp3"
        output_path = os.path.join(output_dir, file_name)
        
        # ✅ Skip if file already exists
        if os.path.exists(output_path):
            # print(f"[{voice_name}] Skipped (already exists): {output_path}")
            continue
        
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_path)
        print(f"[{voice_name}] Saved: {output_path}")

async def generate_all_voices():
    # Loop through all voices in the CSV
    for voice in voices_list:
        print(f"Generating TTS for voice: {voice}")
        await generate_voices_for_voice(voice)

# Run the async function
asyncio.run(generate_all_voices())
time.sleep(50)
