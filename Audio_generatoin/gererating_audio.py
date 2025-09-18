import pandas as pd
import edge_tts
import asyncio
import os
import re

# Load dataset
df = pd.read_csv("data/long_wake_up_dataset.csv", sep="|")  # or "single_wake_up_dataset.csv"

# Create output folder
voice_name = "en-US-AvaMultilingualNeural"
output_dir = os.path.join("long_voices", voice_name)
os.makedirs(output_dir, exist_ok=True)

def sanitize_filename(text):
    # Remove non-alphanumeric characters for filenames
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.replace(" ", "_")
    return text

async def generate_voices():
    for idx, row in df.iterrows():
        text = row["text"]
        file_name = f"{str(idx+1).zfill(3)}_{sanitize_filename(text)}.mp3"
        # print(file_name)
        output_path = os.path.join(output_dir, file_name)
        
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_path)
        print(f"Saved: {output_path}")

# Run the async function
asyncio.run(generate_voices())
