import os
import subprocess

# --------------------------
# Paths
# --------------------------
input_dir = "Insta_reels_audio/m4a_format"
output_dir = "Insta_reels_audio/converted_wav16k"

os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Loop through all .m4a files
# --------------------------
for file_name in os.listdir(input_dir):
    if file_name.endswith(".m4a"):
        input_file = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.wav")

        # ffmpeg command to convert to WAV 16kHz mono
        command = [
            "ffmpeg",
            "-y",            # overwrite if exists
            "-i", input_file,
            "-ar", "16000",  # sample rate 16 kHz
            "-ac", "1",      # mono
            output_file
        ]

        print(f"Converting {file_name} → {base_name}.wav ...")
        subprocess.run(command, check=True)

print("\n✅ All files converted to 16 kHz WAV format.")
