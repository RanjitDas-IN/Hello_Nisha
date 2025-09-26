import os
import librosa
import soundfile as sf

# Input and output directories
input_dir = r"long_voices"
output_dir = r"wav_16k_positive_voices"

# Walk through all files in input_dir
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".mp3"):
            # Full path for input file
            input_path = os.path.join(root, file)

            # Load audio with librosa
            y, sr = librosa.load(input_path, sr=16000, mono=True)  # resample to 16kHz, mono

            # Normalize to [-1,1] (librosa.load already loads as float32 in [-1,1], but safe to normalize)
            y = y / max(1e-8, max(abs(y)))  

            # Compute relative path to maintain folder structure
            relative_path = os.path.relpath(root, input_dir)
            target_folder = os.path.join(output_dir, relative_path)

            # Create target folder if it doesn't exist
            os.makedirs(target_folder, exist_ok=True)

            # Output path
            output_path = os.path.join(target_folder, file.replace(".mp3", ".wav"))

            # Save as WAV float32
            sf.write(output_path, y, 16000, subtype='FLOAT')
            # sf.write(output_path, y, 16000, subtype="PCM_16")

            # print(f"{input_path}")
