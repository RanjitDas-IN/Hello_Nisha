import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np

# ----------------------
# Paths
# ----------------------
manifest_path = r"data/window_manifest.psv"
output_dir = r"/home/ranjit/Desktop/projects/Hello_Nisha/HuBERT_dataset"
os.makedirs(output_dir, exist_ok=True)

# Output file
output_file = os.path.join(output_dir, "hubert_embeddings.npz")

# ----------------------
# Load HuBERT (download first time, then cache)
# ----------------------
device = torch.device("cpu")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model = model.to(device)
model.eval()

# ----------------------
# Read manifest
# ----------------------
df = pd.read_csv(manifest_path, sep="|")

embeddings = []
labels = []

# ----------------------
# Process each row
# ----------------------
for idx, row in df.iterrows():
    wav_path = row["wav_path"]
    start = float(row["window_start"])
    end = float(row["window_end"])
    label = int(row["label"])

    # Load waveform
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # Slice window
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = waveform[:, start_sample:end_sample]

    if segment.size(1) == 0:
        continue  # skip empty windows

    # >>> ADD THIS BLOCK <<<
    MIN_SAMPLES = 400  # ~25 ms at 16kHz
    if segment.size(1) < MIN_SAMPLES:
        pad_len = MIN_SAMPLES - segment.size(1)
        segment = torch.nn.functional.pad(segment, (0, pad_len))


    # Extract features
    inputs = feature_extractor(segment.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, time, dim)
        pooled = hidden_states.mean(dim=1).squeeze().cpu().numpy()  # mean pooling

    embeddings.append(pooled)
    labels.append(label)

    if idx % 500 == 0:
        print(f"Processed {idx}/{len(df)} windows...")

# ----------------------
# Save dataset
# ----------------------
embeddings = np.stack(embeddings)
labels = np.array(labels)

np.savez(output_file, embeddings=embeddings, labels=labels)
print(f"Saved HuBERT dataset to {output_file}")




