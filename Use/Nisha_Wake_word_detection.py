import os
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from onnxruntime import InferenceSession

# -----------------------------
# Step 0: Setup HuBERT
# -----------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", local_files_only=True)
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960", local_files_only=True)

# -----------------------------
# Step 1: Load ONNX model
# -----------------------------
onnx_int8_path = "Model/mlp_model_best_int8.onnx"
session = InferenceSession(onnx_int8_path)
input_name = session.get_inputs()[0].name

# -----------------------------
# Step 2: Define functions
# -----------------------------
def get_hubert_embedding(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    waveform = waveform.squeeze().numpy()
    
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    return embedding.astype(np.float32)

def predict_onnx_with_confidence(embedding):
    pred = session.run(None, {input_name: embedding})[0]

    if pred.ndim == 1:
        # Single-class output
        pred_class = int(pred[0])
        confidence = 1.0  # model only returns class, assume full confidence
    else:
        # Multi-class probabilities
        pred_class = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred))
    return pred_class, confidence

# -----------------------------
# Step 3: Process folder
# -----------------------------
folder_path = r"testing/positive"

wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
# wav_files = wav_files[:10]  # take only first 10 files

results = {}
for file_name in wav_files:
    wav_path = os.path.join(folder_path, file_name)
    emb = get_hubert_embedding(wav_path)
    pred_class, conf = predict_onnx_with_confidence(emb)
    results[file_name] = (pred_class, conf)

# Print results
for k, (v, c) in results.items():
    label = 'Weak word detected' if v == 1 else 'No weak word'
    print(f"{k}: {label} (confidence: {c:.2f})")
