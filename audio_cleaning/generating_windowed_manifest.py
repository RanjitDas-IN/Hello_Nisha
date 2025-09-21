import os
import librosa
import numpy as np
import pandas as pd
import webrtcvad

# ----------------------
# Parameters
# ----------------------
positive_root = r"wav_16k_positive_voices"
negative_root = r"wav_16k_negative_voices"
window_length_sec = 0.8
hop_sec = 0.4
sr = 16000
min_window_sec = 0.5  # Drop windows shorter than this
output_psv = r"data/windowed_manifest.psv"

# ----------------------
# Helper functions
# ----------------------
def get_windows(audio, sr, win_len, hop_len, min_win_len):
    """Split audio into overlapping windows and drop very short trailing windows"""
    n_samples = len(audio)
    win_samples = int(win_len * sr)
    hop_samples = int(hop_len * sr)
    
    windows = []
    start = 0
    while start < n_samples:
        end = start + win_samples
        if end > n_samples:
            end = n_samples
        # Only keep windows >= min_win_len
        if (end - start)/sr >= min_win_len:
            windows.append((start, end))
        start += hop_samples
    return windows

def vad_detect(audio, sr, frame_ms=30):
    """Return a binary mask of speech/non-speech per frame using WebRTC VAD"""
    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)
    speech_mask = np.zeros(len(audio), dtype=bool)
    
    for start in range(0, len(audio), frame_len):
        end = start + frame_len
        frame = audio[start:end]

        if len(frame) < frame_len:
            pad_len = frame_len - len(frame)
            frame = np.pad(frame, (0, pad_len), mode='constant')

        pcm = (frame * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm, sample_rate=sr)

        actual_end = min(end, len(audio))
        speech_mask[start:actual_end] = is_speech

    return speech_mask

# ----------------------
# Process files
# ----------------------
records = []

def process_folder(root_dir, label):
    """Process all WAV files in a folder recursively"""
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(subdir, file)
                y, _ = librosa.load(wav_path, sr=sr, mono=True)

                windows = get_windows(y, sr, window_length_sec, hop_sec, min_window_sec)

                if label == 1:
                    speech_mask = vad_detect(y, sr)

                for start_idx, end_idx in windows:
                    start_time = start_idx / sr
                    end_time = end_idx / sr

                    if label == 0:
                        win_label = 0
                    else:
                        overlap = speech_mask[start_idx:end_idx]
                        win_label = 1 if np.any(overlap) else 0

                    records.append({
                        "wav_path": wav_path,
                        "window_start": start_time,
                        "window_end": end_time,
                        "label": win_label
                    })

# ----------------------
# Run processing
# ----------------------
process_folder(positive_root, label=1)
process_folder(negative_root, label=0)

# ----------------------
# Save manifest
# ----------------------
df = pd.DataFrame(records)
os.makedirs(os.path.dirname(output_psv), exist_ok=True)
df.to_csv(output_psv, sep="|", index=False)
print(f"Windowed manifest saved to {output_psv}")
