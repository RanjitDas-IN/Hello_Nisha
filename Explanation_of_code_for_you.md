# This README is for explaining the written code
## HuBERT_dataset_creation.py:
### Explaination of the [HuBERT dataset creation process](audio_cleaning/HuBERT_dataset_creation.py)

```python
    # Load waveform
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
        
    # Slice window
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = waveform[:, start_sample:end_sample]
```
### Why multiply with sr:
### 1. `start * sr` and `end * sr`

* In your manifest, `start` and `end` are in **seconds** (e.g., 0.0, 0.8, 1.2, …).
* Audio in memory (`waveform`) isn’t stored in seconds, it’s stored in **samples**.

👉 To convert from seconds → samples, you multiply by the sampling rate (`sr`).

* Example: if `sr = 16000` (16kHz audio):

  * `start = 0.8 s` → `start_sample = 0.8 * 16000 = 12800 samples`.
  * `end = 1.6 s` → `end_sample = 1.6 * 16000 = 25600 samples`.

Because the manifest is **time-based (seconds)**, but audio arrays are **sample-based (integers)**. Multiplying by `sr` bridges that gap, so you can cut exactly the right piece of audio.

---

### 2. `segment = waveform[:, start_sample:end_sample]`

* `waveform` is a tensor shaped `(channels, samples)`.

  * For mono audio: `(1, total_samples)`.
* This line slices the waveform between `start_sample` and `end_sample`.

  * Example: take samples `12800 → 25600` from the waveform.
  * That corresponds exactly to `0.8s → 1.6s` of audio.

---


The **`generating_windowed_manifest.py`** script is responsible for creating training data windows. It walks through your aligned corpus, extracts both positive (wake word) and negative (non-wake word) segments, and saves them as `.wav` snippets. Alongside that, it writes a manifest file (`data/window_manifest.psv`) listing the snippet path, start time, end time, and label. This manifest is the bridge between raw aligned audio and downstream embedding extraction, ensuring you have balanced positive/negative samples to train on.

The **`HuBERT_embeddings_creation.py`** script consumes that manifest to generate embeddings. For each snippet path listed, it loads the waveform, resamples to 16 kHz, slices out the window, and passes the audio through a pre-trained HuBERT model (`facebook/hubert-base-ls960`). It then mean-pools the hidden states to get a fixed-size embedding vector, associates it with the label, and finally stores all embeddings/labels in a compressed `.npz` file. In other words, this step turns your audio segments into numerical vectors usable for training wake-word detection models.

👉 If you ran into an error while executing `HuBERT_embeddings_creation.py`, it’s likely linked to either (1) file paths in `window_manifest.psv`, (2) audio snippet sizes (empty segments), or (3) Hugging Face model loading/transformers–torchaudio version mismatch. Would you like me to troubleshoot based on the exact error traceback you saw?
