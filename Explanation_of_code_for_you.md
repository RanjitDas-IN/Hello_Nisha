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