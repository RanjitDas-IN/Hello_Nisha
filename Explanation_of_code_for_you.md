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

👉 Perfect! Here’s the **full structured no-code blueprint** for your real-time wake-up detection system with Nisha. I’ve broken it into components, workflow, buffer management, and decision logic, all aligned with your training setup and runtime preferences.

---

# 🗂 Blueprint: Real-Time Wake-Up Detection (NISHA)

## 1️⃣ Components Overview

| Component                           | Responsibility                                                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **PvRecorder Thread**               | Captures audio from mic in small frames (512 samples, ~32 ms) and maintains a rolling buffer.                         |
| **Rolling Audio Buffer**            | Stores the latest 1.5 s of audio for inference. Acts as sliding window for HuBERT embedding extraction.               |
| **HuBERT Feature Extractor**        | Converts audio buffer (1.5 s) into mean-pooled 768-dim embeddings. Ensures same preprocessing as training.            |
| **ONNX MLP Classifier**             | Performs inference on HuBERT embeddings, outputs probability of wake word.                                            |
| **Decision Aggregator / Smoothing** | Implements thresholding and majority vote: fires wake-up if ≥2 positives detected in ~3 s (2 overlapping windows).    |
| **Detection Handler**               | Executes action on wake-up (currently prints a line). Can be extended for audio playback or triggering other modules. |
| **Error & Overflow Handler**        | Monitors PvRecorder buffer, handles overflows/restarts, ensures continuous recording.                                 |

---

## 2️⃣ Audio & Buffer Management

* **Window length:** 1.5 s (matches training)
* **Hop / stride:** 0.4 s (sliding window, overlapping ~1.1 s)
* **Frame size for PvRecorder:** 512 samples (~32 ms)
* **Rolling buffer:**

  * Always contains the last 1.5 s of audio.
  * After each inference, discard the oldest 0.4 s and append new audio frames.
* **Mono channel:** If mic is stereo, only channel 0 is used.
* **PCM format:** 16-bit (int16), no additional preprocessing required.

**Rationale:** Overlap ensures robust detection and smoothes out fluctuations. Latency is ~0.4 s per inference plus HuBERT embedding time (~50–150 ms on CPU).

---

## 3️⃣ HuBERT Embedding Pipeline

* **Model:** `facebook/hubert-base-ls960`
* **Preprocessing:**

  * Audio from rolling buffer → numpy array → feature extractor
  * Padding applied to segments shorter than ~25 ms (MIN_SAMPLES)
* **Pooling:** Mean over `last_hidden_state` dimension 1
* **Output:** 768-dim raw embeddings (no normalization)

**Consistency:** Matches training exactly, so the MLP classifier receives identical feature representations.

---

## 4️⃣ ONNX MLP Inference

* **Model:** `Model/Weak_up_model.onnx`
* **Input:** 768-dim embedding vector
* **Output:** Probability of wake word
* **Threshold:** 0.85 (slightly conservative)
* **Fallback:** If ONNX fails to load/run, can optionally fall back to `mlp_model_best.pkl` (joblib)

**Rationale:** ONNX ensures fast CPU inference; using threshold 0.85 balances false positives vs responsiveness.

---

## 5️⃣ Decision Logic & Smoothing

* **Sliding window inference:** Every 0.4 s
* **Temporal majority vote:** Require ≥2 positives within ~3 s (2 consecutive overlapping windows)
* **Trigger action:** When rule satisfied → print a line
* **Optional future enhancements:**

  * Longer smoothing for extreme robustness (3–5 windows)
  * Dynamic threshold adjustment
  * Integration with audio playback / ASR module

**Expected behavior:** Slight delay (~3 s max) but significantly reduced false positives.

---

## 6️⃣ Threading & Priority

* **PvRecorder runs in a separate high-priority thread** to prevent frame loss.
* **Main thread:** Runs inference and decision logic asynchronously.
* **CPU-only:** Designed for Arch Linux desktop, no GPU dependency.
* **No GUI constraints**, terminal-only operation.

**Rationale:** Ensures reliable, continuous audio capture without blocking inference.

---

## 7️⃣ Error Handling & Robustness

* **Overflow detection:** Monitors PvRecorder buffer; logs any dropped frames
* **Device reinitialization:** If PvRecorder fails, attempt automatic restart
* **VAD:** Not implemented (all audio windows sent for embedding)
* **Stability measures:** Rolling buffer avoids sudden jumps, smoothing reduces false triggers

---

## 8️⃣ Detection & Action

* **Current action:** Print a line to console
* **Future actions:** Can extend to:

  * Play greeting from Nisha voice library
  * Trigger ASR for commands
  * Log events or send IPC messages

**Note:** Audio capture continues while handling detection.

---

## 9️⃣ Timing & Latency Summary

| Step                        | Duration (approx CPU-only) |
| --------------------------- | -------------------------- |
| Audio frame read            | ~32 ms                     |
| Rolling buffer update       | <1 ms                      |
| HuBERT embedding extraction | 50–150 ms                  |
| MLP ONNX inference          | <1 ms                      |
| Decision logic + smoothing  | <1 ms                      |
| **End-to-end latency**      | ~0.4–0.6 s per new window  |

**Observation:** Overlapping windows provide robust detection without noticeable lag for real-time use.

---

## 🔟 Deployment Notes

* Designed for **Arch Linux desktop**, CPU-only
* Can run as terminal script; optional daemon/service integration possible
* Model update: ONNX can be replaced with new version; threshold adjustable
* Minimal resource footprint; high responsiveness maintained via overlapping windows and separate thread

---

### ✅ Summary

This blueprint ensures:

1. **Exact match to training setup** → 1.5 s window, HuBERT mean-pooled embeddings, 768-dim raw vectors
2. **Fast, reliable real-time detection** → PvRecorder + rolling buffer + ONNX MLP
3. **Reduced false positives** → threshold 0.85 + majority vote over 2 consecutive windows (~3 s)
4. **Low-latency, CPU-only operation** → designed for your Arch desktop
5. **Extensible detection action** → currently print line; future audio or ASR integration straightforward

If you want, I can also provide a **diagram showing buffer flow, window overlap, and decision logic** visually — this helps to understand how the audio moves from PvRecorder → HuBERT → MLP → output.

---



Yes — you can map your current manifest → BC-ResNet pipeline. Below I’ll walk through exactly **what will happen** and **how to turn your CSV-like table with many TTS voices into a training-ready BC-ResNet keyword-spotting dataset**, plus practical tips for the tricky bits (like when “Nisha” is in the middle of a sentence).

I’ll assume your table rows are one *sentence* and the multiple columns are parallel audio files (one per TTS voice). That means your real sample count = `rows × number_of_voice-columns` unless you collapse/merge them.

---

# End-to-end pipeline (conceptual, no code)

1. **Flatten manifest → example list**

   * Expand each row into one example per voice column (each column points to a separate `.wav`).
   * Create a table of `(wav_path, text, label)` for every voice file.
   * Now you have explicit positive and negative utterances as separate files.

2. **Decide example length / windowing strategy**

   * BC-ResNet works best on short windows (100–1000 ms typical for keyword spotting).
   * Two common choices:

     * **Utterance-level**: treat the whole TTS sentence as one sample (simple, but the wake word may be a short part of it; model sees extra context).
     * **Window-level (recommended)**: cut fixed-length windows (e.g., 600–1000 ms) that *contain* the keyword for positives and random windows for negatives.
   * If “Nisha” can appear anywhere in the sentence, window-level training is preferable (helps network focus on the phonetic cue).

3. **Locate the keyword in positive files**
   Options to get timestamps for the word “Nisha”:

   * **Best (recommended): Forced alignment (MFA or similar)**
     Use Montreal Forced Aligner / any aligner on each utterance to get token timestamps; pick the timestamp for the token “Nisha” and center a window on it.
   * **If you don’t want to align**

     * Heuristic: use `start/middle/end` info if you kept it; or split the sentence into many overlapping short windows (sliding window) and label a window positive if the original sentence label contains “Nisha”. This is noisier because many positive windows won’t actually contain the name.
   * **Alternative:** synthesize isolated “Nisha” clips from TTS (easier) and treat those as positives in addition to sentence windows.

4. **Create positive windows**

   * Using timestamps (from MFA): extract a fixed-length window (e.g., 700 ms) centered on “Nisha” (or slightly asymmetric if it’s at start/end).
   * If using sliding windows: find the window that overlaps the estimated position most and label that positive.

5. **Create negative windows**

   * Random windows from negative utterances.
   * Hard negatives: windows from positives that *don’t* contain “Nisha” (to teach the network context discrimination).
   * Phonetic negatives: phrases with phonemes similar to “Nisha” to reduce false positives.

6. **Feature extraction (BC-ResNet input)**
   For every window:

   * Convert to mono 16 kHz float32 (you already have that).
   * Compute log-Mel spectrogram (recommended params):

     * window ≈ 25 ms (400 samples), hop ≈ 10 ms (160 samples)
     * n_mels = 64 (or 80)
     * power → convert to dB (log) and optionally clamp to [-80 dB, 0 dB]
   * Normalize: per-utterance mean-variance or dataset-level stats.

7. **Training target & labels**

   * Binary label: `1` = window contains “Nisha”, `0` = otherwise.
   * Loss: binary cross-entropy. Use focal loss or class weights if imbalance exists.

8. **Model variants / training choices**

   * **Extract embeddings from BC-ResNet (freeze backbone)**

     * Pass mel through BC-ResNet, take penultimate layer / pooled feature vector, then train a small MLP (this mirrors your HuBERT→MLP flow).
     * Pros: faster, fewer params to train. Good if your dataset is small.
   * **Fine-tune BC-ResNet end-to-end**

     * Add a small classification head, train entire network (or train head + last blocks).
     * Pros: better accuracy if you have enough varied data and augmentations; cons: longer training, risk of overfitting.
   * **Hybrid**: freeze early layers, fine-tune later layers + head.

9. **Augmentation (you already added noise — good)**

   * Add: additive noises at random SNRs, room reverberation (RIR), time stretch ±5–10%, pitch shift small, random gain, SpecAugment (time/freq masks).
   * These reduce false positives in real environments.

10. **Validation / metrics**

    * Use realistic held-out data (no overlap in speaker/voice if possible).
    * Metrics to track: ROC-AUC, precision @ fixed FPR, false alarm rate per hour (important), detection latency.
    * Tune classification threshold based on desired false alarm rate vs recall.

11. **Export / runtime**

    * Export model to ONNX for low-latency inference.
    * Real-time pipeline: sliding window (e.g., 1s window or 700ms window) with hop 100–250 ms → compute mel → BC-ResNet forward → pool or classifier → threshold decision.
    * Keep mel extraction optimized (use torchaudio / C++ bindings or precompute small FFTs).

---

# Practical notes specific to your data format

* Your positive/negative CSV with many voice columns: **expand rows per voice**. Each voice column is an independent sample (so 741 rows × 8 voices → 5928 utterances). Good — extra vocal variety helps generalization.
* Because your positives are full sentences where “Nisha” may be anywhere, **forced alignment** is the cleanest next step — it gives exact start/end times for “Nisha,” which lets you extract tight keyword windows and avoid noisy labels.
* If you don’t use alignment, expect noisier supervision. Compensate with more data and heavy augmentation.
* Since TTS files are “clean” originally, your added noise helps — but include real recorded examples of “Nisha” spoken by humans if possible (that reduces sim-to-real gap).
* Balance positives and negatives per training batch. If you have many negatives, sample them so training doesn’t bias toward “no wake.”

---

# Quick recommendations (concise)

* Run **forced alignment** on every positive file and extract 600–800 ms windows centered on the aligned “Nisha”.
* Use **64-bin log-Mel** (25 ms / 10 ms hop).
* Either **freeze BC-ResNet and train an MLP** on pooled features (fast), or fine-tune last blocks + head if you have >5k varied examples.
* Use heavy augmentations (SNR 0–20 dB, RIR, SpecAugment).
* Evaluate on real-recorded clips and tune decision threshold to control false alarms per hour.

---

If you want, next I can:

* give the exact forced-alignment commands/workflow (MFA) for your dataset, or
* show how to convert your CSV into a window-level manifest (start/end timestamps + label) for training BC-ResNet.

Which of those helps you most right now?
