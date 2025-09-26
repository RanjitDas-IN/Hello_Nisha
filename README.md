# Nisha Wake-Up Detection Dataset & Training a CRNN
In this project i'm using format `.psv` pipe-separated file. I use a pipe (|) as a separator
---

## Overview

This project is focused on building a **wake-up word detection system** for my personal assistant, Nisha. The goal is to ensure that Nisha can **accurately detect the wake-up phrase** while ignoring all irrelevant background conversations.

## Dataset Structure

The dataset is carefully curated and split into two categories:

* **Positive Samples:**

  * Audio clips where the wake-up command (e.g., "Hello Nisha", "Hey Nisha") is spoken.
  * These examples train the model to recognize when it should respond.
  * [Positive Dataset](data/positive_mapping.psv)

* **Negative Samples:**

  * Audio clips of normal conversations that do **not** contain wake-up commands.
  * These examples help the model **ignore casual chatter** and reduce false positives.
  * [Negative Dataset](data/negative_mapping.psv)

### Example dataset Structure

```
text|voice1|voice2|voice3|label
Hello Nisha|001_Hello_Nisha.mp3|001_Hello_Nisha.mp3|001_Hello_Nisha.mp3|1
I can’t find my keys|neg_001_I_cant_find_my_keys.mp3|neg_001_I_cant_find_my_keys.mp3|neg_001.mp3|0
```

* `label=1` → Positive (wake-up)
* `label=0` → Negative (background conversation)

# Example graph for Text length distribution

![Text Length Distribution](graphs/text_length_distribution.png)


[click here to see the graph](graphs/text_length_distribution.png)

### 🔹 A big peak around 17-27 characters → means most sentences are medium-length greetings or general conversations like “Hello Nisha, how are you doing today?”

### 🔹 What the graph is?

It’s a **histogram** (with KDE curve if enabled) that shows how many of your wake-up sentences fall into different ranges of length.

* **X-axis** → the **length of the text** (in characters).

  * e.g., `"Nisha"` = 5 characters
  * `"Hello Nisha, how are you?"` ≈ 27 characters

* **Y-axis** → the **frequency** (how many sentences have that length).

* The **bars** → show counts of texts within that range.

* The **smooth KDE curve** (if shown) → estimates the probability distribution (a smoothed line over the bars).

---


# How MP3 Files Are Created for Nisha's Wake-Up System

This section explains, in simple terms, how the system generates MP3 files for Nisha, including a bit of technical workflow for context.

## 1. Overview

The system converts text sentences into MP3 audio files using multiple voices automatically, so you don't have to record each sentence manually. It uses Microsoft Edge TTS for [voice generation](Audio_generatoin/continuous_generation.py).

## 2. Step-by-Step Workflow

1. **Load all text data**

   * [The program](Audio_generatoin/continuous_generation.py) reads the dataset containing all the sentences that need to be spoken.
   * Example: `df = pd.read_csv("data/Negative_long_wakeup_dataset.psv", sep="|")`

2. **Fetch available voices**

   * Reads the [edge_tts.psv](Audio_generatoin/egde_tts.psv)
 file to get all the voices available for TTS.
   * Stores them in a list for processing.
   * Example:

     ```python
     voices_df = pd.read_csv("Audio_generatoin/egde_tts.psv", sep = "|")
     voices_list = voices_df["voices"].str.strip().tolist()
     ```

3. **Asynchronous communication with Edge TTS**

   * For each sentence and each voice, the system asynchronously sends the text to Edge TTS.
   * Edge TTS generates the audio and saves it as an MP3 file.
   * Skips any files that already exist to save time.

4. **Organize and save MP3 files**

   * Each voice has its own folder.
   * Filenames are sanitized and zero-padded for easy tracking.
   * Example filename: `001_Hello_Nisha.mp3` & `002_Weak_up_Nisha.mp3`

## 3. Analogy

Think of it like a **bakery**:

* Recipes = sentences
* Bakers = voices
* Cakes = MP3 files
* Labels on cakes = file names
* If a cake already exists, the baker skips making it.

## 4. Summary

The script automatically converts all text sentences into audio files using multiple voices, handling the process asynchronously and efficiently. This ensures that hundreds of audio clips can be generated quickly without manual recording.



## What I Need to Do Next (Step-by-step plan)

### **Step 1: Train classifier**

* Start simple: **Logistic Regression / Linear Layer / MLP** on embeddings.
* Input: 768-dim HuBERT embedding.
* Output: probability of “wake word present” (sigmoid → binary cross-entropy loss).

---

### **Step 2: Validate**

* Split manifest into **train/val** (e.g., 80/20 by file, so same file doesn’t leak into both).
* Train classifier, check accuracy/F1.
* Tune threshold (e.g., 0.5 → 0.7) to reduce false positives.

---

### **Step 3: Inference (real usage)**

* Take streaming audio (1–7s or longer).
* Apply same windowing (0.8s with 0.4 hop).
* For each window:

  * Extract HuBERT embedding.
  * Run classifier → probability.
* If several consecutive windows exceed threshold → **detect “Nisha”**.

---

### **Optional next steps**

* **Data augmentation**: add background noise, reverb, pitch shifts to improve robustness.
* **Model upgrade**: instead of frozen HuBERT + classifier, try fine-tuning HuBERT itself (if GPU resources allow).
* **Sequence modeling**: add a tiny CNN or RNN on top of embeddings to capture context across overlapping windows.

---