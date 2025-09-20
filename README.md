# Nisha Wake-Up Detection Dataset & Training a CRNN
In this project i use my own format `.psv` pipe-separated file. I use a pipe (|) as a separator
---

## Overview

This project is focused on building a **wake-up word detection system** for my personal assistant, Nisha. The goal is to ensure that Nisha can **accurately detect the wake-up phrase** while ignoring all irrelevant background conversations.

## Dataset Structure

The dataset is carefully curated and split into two categories:

* **Positive Samples:**

  * Audio clips where the wake-up command (e.g., "Hello Nisha", "Hey Nisha") is spoken.
  * These examples train the model to recognize when it should respond.
  * [Positive Dataset](data/positive_mapping.csv)

* **Negative Samples:**

  * Audio clips of normal conversations that do **not** contain wake-up commands.
  * These examples help the model **ignore casual chatter** and reduce false positives.
  * [Negative Dataset](data/negative_mapping.csv)

### Example CSV Structure

```
text|voice1|voice2|voice3|label
Hello Nisha|001_Hello_Nisha.mp3|001_Hello_Nisha.mp3|001_Hello_Nisha.mp3|1
I can’t find my keys|neg_001_I_cant_find_my_keys.mp3|neg_001_I_cant_find_my_keys.mp3|neg_001.mp3|0
```

* `label=1` → Positive (wake-up)
* `label=0` → Negative (background conversation)
---


# How MP3 Files Are Created for Nisha's Wake-Up System

This section explains, in simple terms, how the system generates MP3 files for Nisha, including a bit of technical workflow for context.

## 1. Overview

The system converts text sentences into MP3 audio files using multiple voices automatically, so you don't have to record each sentence manually. It uses Microsoft Edge TTS for voice generation.

## 2. Step-by-Step Workflow

1. **Load all text data**

   * The program reads the dataset containing all the sentences that need to be spoken.
   * Example: `df = pd.read_csv("data/Negative_long_wakeup_dataset.csv", sep="|")`

2. **Fetch available voices**

   * Reads the [edge_tts.csv](Audio_generatoin/egde_tts.csv)
 file to get all the voices available for TTS.
   * Stores them in a list for processing.
   * Example:

     ```python
     voices_df = pd.read_csv("Audio_generatoin/egde_tts.csv")
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




## Training Objective

The model aims to:

1. **Wake up reliably** when the trigger phrase is spoken.
2. **Ignore irrelevant conversations** or background noise.
3. Maintain **low latency** for instant responsiveness.
4. Achieve **high accuracy**, minimizing both false positives and false negatives.

## Best Practices I found

1. **Keep Positive & Negative Samples Separate:**

   * Map each text example to its corresponding MP3 file before merging.
   * Ensure positive and negative samples are labeled correctly.

2. **Add a Label Column:**

   * Positive samples: `label=1`
   * Negative samples: `label=0`

3. **Merge Carefully for Training:**

   * Once both CSVs are labeled, merge into a single training file.
   * Preserve mapping of each text to its respective MP3 files.

4. **Maintain Diversity:**

   * Use a mix of short and long sentences.
   * Include varied conversational tones in negative samples.

---

## Getting Started

1. Prepare your positive and negative CSVs with MP3 mappings.
2. Add the `label` column for each.
3. Merge them carefully into a single CSV if desired.
4. Train your wake-up detection model using this labeled dataset.

This setup ensures that Nisha can **respond only to intended wake-up phrases** and ignore irrelevant chatter, making her highly accurate and robust in real-world environments.
