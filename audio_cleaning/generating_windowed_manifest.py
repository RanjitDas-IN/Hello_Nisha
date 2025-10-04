#!/usr/bin/env python3
import os
import random
import numpy as np
import librosa
import soundfile as sf
from praatio import tgio

# === CONFIG ===
CORPUS_ROOT = "corpus_for_mfa"
ALIGNED_ROOT = "corpus_for_mfa/aligned"
OUT_DIR = "corpus_for_mfa/windowed"
MANIFEST = os.path.join(r"data", "window_manifest.psv")

WAKEWORD = "NISHA"            # word to consider positive (case-insensitive)
WINDOW_S = 1.5                # window length in seconds
NEG_PER_FILE = 2              # how many negatives to sample per file with positives
SAMPLE_RATE = 16000           # expected audio sample rate

os.makedirs(OUT_DIR, exist_ok=True)

def safe_makedirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def get_textgrid_path(wav_path):
    # assumes aligned dir mirrors corpus structure and .TextGrid extension
    rel = os.path.relpath(wav_path, CORPUS_ROOT)
    base = os.path.splitext(rel)[0]
    tg_path = os.path.join(ALIGNED_ROOT, base + ".TextGrid")
    return tg_path

def clamp(a, lo, hi):
    return max(lo, min(a, hi))

manifest_lines = []
count_pos = count_neg = 0

for root, _, files in os.walk(CORPUS_ROOT):
    for fname in files:
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(root, fname)
        tg_path = get_textgrid_path(wav_path)
        if not os.path.exists(tg_path):
            # skip if no alignment exists
            continue

        # load audio length
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        audio_dur = len(y) / sr

        tg = tgio.openTextgrid(tg_path)
        # get words tier (case-insensitive)
        try:
            words_tier = tg.tierDict["words"]
        except KeyError:
            continue

        # collect positive intervals (where text equals WAKEWORD)
        pos_intervals = []
        for (start, end, label) in words_tier.entryList:

            lab = label.strip()
            if lab.upper() == WAKEWORD.upper():
                pos_intervals.append((start, end))

        # create positive windows
        for (s, e) in pos_intervals:
            center = (s + e) / 2.0
            wstart = clamp(center - WINDOW_S/2.0, 0.0, audio_dur - WINDOW_S)
            wend = wstart + WINDOW_S
            # extract audio
            start_sample = int(round(wstart * sr))
            end_sample = int(round(wend * sr))
            seg = y[start_sample:end_sample]
            out_rel = os.path.relpath(wav_path, CORPUS_ROOT)
            out_rel = out_rel.replace(os.sep, "_")
            out_name = f"{os.path.splitext(out_rel)[0]}_{int(wstart*1000)}_{int(wend*1000)}_pos.wav"
            out_path = os.path.join(OUT_DIR, out_name)
            sf.write(out_path, seg, sr)
            manifest_lines.append(f"{out_path}|{wstart:.3f}|{wend:.3f}|1")
            count_pos += 1

        # sample negatives: random windows that don't overlap any pos interval
        # make a list of banned regions (pos +/- small margin)
        banned = []
        margin = 0.01
        for (s, e) in pos_intervals:
            banned.append((max(0.0, s - margin), min(audio_dur, e + margin)))

        # candidate start range: [0, audio_dur - WINDOW_S]
        max_start = max(0.0, audio_dur - WINDOW_S)
        tries = 0
        neg_created = 0
        while neg_created < max(0, NEG_PER_FILE * max(1, len(pos_intervals))) and tries < 200:
            tries += 1
            s_try = random.random() * max_start
            e_try = s_try + WINDOW_S
            overlap = False
            for (b0, b1) in banned:
                if not (e_try <= b0 or s_try >= b1):
                    overlap = True
                    break
            if overlap:
                continue
            # accept
            start_sample = int(round(s_try * sr))
            end_sample = int(round(e_try * sr))
            seg = y[start_sample:end_sample]
            out_rel = os.path.relpath(wav_path, CORPUS_ROOT)
            out_rel = out_rel.replace(os.sep, "_")
            out_name = f"{os.path.splitext(out_rel)[0]}_{int(s_try*1000)}_{int(e_try*1000)}_neg.wav"
            out_path = os.path.join(OUT_DIR, out_name)
            sf.write(out_path, seg, sr)
            manifest_lines.append(f"{out_path}|{s_try:.3f}|{e_try:.3f}|0")
            neg_created += 1
            count_neg += 1

# write manifest (pipe-separated)
with open(MANIFEST, "w", encoding="utf-8") as mf:
    mf.write("wav_path|window_start|window_end|label\n")
    for line in manifest_lines:
        mf.write(line + "\n")

print(f"Done. Pos: {count_pos}, Neg: {count_neg}, manifest: {MANIFEST}")
