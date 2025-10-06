import os
import numpy as np
import librosa
import soundfile as sf
import math
import random

# -----------------------
# Helper functions
# -----------------------
def lowpass_onepole(x, sr, cutoff):
    if cutoff <= 0 or cutoff >= sr/2:
        return x
    dt = 1.0 / sr
    RC = 1.0 / (2 * math.pi * cutoff)
    alpha = dt / (RC + dt)
    y = np.zeros_like(x)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = y[n-1] + alpha * (x[n] - y[n-1])
    return y

def bandshape_noise(noise, sr, lowcut=200.0, highcut=5000.0):
    # Lowpass at highcut
    lp = lowpass_onepole(noise, sr, highcut)
    # Lowpass at lowcut then subtract -> bandpass approx
    lp_low = lowpass_onepole(noise, sr, lowcut)
    band = lp - lp_low
    t = np.linspace(0, len(noise)/sr, len(noise), endpoint=False)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)  # slow 0.25Hz modulation
    return band * mod

def rms_of(x):
    if x.size == 0:
        return 0.0
    return np.sqrt(np.mean(x**2))

def bitcrush_chunk(chunk, factor):
    if factor <= 1:
        return chunk
    down = chunk[::factor]
    if down.size == 0:
        return chunk
    up = np.repeat(down, factor)
    return up[:len(chunk)]

# -----------------------
# Parameters
# -----------------------
input_dir = r"corpus_for_mfa/wav_16k_negative_wav_lab"
sr_target = 16000

# Pitch
pitch_voiced_semitones = +2.0
pitch_gap_semitones = -2.0

# Voiced detection
frame_length = 1024
hop_length = 512
voiced_percentile = 60

# Chunk
chunk_size = 4096

# Echo
delay_seconds = [0.05, 0.12]
gains = [0.55, 0.28]
lowpass_cutoff_hz = 3000

# Noise/BG
noise_factor = 0.0001  # user requested (base randomness level)
bg_lowcut = 30.0
bg_highcut = 500.0
bg_target_ratio = 0.1  # background will be 30% of amplified words RMS

# loudness: only words
loudness_gain = 20.0

# Broken artifacts
dropout_prob = 0.025
dropout_max_ms = 80
bitcrush_prob = 0.12
bitcrush_factor_min = 2
bitcrush_factor_max = 8
crackle_prob = 0.02
crackle_max_clicks = 6
crackle_amp = 0.25
flutter_prob = 0.08
flutter_semitones = 0.08
flutter_rate_hz = 6.0

# -----------------------
# Process single file
# -----------------------
def process_file(audio_path):
    print(f"Processing: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr_target)
    y = y.astype(np.float32)
    n = len(y)
    duration = n / sr

    # -----------------------
    # Voiced detection
    # -----------------------
    frames_energy = np.array([np.sum(np.abs(y[i:i+frame_length])**2)
                              for i in range(0, n, hop_length)], dtype=np.float64)
    if frames_energy.max() == 0:
        frames_energy += 1e-8

    threshold = np.percentile(frames_energy, voiced_percentile)
    voiced_frames = (frames_energy >= threshold).astype(np.float32)

    smooth_win = np.ones(5)
    voiced_smoothed = np.convolve(voiced_frames, smooth_win, mode='same')
    voiced_smoothed = (voiced_smoothed > 0.5).astype(np.float32)
    voiced_mask = np.repeat(voiced_smoothed, hop_length)[:n]

    # smoothing envelope for gain application
    env_win_ms = 20
    env_win = int(env_win_ms * sr / 1000)
    if env_win < 3:
        env = np.array([1.0])
    else:
        t_env = np.linspace(-math.pi/2, math.pi/2, env_win)
        env = (np.sin(t_env) + 1.0) / 2.0
    voiced_mask_smoothed = np.convolve(voiced_mask, env/np.sum(env), mode='same')
    voiced_mask_smoothed = np.clip(voiced_mask_smoothed, 0.0, 1.0)

    # -----------------------
    # Pitch shifting
    # -----------------------
    semitone_per_frame = np.where(voiced_smoothed >= 0.5, pitch_voiced_semitones, pitch_gap_semitones)
    semitone_per_sample = np.repeat(semitone_per_frame, hop_length)[:n]
    smooth_samples = int(0.05 * sr)
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        semitone_per_sample = np.convolve(semitone_per_sample, kernel, mode='same')

    processed = np.zeros_like(y, dtype=np.float32)
    for i in range(0, n, chunk_size):
        chunk = y[i:i+chunk_size]
        if chunk.size == 0:
            continue
        avg_semitone = float(np.mean(semitone_per_sample[i:i+len(chunk)]))
        shifted = librosa.effects.pitch_shift(y=chunk, sr=sr, n_steps=avg_semitone)
        if shifted.shape[0] > chunk.shape[0]:
            shifted = shifted[:chunk.shape[0]]
        elif shifted.shape[0] < chunk.shape[0]:
            shifted = np.pad(shifted, (0, chunk.shape[0] - shifted.shape[0]))
        processed[i:i+len(chunk)] = shifted

    # -----------------------
    # Echo
    # -----------------------
    filtered = lowpass_onepole(processed, sr, lowpass_cutoff_hz)
    echo_mix = np.zeros_like(processed)
    for delay_s, g in zip(delay_seconds, gains):
        delay_samples = int(delay_s * sr)
        echo = np.pad(filtered * g, (delay_samples, 0))[:n]
        echo_mix += echo
    direct_gain = 1.0
    processed = processed * direct_gain + echo_mix

    # -----------------------
    # Broken artifacts
    # -----------------------
    # Dropouts
    num_dropout_trials = max(1, int(duration * dropout_prob * 4))
    for _ in range(num_dropout_trials):
        if random.random() < dropout_prob * 4:
            start = random.randint(0, n-1)
            length_ms = random.randint(6, dropout_max_ms)
            length = int(length_ms * sr / 1000)
            end = min(n, start + length)
            if end > start:
                processed[start:end] *= np.linspace(1.0, 0.0, end-start)

    # Bitcrush
    for i in range(0, n, chunk_size):
        if random.random() < bitcrush_prob:
            factor = random.randint(bitcrush_factor_min, bitcrush_factor_max)
            chunk = processed[i:i+chunk_size]
            crushed = bitcrush_chunk(chunk, factor)
            if crushed.shape[0] < chunk.shape[0]:
                crushed = np.pad(crushed, (0, chunk.shape[0] - crushed.shape[0]))
            processed[i:i+chunk_size] = (1.0 - 0.7 * (factor/bitcrush_factor_max)) * processed[i:i+chunk_size] + 0.7 * crushed

    # Crackle clicks
    num_crackle_trials = int(duration * crackle_prob * 10)
    for _ in range(num_crackle_trials):
        if random.random() < crackle_prob * 10:
            pos = random.randint(0, n-1)
            clicks = random.randint(1, crackle_max_clicks)
            for c in range(clicks):
                off = pos + random.randint(0, min(400, n-pos-1))
                if off < n:
                    processed[off] += (np.random.uniform(-1, 1) * crackle_amp)
    processed = lowpass_onepole(processed, sr, cutoff=sr/2 * 0.98)

    # -----------------------
    # Loudness only on voiced words
    # -----------------------
    mult_env = 1.0 + (loudness_gain - 1.0) * voiced_mask_smoothed
    amplified_words = processed * mult_env
    voiced_samples = amplified_words[voiced_mask_smoothed > 0.01]
    voiced_rms = rms_of(voiced_samples) if voiced_samples.size > 0 else rms_of(amplified_words)

    # -----------------------
    # Generate room-like BG noise
    # -----------------------
    bg_noise = np.random.normal(0.0, 1.0, n) * noise_factor
    bg_noise = bandshape_noise(bg_noise, sr, lowcut=bg_lowcut, highcut=bg_highcut)
    target_bg_rms = voiced_rms * bg_target_ratio
    current_bg_rms = rms_of(bg_noise)
    if current_bg_rms > 0:
        bg_noise = bg_noise * (target_bg_rms / (current_bg_rms + 1e-12))
    bg_noise = lowpass_onepole(bg_noise, sr, cutoff=3000.0)
    duck_amount = 0.6
    bg_noise = bg_noise * (1.0 - duck_amount * voiced_mask_smoothed)

    # -----------------------
    # Mix words + BG
    # -----------------------
    final = amplified_words + bg_noise

    # -----------------------
    # Soft clipping + normalization
    # -----------------------
    final = np.tanh(final * 2.0)
    max_abs = np.max(np.abs(final))
    if max_abs > 0:
        final = final / max_abs * 0.98

    # Save (overwrite original file)
    final = final.astype(np.float32)
    sf.write(audio_path, final, sr)

# -----------------------
# Walk directory and process all .wav files
# -----------------------
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            process_file(full_path)
