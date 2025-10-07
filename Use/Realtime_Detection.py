"""
Realtime HuBERT -> MLP(ONNX) wake-word detector (streaming, sliding-window)
with lightweight real-time denoising (high-pass + spectral subtraction)
Save as: Realtime_Detection_realtime_denoise.py
"""
import argparse
import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from collections import deque
import sys
import signal
import math
import scipy.signal as sps

ONNX_PATH = r"Model/mlp_model_best_int8.onnx"
HUBERT_ID = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000

DEVICE = torch.device("cpu")

# ------------------------
# Model loading & embedding
# ------------------------
def load_models(onnx_path: str, hubert_id: str, local_only=True):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    try:
        feat = Wav2Vec2FeatureExtractor.from_pretrained(hubert_id, local_files_only=local_only)
        model = HubertModel.from_pretrained(hubert_id, local_files_only=local_only)
    except Exception as e:
        if local_only:
            print("Local hubert not found; falling back to online download (remove this if you want strict offline).")
            feat = Wav2Vec2FeatureExtractor.from_pretrained(hubert_id, local_files_only=False)
            model = HubertModel.from_pretrained(hubert_id, local_files_only=False)
        else:
            raise

    model = model.to(DEVICE)
    model.eval()

    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    return sess, inp_name, feat, model

def make_embedding(feature_extractor, hubert_model, waveform: np.ndarray, sr: int):
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hubert_model(**inputs)
        hidden = outputs.last_hidden_state      # (1, T, D)
        pooled = hidden.mean(dim=1).squeeze(0)  # (D,)
        emb = pooled.cpu().numpy().astype(np.float32)[None, :]  # (1, D)
    return emb

def parse_onnx_output(raw_outputs):
    try:
        if isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) >= 1:
            label_candidate = raw_outputs[0]
            prob_candidate = raw_outputs[1] if len(raw_outputs) > 1 else None
        else:
            label_candidate = raw_outputs
            prob_candidate = None

        pred = None
        if isinstance(label_candidate, np.ndarray):
            if label_candidate.size >= 1:
                pred = int(np.array(label_candidate).flatten()[0])
        elif isinstance(label_candidate, (int, np.integer)):
            pred = int(label_candidate)

        prob = None
        if isinstance(prob_candidate, list) and len(prob_candidate) > 0:
            first = prob_candidate[0]
            if isinstance(first, dict):
                for key in (1, '1', 'pos', 'positive'):
                    if key in first:
                        prob = float(first[key]); break
                else:
                    try:
                        if 1 in first:
                            prob = float(first[1])
                        else:
                            vals = list(first.values()); prob = float(vals[0])
                    except Exception:
                        prob = None
        elif isinstance(prob_candidate, np.ndarray):
            arr = prob_candidate
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] >= 2:
                prob = float(arr[0, 1])
            elif arr.ndim == 1 and arr.size >= 2:
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[1]) if arr.size > 1 else float(arr[0])
                else:
                    exps = np.exp(arr - np.max(arr)); ps = exps / exps.sum(); prob = float(ps[1]) if arr.size > 1 else float(ps[0])
        elif isinstance(label_candidate, np.ndarray):
            arr = label_candidate
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] >= 2:
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[0, 1]); pred = int(np.argmax(arr[0]))
                else:
                    logits = arr[0]; exps = np.exp(logits - np.max(logits)); ps = exps / exps.sum(); pred = int(np.argmax(ps)); prob = float(ps[1]) if ps.size > 1 else float(ps[0])
            elif arr.ndim == 1 and arr.size >= 2:
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[1]); pred = int(np.argmax(arr))
                else:
                    exps = np.exp(arr - np.max(arr)); ps = exps / exps.sum(); pred = int(np.argmax(ps)); prob = float(ps[1]) if ps.size > 1 else float(ps[0])
        return pred, prob, raw_outputs
    except Exception as e:
        print("parse_onnx_output error:", e)
        return None, None, raw_outputs

# ------------------------
# Denoise helpers
# ------------------------
def one_pole_highpass(x: np.ndarray, sr: int, cutoff=80.0):
    """
    Simple one-pole high-pass filter implemented in time domain.
    Keeps small CPU footprint and introduces minimal latency.
    """
    if cutoff <= 0:
        return x
    dt = 1.0 / sr
    rc = 1.0 / (2 * math.pi * cutoff)
    alpha = rc / (rc + dt)
    y = np.empty_like(x)
    y_prev = 0.0
    x_prev = 0.0
    for i, xi in enumerate(x):
        yi = alpha * (y_prev + xi - x_prev)
        y[i] = yi
        y_prev = yi
        x_prev = xi
    return y

def spectral_subtract_denoise(waveform: np.ndarray, sr: int, state: dict,
                              n_fft=1024, hop=256, over_sub=1.1, floor_db=-80.0):
    """
    Performs a simple spectral subtraction denoiser.
    Maintains running noise magnitude estimate in `state['noise_mag']`.
    Returns denoised waveform and updated state.
    """
    # STFT
    noverlap = n_fft - hop
    f, t, Zxx = sps.stft(waveform, fs=sr, nperseg=n_fft, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # initialize noise estimate on first call
    if 'noise_mag' not in state or state['noise_mag'] is None:
        # initial guess: median of magnitudes across time (robust)
        state['noise_mag'] = np.median(mag, axis=1, keepdims=True)
        # also track min as conservative baseline
        state['noise_min'] = np.min(mag, axis=1, keepdims=True)
    noise_mag = state['noise_mag']

    # detect low-energy frames (likely noise-only) to update noise estimate
    frame_energy = np.sum(mag, axis=0)
    median_energy = np.median(frame_energy) + 1e-9
    low_energy_mask = frame_energy < (median_energy * 1.5)  # tweakable

    # update noise estimate using frames marked as low-energy (elementwise EMA)
    alpha = 0.95  # smoothing for noise estimate
    if np.any(low_energy_mask):
        noise_frames = mag[:, low_energy_mask]
        new_est = np.mean(noise_frames, axis=1, keepdims=True)
        noise_mag = alpha * noise_mag + (1.0 - alpha) * new_est
        state['noise_mag'] = noise_mag

    # spectral subtraction
    sub_mag = mag - over_sub * noise_mag
    # floor small values
    min_floor = 10 ** (floor_db / 20.0)
    sub_mag = np.maximum(sub_mag, min_floor)

    # rebuild complex STFT and ISTFT
    Zxx_clean = sub_mag * np.exp(1j * phase)
    _, x_rec = sps.istft(Zxx_clean, fs=sr, nperseg=n_fft, noverlap=noverlap, input_onesided=True, boundary=None)

    # match length
    if x_rec.shape[0] > waveform.shape[0]:
        x_rec = x_rec[:waveform.shape[0]]
    elif x_rec.shape[0] < waveform.shape[0]:
        x_rec = np.pad(x_rec, (waveform.shape[0] - x_rec.shape[0], 0), mode='constant')

    return x_rec.astype(np.float32), state

# ------------------------
# Audio stream callback builder
# ------------------------
def build_stream_callback(ringbuf: deque, sr: int):
    def callback(indata, frames, time_info, status):
        # if status:
            # print("Stream status:", status, file=sys.stderr)
        samples = indata[:, 0].astype(np.float32)
        ringbuf.extend(samples)
    return callback

# ------------------------
# Main loop
# ------------------------
def main(args):
    sess, inp_name, feat, hubert = load_models(args.onnx, args.hubert, local_only=args.local_hubert_only)
    print("Models loaded. ONNX input name:", inp_name)
    print("Window:", args.window_s, "s  Hop:", args.hop_s, "s  SR:", args.sr, "Threshold:", args.threshold)

    buffer_len = int(args.window_s * args.sr)
    hop_len = int(args.hop_s * args.sr)
    ringbuf = deque(maxlen=buffer_len)
    callback = build_stream_callback(ringbuf, args.sr)

    # denoise state persists between windows
    denoise_state = {'noise_mag': None, 'noise_min': None}

    stop_signal = {"stop": False}

    def handle_sigint(sig, frame):
        print("\nInterrupted. Exiting...")
        stop_signal["stop"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    with sd.InputStream(channels=1, samplerate=args.sr, blocksize=512, callback=callback):
        print("Listening... (Ctrl-C to stop)")
        last_processed = None

        while not stop_signal["stop"]:
            if len(ringbuf) < buffer_len:
                time.sleep(0.01)
                continue

            now = time.time()
            if last_processed is not None and (now - last_processed) < (args.hop_s * 0.8):
                time.sleep(0.005)
                continue

            # copy most recent buffer_len samples
            waveform = np.array(ringbuf, dtype=np.float32)
            if waveform.shape[0] < buffer_len:
                waveform = np.pad(waveform, (buffer_len - waveform.shape[0], 0), mode="constant")

            # --- DENOISING (lightweight) ---
            # 1) DC removal + tiny normalization
            waveform = waveform - np.mean(waveform)
            max_abs = max(1e-6, np.max(np.abs(waveform)))
            waveform = waveform / max_abs * 0.98  # prevent accidental clipping

            # 2) High-pass to cut low-frequency hum / fan rumble
            waveform = one_pole_highpass(waveform, args.sr, cutoff=80.0)

            # 3) Spectral subtraction denoise (operates on the whole window)
            waveform, denoise_state = spectral_subtract_denoise(waveform, args.sr, denoise_state,
                                                                n_fft=1024, hop=int(args.sr * 0.016),
                                                                over_sub=1.05, floor_db=-80.0)

            # small safety clamp
            waveform = np.clip(waveform, -0.999, 0.999)

            # Embedding + ONNX inference
            t0 = time.time()
            emb = make_embedding(feat, hubert, waveform, args.sr)   # (1, D)
            t_emb = time.time()

            # ONNX run + timing
            try:
                t_before_onnx = time.time()
                raw_outs = sess.run(None, {inp_name: emb})
                t_onnx = time.time()
            except Exception as e:
                print("ONNX inference error:", e)
                t_onnx = time.time()
                raw_outs = None

            pred, prob, raw = parse_onnx_output(raw_outs)

            # Decide detection
            detected = False
            if prob is not None:
                detected = (prob >= args.threshold)
            elif pred is not None:
                detected = (pred == 1)

            # print(f"[tdiff] emb={t_emb-t0:.3f}s onnx={(t_onnx - t_before_onnx):.3f}s total={(t_onnx - t0):.3f}s  pred={pred} prob={prob}")

            if detected:
                print(f"DETECTED at {time.strftime('%H:%M:%S')}  pred={pred} prob={prob}")
            last_processed = now
            time.sleep(0.005)

    print("Stream closed. Bye.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=ONNX_PATH)
    p.add_argument("--hubert", default=HUBERT_ID)
    p.add_argument("--sr", type=int, default=SAMPLE_RATE)
    p.add_argument("--window-s", dest="window_s", type=float, default=1.7)
    p.add_argument("--hop-s", dest="hop_s", type=float, default=0.25,
                   help="hop size (seconds) between subsequent inferences")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--exit-on-detect", dest="exit_on_detect", action="store_true")
    p.add_argument("--local-hubert-only", dest="local_hubert_only", action="store_true",
                   help="If set, fail if hubert model isn't present locally")
    args = p.parse_args()
    main(args)
