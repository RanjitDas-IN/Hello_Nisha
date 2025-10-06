"""
Realtime HuBERT -> MLP(ONNX) wake-word detector (streaming, sliding-window)
Save as: Realtime_Detection_realtime.py
"""
import argparse
import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import torch
from Use.full_duplex_mood import FullDuplexPlayer
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from collections import deque
import sys
import signal

ONNX_PATH = r"Model/mlp_model_best_int8.onnx"
HUBERT_ID = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000

DEVICE = torch.device("cpu")

def load_models(onnx_path: str, hubert_id: str, local_only=True):
    # ONNX session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # Feature extractor + HuBERT
    try:
        feat = Wav2Vec2FeatureExtractor.from_pretrained(hubert_id, local_files_only=local_only)
        model = HubertModel.from_pretrained(hubert_id, local_files_only=local_only)
    except Exception as e:
        # If local_files_only caused an error, fall back to download
        if local_only:
            print("Local hubert not found; falling back to online download (remove this if you want strict offline).")
            feat = Wav2Vec2FeatureExtractor.from_pretrained(hubert_id, local_files_only=False)
            model = HubertModel.from_pretrained(hubert_id, local_files_only=False)
        else:
            raise

    model = model.to(DEVICE)
    model.eval()
    
    
        # --- debug prints right after InferenceSession created (or right after load_models returns sess) ---
    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])



    
    
    return sess, inp_name, feat, model

def make_embedding(feature_extractor, hubert_model, waveform: np.ndarray, sr: int):
    """
    waveform: 1-d numpy float32, expected sr sampling rate
    returns: numpy array shape (1, D) float32 ready for ONNX
    """
    # feature_extractor expects Python list or numpy and returns tensors
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hubert_model(**inputs)
        hidden = outputs.last_hidden_state      # (1, T, D)
        pooled = hidden.mean(dim=1).squeeze(0)  # (D,)
        emb = pooled.cpu().numpy().astype(np.float32)[None, :]  # (1, D)
    return emb
# Add this function here, after other helpers
def _summarize(o):
    if isinstance(o, np.ndarray):
        return f"ndarray shape={o.shape} dtype={o.dtype} min={np.min(o):.4f} max={np.max(o):.4f}"
    if isinstance(o, (list, tuple)):
        return f"{type(o)} len={len(o)}"
    return f"{type(o)} {str(o)[:200]}"
def parse_onnx_output(raw_outputs):
    """
    Return (pred:int|None, prob:float|None, raw_outputs)
    Handles common ONNX output formats, including:
      - [ndarray(label), list([{0:..., 1:...}])]  <-- your case
      - [ndarray(label), ndarray([[p0,p1]])]
      - single ndarray logits/probs, scalars, dicts, etc.
    """
    try:
        # canonicalize raw_outputs to a sequence if possible
        if isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) >= 1:
            # prefer first element as label, second as probabilities if present
            label_candidate = raw_outputs[0]
            prob_candidate = raw_outputs[1] if len(raw_outputs) > 1 else None
        else:
            label_candidate = raw_outputs
            prob_candidate = None

        # extract predicted label if possible
        pred = None
        if isinstance(label_candidate, np.ndarray):
            if label_candidate.size >= 1:
                pred = int(np.array(label_candidate).flatten()[0])
        elif isinstance(label_candidate, (int, np.integer)):
            pred = int(label_candidate)

        # extract probability
        prob = None
        # Case A: second output is a Python list containing a dict [{0:...,1:...}]
        if isinstance(prob_candidate, list) and len(prob_candidate) > 0:
            first = prob_candidate[0]
            if isinstance(first, dict):
                # try keys 1 or '1' else fallback to key for positive class
                for key in (1, '1', 'pos', 'positive'):
                    if key in first:
                        prob = float(first[key])
                        break
                else:
                    # try the numeric largest key or key '1' fallback
                    try:
                        # pick the value for key 1 if present, else the max value
                        if 1 in first:
                            prob = float(first[1])
                        else:
                            # choose value with largest key or largest value
                            vals = list(first.values())
                            prob = float(vals[0])
                    except Exception:
                        prob = None

        # Case B: second output is an ndarray of shape (1,2) or (2,)
        elif isinstance(prob_candidate, np.ndarray):
            arr = prob_candidate
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] >= 2:
                prob = float(arr[0, 1])  # positive class prob
            elif arr.ndim == 1 and arr.size >= 2:
                # if arr contains probabilities or logits — if sum<=1 treat as probs else softmax
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[1]) if arr.size > 1 else float(arr[0])
                else:
                    exps = np.exp(arr - np.max(arr))
                    ps = exps / exps.sum()
                    prob = float(ps[1]) if arr.size > 1 else float(ps[0])

        # Case C: no prob candidate but label may be present and model is single-output logits/probs
        elif isinstance(label_candidate, np.ndarray):
            arr = label_candidate
            if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] >= 2:
                # treat as logits/probs
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[0, 1])
                    pred = int(np.argmax(arr[0]))
                else:
                    logits = arr[0]
                    exps = np.exp(logits - np.max(logits))
                    ps = exps / exps.sum()
                    pred = int(np.argmax(ps))
                    prob = float(ps[1]) if ps.size > 1 else float(ps[0])
            elif arr.ndim == 1 and arr.size >= 2:
                # 1-D logits/probs
                if np.all((arr >= 0) & (arr <= 1)) and abs(arr.sum() - 1.0) < 1e-3:
                    prob = float(arr[1])
                    pred = int(np.argmax(arr))
                else:
                    exps = np.exp(arr - np.max(arr))
                    ps = exps / exps.sum()
                    pred = int(np.argmax(ps))
                    prob = float(ps[1]) if ps.size > 1 else float(ps[0])

        # final fallback: if we have a pred but no prob, leave prob None
        return pred, prob, raw_outputs

    except Exception as e:
        print("parse_onnx_output error:", e)
        return None, None, raw_outputs

def build_stream_callback(ringbuf: deque, sr: int):
    def callback(indata, frames, time_info, status):
        if status:
            print("Stream status:", status, file=sys.stderr)
        # indata shape (frames, channels). We'll take first channel.
        samples = indata[:, 0].astype(np.float32)
        ringbuf.extend(samples)
    return callback


def main(args):
    sess, inp_name, feat, hubert = load_models(args.onnx, args.hubert, local_only=args.local_hubert_only)
    print("Models loaded. ONNX input name:", inp_name)
    print("Window:", args.window_s, "s  Hop:", args.hop_s, "s  SR:", args.sr, "Threshold:", args.threshold)

    buffer_len = int(args.window_s * args.sr)
    hop_len = int(args.hop_s * args.sr)
    ringbuf = deque(maxlen=buffer_len)
    callback = build_stream_callback(ringbuf, args.sr)

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

            # Decide detection: prefer probability threshold when available, else predicted class==1
            detected = False
            if prob is not None:
                detected = (prob >= args.threshold)
            elif pred is not None:
                detected = (pred == 1)
# -------------------------------------------------------------------------------------------------------------------------------------------------
            # print(f"[tdiff] emb={t_emb-t0:.3f}s onnx={(t_onnx - t_before_onnx):.3f}s total={(t_onnx - t0):.3f}s  pred={pred} prob={prob}")
# -------------------------------------------------------------------------------------------------------------------------------------------------

            if detected:
                print(f"DETECTED at {time.strftime('%H:%M:%S')}  pred={pred} prob={prob}")
                FullDuplexPlayer()
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
