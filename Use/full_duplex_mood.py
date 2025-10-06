# full_duplex_mood_updated.py
# Full-duplex playback + mic listening with NLMS echo cancellation.
# Now accepts a custom audio file path: FullDuplexPlayer("my.wav")

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading

# ---------------- DEFAULT CONFIG (can be overridden per-call) ----------------
DEFAULT_MIC_DEVICE_INDEX = 0    # None = system default input
DEFAULT_OUTPUT_DEVICE_INDEX = 7 # None = system default output
DEFAULT_THRESHOLD = 1500           # detection threshold on residual RMS (tune)
DEFAULT_BLOCKSIZE = 100            # lower -> lower latency
DEFAULT_LISTEN_DELAY = 1.5         # seconds to wait before enabling detection

# NLMS adaptive filter default params (echo canceller)
DEFAULT_FILTER_LEN = 512           # number of taps in adaptive filter (tune: 256..1024)
DEFAULT_MU = 0.2                   # NLMS step size (tune 0.05..0.5)
DEFAULT_EPS = 1e-8                 # small constant to avoid divide-by-zero
# -----------------------------------------------------------------------------

# helper RMS
def rms_flat(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
# tblfm

def FullDuplexPlayer(audio_file_path: str,
                     mic_device_index=DEFAULT_MIC_DEVICE_INDEX,
                     output_device_index=DEFAULT_OUTPUT_DEVICE_INDEX,
                     threshold=DEFAULT_THRESHOLD,
                     blocksize=DEFAULT_BLOCKSIZE,
                     listen_delay=DEFAULT_LISTEN_DELAY,
                     filter_len=DEFAULT_FILTER_LEN,
                     mu=DEFAULT_MU,
                     eps=DEFAULT_EPS):
    """
    Play `audio_file_path` while listening to the mic with NLMS echo cancellation.
    Stops playback and exits if residual RMS (speech) exceeds `threshold`.
    """

    # Open the WAV (keeps the SoundFile object alive while the stream runs)
    wav = sf.SoundFile(audio_file_path, 'r')

    # Local state and synchronization
    stop_flag = threading.Event()
    lock = threading.Lock()

    # Adaptive filter state (mutable arrays)
    w = np.zeros(filter_len, dtype=np.float32)
    ref_buffer = np.zeros(filter_len + blocksize, dtype=np.float32)
    listen_start_time = time.time() + listen_delay

    print(f"Starting playback of: {audio_file_path}")
    print("Listening (with echo cancellation) starts after", listen_delay, "s")

    # callback closure uses the local wav, w, ref_buffer, etc.
    def callback(indata, outdata, frames, time_info, status):
        nonlocal ref_buffer, w

        if status:
            # occasional status (underflow, etc.)
            print("[Stream status]", status)
            

        # default silence
        outdata.fill(0)

        # read audio to play
        with lock:
            data = wav.read(frames, dtype='float32', always_2d=True)

        if data.shape[0] == 0:
            # EOF: stop playback gracefully
            stop_flag.set()
            return

        # place playback into outdata (handle mono->stereo)
        out_ch = outdata.shape[1]
        file_ch = data.shape[1]
        n_copy = data.shape[0]
        if file_ch == out_ch:
            outdata[:n_copy, :] = data
        elif file_ch == 1 and out_ch >= 2:
            # duplicate mono into stereo (or more)
            for c in range(out_ch):
                outdata[:n_copy, c] = data[:, 0]
        else:
            outdata[:n_copy, :file_ch] = data

        # Build reference signal (mono) from playback data
        if data.shape[1] == 1:
            ref_block = data[:, 0].astype(np.float32)
        else:
            ref_block = np.mean(data, axis=1).astype(np.float32)

        # update ref_buffer as a ring (simple roll and append)
        rb_len = ref_buffer.shape[0]
        if n_copy <= rb_len:
            ref_buffer = np.roll(ref_buffer, -n_copy)
            ref_buffer[-n_copy:] = ref_block
        else:
            # block larger than buffer (unlikely)
            ref_buffer = np.concatenate((ref_buffer[-filter_len:], ref_block[-filter_len:]))
            ref_buffer = ref_buffer[-rb_len:]

        # If listening not enabled yet, skip AEC/RMS
        if time.time() < listen_start_time:
            return

        # Ensure we have mic data
        if indata is None or indata.size == 0:
            return

        mic = indata
        if mic.ndim > 1:
            mic_mono = np.mean(mic, axis=1).astype(np.float32)
        else:
            mic_mono = mic.astype(np.float32)

        # NLMS update sample-wise (simple implementation)
        # For each sample n in block, construct reference vector x and update w.
        for n in range(n_copy):
            # reference vector x: most recent FILTER_LEN samples aligned to sample n
            # x selection logic mirrors the original implementation
            start = -filter_len - (n_copy - 1 - n)
            end = - (n_copy - 1 - n) if (n_copy - 1 - n) != 0 else None
            x = ref_buffer[start:end]
            if x.shape[0] != filter_len:
                if x.shape[0] < filter_len:
                    x = np.concatenate((np.zeros(filter_len - x.shape[0], dtype=np.float32), x))
                else:
                    x = x[-filter_len:]
            y_hat = float(np.dot(w, x))
            e = float(mic_mono[n] - y_hat)
            norm_x = float(np.dot(x, x)) + eps
            w += (mu * e * x) / norm_x

        # Compute residual for the block via convolution
        recent_ref = ref_buffer[-(filter_len + n_copy - 1):]
        conv_full = np.convolve(recent_ref, w, mode='valid')  # should yield length n_copy
        echo_est = conv_full[-n_copy:]
        residual = mic_mono[:n_copy] - echo_est[:n_copy]

        residual_rms = rms_flat(residual)

        if residual_rms >= threshold:
            print(f"[Residual Interrupt] RMS={residual_rms:.5f} -> stopping playback!")
            stop_flag.set()
            return

    # Build device tuple (sounddevice accepts (input, output), None = default)
    device = (mic_device_index, output_device_index)

    try:
        with sd.Stream(samplerate=wav.samplerate,
                       blocksize=blocksize,
                       dtype='float32',
                       channels=(1, wav.channels),
                       callback=callback,
                       device=device,
                       latency='low'):
            while not stop_flag.is_set():
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        wav.close()
        print("Done.")

# Example usage (uncomment to run directly)
if __name__ == "__main__":
    # Replace "my.wav" with your file. If you want default devices, pass None for mic/output indices.
    FullDuplexPlayer(r"/home/ranjit/Desktop/projects/Hello_Nisha/Hello_Nisha.wav")