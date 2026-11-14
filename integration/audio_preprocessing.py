import numpy as np
import librosa


# ==========================================
# AUDIO CONFIGURATION
# ==========================================

SAMPLE_RATE = 16000
TARGET_DURATION = 8
TARGET_LENGTH = SAMPLE_RATE * TARGET_DURATION


# ==========================================
# LOG-MEL CONFIGURATION
# ==========================================

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 20
FMAX = 7600


# ==========================================
# AUDIO FEATURE EXTRACTION
# ==========================================

def extract_logmel_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Pad or truncate to 8 seconds
    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )

    else:

        audio = audio[:TARGET_LENGTH]

    # Generate Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # Convert to Log-Mel
    log_mel = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    return log_mel.astype(np.float32)


print("Audio preprocessing module loaded successfully!")