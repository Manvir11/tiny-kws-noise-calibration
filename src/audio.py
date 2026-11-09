"""Audio mixing and the MLPerf Tiny keyword spotting MFCC front end.

The MFCC parameters and transform follow MLPerf Tiny's get_dataset.py at
commit 4addd0fa08d216e20637637874e084895f289da4 (Apache-2.0).
"""

import ast
from pathlib import Path
import wave

import numpy as np
import tensorflow as tf


SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
FEATURE_SHAPE = (49, 10, 1)
FRAME_LENGTH = 480
FRAME_STEP = 320
FFT_LENGTH = 512


def _read_pcm16(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"Expected 16 kHz mono 16-bit PCM: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if not samples.size:
        raise ValueError(f"Empty WAV file: {path}")
    return samples


def read_clip(path):
    """Decode a clip, truncating long clips and zero-padding short clips."""
    samples = _read_pcm16(path)[:CLIP_SAMPLES].astype(np.float32) / 32768.0
    return np.pad(samples, (0, CLIP_SAMPLES - len(samples)))


def read_noise(path):
    """Decode one mono channel of an environmental noise recording."""
    samples = _read_pcm16(path)
    if len(samples) < CLIP_SAMPLES:
        raise ValueError(f"Noise recording is shorter than one second: {path}")
    return samples.astype(np.float32) / 32768.0


def mix_at_snr(speech, noise, snr_db):
    """Mix one second of speech and noise at the requested full-clip RMS SNR.

    Return the float32 waveform, achieved SNR in dB, and the common gain
    applied to both components to avoid clipping.
    """
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if speech.shape != (CLIP_SAMPLES,) or noise.shape != (CLIP_SAMPLES,):
        raise ValueError("Mixing requires two one-second mono arrays")
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    speech_rms = np.sqrt(np.mean(np.square(speech)))
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    if not np.isfinite(speech_rms + noise_rms) or min(speech_rms, noise_rms) < 1e-10:
        raise ValueError("Speech and noise must both have finite nonzero RMS")

    noise = noise * (speech_rms / noise_rms) * 10.0 ** (-snr_db / 20.0)
    mixed = speech + noise
    peak = np.max(np.abs(mixed))
    common_gain = 0.99 / peak if peak > 0.99 else 1.0
    realized_snr = 20 * np.log10(
        (speech_rms * common_gain) / np.sqrt(np.mean(np.square(noise * common_gain)))
    )
    return (mixed * common_gain).astype(np.float32), float(realized_snr), float(common_gain)


@tf.function(input_signature=[tf.TensorSpec([None, CLIP_SAMPLES], tf.float32)])
def _batch_mfcc(waveforms):
    positive_peak = tf.reduce_max(waveforms, axis=-1, keepdims=True)
    # Ordinary clips follow the upstream positive-peak normalization. For
    # nonpositive clips, use absolute peak and map digital silence to zero.
    peak = tf.where(
        positive_peak > 0.0,
        positive_peak,
        tf.reduce_max(tf.abs(waveforms), axis=-1, keepdims=True),
    )
    normalized = tf.math.divide_no_nan(waveforms, peak)
    spectrum = tf.abs(
        tf.signal.stft(
            normalized,
            frame_length=FRAME_LENGTH,
            frame_step=FRAME_STEP,
            fft_length=FFT_LENGTH,
            window_fn=tf.signal.hann_window,
        )
    )
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        40, FFT_LENGTH // 2 + 1, SAMPLE_RATE, 20.0, 4000.0
    )
    log_mel = tf.math.log(tf.tensordot(spectrum, mel_weights, axes=1) + 1e-6)
    coefficients = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :10]
    return coefficients[..., tf.newaxis]


def extract_mfcc(waveforms):
    """Return float32 MFCCs with shape (clips, 49, 10, 1)."""
    waveforms = np.atleast_2d(np.asarray(waveforms, dtype=np.float32))
    if waveforms.ndim != 2 or waveforms.shape[1] != CLIP_SAMPLES or not len(waveforms):
        raise ValueError("Expected one or more arrays of 16000 audio samples")
    if not np.isfinite(waveforms).all():
        raise ValueError("MFCC input must be finite")
    return _batch_mfcc(waveforms).numpy()


def check_upstream_parity(paths, upstream_path):
    """Compare WAV features against MLPerf Tiny's preprocessing function."""
    source = Path(upstream_path).read_text()
    original = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_preprocess_audio_func"
    )
    # The MFCC inference path needs only TensorFlow. Evaluate that definition
    # alone to avoid loading the upstream training and dataset dependencies.
    namespace = {"tf": tf}
    module = ast.Module(body=[original], type_ignores=[])
    exec(compile(module, str(upstream_path), "exec"), namespace)
    preprocess = namespace["get_preprocess_audio_func"](
        {
            "desired_samples": CLIP_SAMPLES,
            "background_frequency": 0.0,
            "background_volume_range_": 0.0,
            "feature_type": "mfcc",
            "sample_rate": SAMPLE_RATE,
            "window_size_samples": FRAME_LENGTH,
            "window_stride_samples": FRAME_STEP,
            "dct_coefficient_count": 10,
            "spectrogram_length": FEATURE_SHAPE[0],
        }
    )
    paths = list(paths)
    if not paths:
        raise ValueError("Select at least one WAV for the parity check")
    expected = []
    for path in paths:
        original_audio = _read_pcm16(path)
        if len(original_audio) > CLIP_SAMPLES or original_audio.max() <= 0:
            raise ValueError(f"Upstream parity requires an ordinary one-second-or-shorter WAV: {path}")
        expected.append(preprocess({"audio": original_audio})["audio"].numpy())
    actual = extract_mfcc(np.stack([read_clip(path) for path in paths]))
    np.testing.assert_allclose(actual, np.stack(expected), atol=3e-5, rtol=0)
    return {
        "clips": len(paths),
        "max_abs_mfcc_difference": float(np.max(np.abs(actual - expected))),
        "absolute_tolerance": 3e-5,
    }
