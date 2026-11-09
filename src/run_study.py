"""Run the locked paired evaluation of one frozen keyword spotter."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from audio import CLIP_SAMPLES, extract_mfcc, mix_at_snr, read_clip, read_noise
from models import check_float_parity, convert_float, convert_int8, predict


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIGEST = "24e4e02f6190a1bb4a00f7e84ee27f3b0620ac6f45fcf6c5dafeb27e78b2a820"
UPSTREAM_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MODEL_DIR = "benchmark/training/keyword_spotting/trained_models/kws_ref_model"
LABELS = ("down", "go", "left", "no", "off", "on", "right", "stop", "up",
          "yes", "silence", "unknown")
SPEECH_LABELS = tuple(label for label in LABELS if label != "silence")
CAL_NOISE = ("DKITCHEN", "OOFFICE", "TCAR")
TEST_NOISE = ("PCAFETER", "STRAFFIC", "TMETRO", "NRIVER")
DRAWS = range(20)
MIXTURES = (0, 50, 100)
SNRS = (20, 10, 0)
PREDICTION_FIELDS = (
    "clip_id", "speaker", "true_label", "condition", "noise_recording",
    "noise_offset", "realized_snr_db", "common_gain", "model", "draw",
    "mixture_percent", "predicted_label", "input_clamp_count",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(data):
    with (data / "speech_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cal = {draw: [] for draw in DRAWS}
    test = []
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest paths must be relative to the data directory.")
        if "_nohash_" not in path.stem or row["speaker"] != path.stem.split("_nohash_", 1)[0]:
            raise ValueError("Invalid speaker identifier in manifest.")
        if row["partition"] == "calibration":
            cal[int(row["draw"])].append(row)
        elif row["partition"] == "test" and not row["draw"]:
            test.append(row)
        else:
            raise ValueError("Unexpected split or draw in manifest.")
    for group, count in [(group, 10) for group in cal.values()] + [(test, 200)]:
        if Counter(row["label"] for row in group) != dict.fromkeys(SPEECH_LABELS, count):
            raise ValueError("Manifest violates the locked sample sizes.")
        if len({row["clip_id"] for row in group}) != len(group):
            raise ValueError("Manifest repeats an utterance within a draw or test.")
    unknown_counts = Counter(row["word"] for row in test if row["label"] == "unknown")
    if len(unknown_counts) != 25 or set(unknown_counts.values()) != {8}:
        raise ValueError("Unexpected unknown-word stratum counts.")
    cal_speakers = {row["speaker"] for group in cal.values() for row in group}
    test_speakers = {row["speaker"] for row in test}
    if cal_speakers & test_speakers:
        raise ValueError("Calibration and test speakers overlap.")
    return cal, test, len(cal_speakers), len(test_speakers)


def mix(speech, noise, offset, snr):
    waveform, actual, gain = mix_at_snr(speech, noise[offset:offset + CLIP_SAMPLES], snr)
    if abs(actual - snr) > 1e-8 or np.max(np.abs(waveform)) > 1.0:
        raise RuntimeError("Mixed waveform violated SNR or headroom requirement.")
    return waveform, actual, gain


def speech_item(row, condition, recording="", offset="", snr="", gain=1.0):
    return {
        "clip_id": row["clip_id"], "speaker": row["speaker"],
        "true_label": row["label"], "condition": condition, "noise_recording": recording,
        "noise_offset": offset, "realized_snr_db": snr, "common_gain": gain,
    }


def background_item(recording, offset):
    row = {"clip_id": f"{recording}:{offset}", "speaker": "", "label": "silence"}
    return speech_item(row, "background only", recording, offset)


def test_inputs(data, rows, noises):
    """Generate and cache features in batches, with one window per utterance/SNR family."""
    rng = np.random.default_rng(731901)
    speech = np.stack([read_clip(data / row["path"]) for row in rows])
    batches = [extract_mfcc(speech[i:i + 128]) for i in range(0, len(rows), 128)]
    items = [speech_item(row, "clean") for row in rows]
    for recording in TEST_NOISE:
        noise = noises[recording]
        shuffled = rng.permutation(len(noise) // CLIP_SAMPLES)
        negative = np.sort(shuffled[:100]) * CLIP_SAMPLES
        available = shuffled[100:]
        if not len(available):
            raise ValueError(f"Not enough full one-second windows in {recording}.")
        cycles = (len(rows) + len(available) - 1) // len(available)
        offsets = np.concatenate([rng.permutation(available) for _ in range(cycles)])
        offsets = offsets[:len(rows)] * CLIP_SAMPLES
        for snr in SNRS:
            for start in range(0, len(rows), 128):
                waves = []
                for i in range(start, min(start + 128, len(rows))):
                    offset = int(offsets[i])
                    wave, actual, gain = mix(speech[i], noise, offset, snr)
                    waves.append(wave)
                    items.append(speech_item(rows[i], f"{snr} dB", recording, offset, actual, gain))
                batches.append(extract_mfcc(np.stack(waves)))
        batches.append(extract_mfcc(np.stack([
            noise[offset:offset + CLIP_SAMPLES] for offset in negative
        ])))
        items.extend(background_item(recording, int(offset)) for offset in negative)
    return np.concatenate(batches), items


def balanced_recordings(rng, count):
    assignments = np.resize(rng.permutation(CAL_NOISE), count)
    return rng.permutation(assignments)


def calibration_inputs(data, rows, noises, draw):
    rng = np.random.default_rng(731902 + draw)
    clean = np.stack([read_clip(data / row["path"]) for row in rows])
    half = np.zeros(len(rows), dtype=bool)
    for label in SPEECH_LABELS:
        candidates = [i for i, row in enumerate(rows) if row["label"] == label]
        half[rng.choice(candidates, size=5, replace=False)] = True
    assignments = np.empty(len(rows), dtype="U16")
    for mask in (half, ~half):
        assignments[mask] = balanced_recordings(rng, int(np.sum(mask)))
    background = balanced_recordings(rng, 10)
    windows = {name: iter(rng.permutation(len(noise) // CLIP_SAMPLES) * CLIP_SAMPLES)
               for name, noise in noises.items()}
    noisy = []
    counterparts = []
    for row, waveform, recording in zip(rows, clean, assignments):
        offset = int(next(windows[recording]))
        mixed, actual, gain = mix(waveform, noises[recording], offset, 10)
        noisy.append(mixed)
        counterparts.append(speech_item(row, "10 dB", str(recording), offset, actual, gain))
    negatives = []
    negative_waves = []
    for recording in background:
        offset = int(next(windows[recording]))
        negatives.append(background_item(str(recording), offset))
        negative_waves.append(noises[recording][offset:offset + CLIP_SAMPLES])
    clean_features = extract_mfcc(clean)
    noisy_features = extract_mfcc(np.stack(noisy))
    negative_features = extract_mfcc(np.stack(negative_waves))
    for mixture, mask in zip(MIXTURES, (np.zeros(len(rows), dtype=bool), half,
                                        np.ones(len(rows), dtype=bool))):
        features = np.concatenate([
            np.where(mask[:, None, None, None], noisy_features, clean_features),
            negative_features,
        ])
        items = [counterparts[i] if mask[i] else speech_item(row, "clean")
                 for i, row in enumerate(rows)] + negatives
        yield mixture, features, items


def write_predictions(writer, items, model_name, model, features, draw="", mixture=""):
    labels, clamps, metadata = predict(model, features)
    writer.writerows({
        **item, "model": model_name, "draw": draw, "mixture_percent": mixture,
        "predicted_label": LABELS[int(label)], "input_clamp_count": int(clamp),
    } for item, label, clamp in zip(items, labels, clamps))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "final")
    parser.add_argument("--model", type=Path, default=ROOT / "upstream" / "mlperf-tiny" / MODEL_DIR)
    args = parser.parse_args()
    if sha256(ROOT / "PROTOCOL.md") != PROTOCOL_DIGEST:
        raise ValueError("The held-out protocol does not match the locked copy.")
    if (args.output / "run.json").exists():
        raise FileExistsError("Completed run exists. Choose a separate output directory.")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    data_audit = json.loads((args.data / "data_audit.json").read_text())
    manifest_hash = sha256(args.data / "speech_manifest.csv")
    if (data_audit["protocol_lock"]["sha256"] != PROTOCOL_DIGEST
            or data_audit["speech_manifest_sha256"] != manifest_hash
            or any(data_audit["official_split_speaker_overlaps"].values())):
        raise ValueError("The audited data does not match the protocol and manifest.")
    cal, test, cal_speakers, test_speakers = load_manifest(args.data)
    paths = {name: args.data / "noise" / f"{name}.wav" for name in CAL_NOISE + TEST_NOISE}
    noise_hashes = {name: sha256(path) for name, path in paths.items()}
    if len(set(noise_hashes.values())) != len(paths):
        raise ValueError("Two recording IDs have the same waveform.")
    noises = {name: read_noise(path) for name, path in paths.items()}
    source_hashes = {str(path.relative_to(args.model)): sha256(path)
                     for path in sorted(args.model.rglob("*")) if path.is_file()}
    features, items = test_inputs(args.data, test, noises)
    if len(items) != 29000 or features.shape != (29000, 49, 10, 1):
        raise ValueError("Final input count does not match the protocol.")
    print(f"Cached {len(items):,} inputs from {test_speakers} held-out speakers.", flush=True)
    base = convert_float(args.model)
    parity = check_float_parity(base, features[::1000], args.model)
    if not parity["allclose"] or parity["argmax_disagreements"]:
        raise RuntimeError("The float export failed the SavedModel parity check.")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "predictions.csv.gz"
    temp = path.with_suffix(".gz.part")
    calibration_manifest = []
    model_metadata = {}
    cal_noises = {name: noises[name] for name in CAL_NOISE}
    with gzip.open(temp, "wt", newline="", compresslevel=3) as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        model_metadata["float"] = write_predictions(writer, items, "float", base, features)
        for draw in DRAWS:
            for mixture, representatives, inputs in calibration_inputs(
                args.data, cal[draw], cal_noises, draw
            ):
                model_name = f"draw_{draw}_mixture_{mixture}"
                int8 = convert_int8(representatives, args.model)
                metadata = write_predictions(writer, items, model_name, int8, features, draw, mixture)
                if metadata["input_dtype"] != "int8" or "float32" in metadata["tensor_dtype_counts"]:
                    raise RuntimeError("Converted model was not fully integer.")
                model_metadata[model_name] = metadata
                calibration_manifest.extend({**item, "draw": draw, "mixture_percent": mixture}
                                            for item in inputs)
            print(f"Completed draw {draw + 1}/{len(DRAWS)}.", flush=True)
    if any(sha256(args.model / file) != digest for file, digest in source_hashes.items()):
        raise RuntimeError("SavedModel files changed during evaluation.")
    temp.replace(path)
    with (args.output / "calibration_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_manifest[0]))
        writer.writeheader()
        writer.writerows(calibration_manifest)
    with (args.output / "test_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    meta = {
        "protocol_sha256": PROTOCOL_DIGEST, "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "tf_keras": version("tf-keras"), "numpy": np.__version__,
        "architecture": platform.machine(), "source_sha256": source_hashes,
        "noise_sha256": noise_hashes,
        "speech_manifest_sha256": manifest_hash, "started_at_utc": started_at,
        "test_speech_clips": len(test), "test_feature_inputs": len(items),
        "calibration_speakers": cal_speakers, "test_speakers": test_speakers,
        "overlapping_speakers": 0, "float_parity": parity, "models": model_metadata,
        "predictions_sha256": sha256(path), "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Completed locked evaluation in {meta['elapsed_seconds']:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
