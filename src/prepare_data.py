"""Build the locked Speech Commands and DEMAND manifests.

Fetch public source archives only when --download is given. Reuse a verified
archive cache when supplied. The outputs contain locally generated WAVs and
should not be committed to a public repository.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import tarfile
import wave
import zipfile


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent if HERE.name == "src" else HERE
DEFAULT_DATA = PROJECT / "data"
SPEECH_ARCHIVE = "speech_commands_v0.02.tar.gz"
CALIBRATION_NOISES = ("DKITCHEN", "OOFFICE", "TCAR")
TEST_NOISES = ("PCAFETER", "STRAFFIC", "TMETRO", "NRIVER")
VALIDATION_NOISES = ("DLIVING", "NPARK")
NOISE_ARCHIVES = {
    "DKITCHEN": (110501049, "7ffbf52d7f4699f96927846103dc8788"),
    "OOFFICE": (88995191, "7b61cc2d182d5a654cb9c3101ddd4041"),
    "TCAR": (130045975, "4d930012796bd298932245a26189f973"),
    "PCAFETER": (107431494, "99927d148128254141a9417d051510bb"),
    "STRAFFIC": (118572691, "2efa87262f272bbf9ba578088e81939c"),
    "TMETRO": (126589090, "95daf4df678e13b120e14211e6d89571"),
    "NRIVER": (98689565, "54264db61d3fe073fb81f2e40e0d19b5"),
}
ARCHIVES = {
    SPEECH_ARCHIVE: (
        "https://storage.googleapis.com/download.tensorflow.org/data/" + SPEECH_ARCHIVE,
        2428923189,
        "sha256",
        "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    ),
}
for recording, (size, checksum) in NOISE_ARCHIVES.items():
    filename = f"{recording}_16k.zip"
    ARCHIVES[filename] = (
        f"https://zenodo.org/records/1227121/files/{filename}?download=1",
        size,
        "md5",
        checksum,
    )
TARGETS = ("down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes")
CLIP_NAME = re.compile(r"[0-9a-f]+_nohash_[0-9]+\.wav")
FIELDS = ("partition", "draw", "label", "word", "speaker", "clip_id", "path")
CALIBRATION_DRAWS = 20
CALIBRATION_SEED = 726100
TEST_SEED = 726200
TEST_COUNT = 200
UNKNOWN_PER_WORD = 8


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_protocol_lock():
    expected, filename = (PROJECT / "PROTOCOL.sha256").read_text().split()
    locked_at = (PROJECT / "LOCKED_AT_UTC.txt").read_text().strip()
    if filename != "PROTOCOL.md" or not locked_at or digest(PROJECT / filename) != expected:
        raise ValueError("The held-out protocol does not match its local lock.")
    return {"sha256": expected, "locked_at_utc": locked_at}


def checked_archive(name, data, cache, download):
    url, size, algorithm, checksum = ARCHIVES[name]
    local = data / ".downloads" / name
    path = cache / name if cache and (cache / name).is_file() else local
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {name}. Supply a cache or rerun with --download.")
        local.parent.mkdir(parents=True, exist_ok=True)
        path = local.with_suffix(local.suffix + ".part")
        subprocess.run(
            ["curl", "-q", "--fail", "--location", "--silent", "--show-error",
             "--proto", "=https", "--retry", "2", "--connect-timeout", "15",
             "--output", str(path), url],
            check=True,
        )
    if path.stat().st_size != size or digest(path, algorithm) != checksum:
        raise ValueError(f"Archive failed published size or checksum verification: {path}")
    if path.suffix == ".part":
        path.replace(local)
        path = local
    print(f"Verified {name}", flush=True)
    return path


def speech_catalog(path):
    clips = set()
    lists = {}
    with tarfile.open(path, "r|gz") as archive:
        for entry in archive:
            if not entry.isfile():
                continue
            parts = PurePosixPath(entry.name).parts
            if parts in (("validation_list.txt",), ("testing_list.txt",)):
                lines = archive.extractfile(entry).read().decode("utf-8").splitlines()
                lists[parts[0]] = {line.strip() for line in lines if line.strip()}
            elif len(parts) == 2 and parts[0].isalpha() and CLIP_NAME.fullmatch(parts[1]):
                clips.add("/".join(parts))
    validation = lists["validation_list.txt"]
    test = lists["testing_list.txt"]
    if not (validation | test) <= clips or validation & test:
        raise ValueError("Official splits do not match the speech archive.")
    calibration = clips - validation - test
    return {"calibration": calibration, "validation": validation, "test": test}


def speaker(clip):
    return PurePosixPath(clip).name.split("_nohash_", 1)[0]


def speech_by_word(clips):
    by_word = {}
    for clip in sorted(clips):
        by_word.setdefault(clip.split("/")[0], []).append(clip)
    if len(set(by_word) - set(TARGETS)) != 25 or not set(TARGETS) <= set(by_word):
        raise ValueError("Expected the ten commands and 25 other Speech Commands words.")
    return by_word


def prefer_distinct_speakers(clips, count, rng):
    candidates = list(clips)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible clips for a {count}-clip sample.")
    selected, repeated = [], []
    seen = set()
    for clip in candidates:
        speaker_id = speaker(clip)
        if speaker_id in seen:
            repeated.append(clip)
        else:
            seen.add(speaker_id)
            selected.append(clip)
            if len(selected) == count:
                return selected
    return selected + repeated[:count - len(selected)]


def stratified_test_unknown(by_word, rng):
    """Sample eight clips per word, preferring speakers not yet used in unknown."""
    words = sorted(set(by_word) - set(TARGETS))
    candidates = {word: list(by_word[word]) for word in words}
    for word in words:
        rng.shuffle(candidates[word])
        if len(candidates[word]) < UNKNOWN_PER_WORD:
            raise ValueError(f"Not enough test examples for unknown word {word}.")
    selected = []
    seen = set()
    for _ in range(UNKNOWN_PER_WORD):
        round_words = list(words)
        rng.shuffle(round_words)
        for word in round_words:
            pool = candidates[word]
            index = next((i for i, clip in enumerate(pool) if speaker(clip) not in seen), 0)
            clip = pool.pop(index)
            selected.append(clip)
            seen.add(speaker(clip))
    return selected


def manifest_rows(selections, partition, draw=""):
    return [
        {"partition": partition, "draw": draw, "label": label,
         "word": clip.split("/")[0], "speaker": speaker(clip), "clip_id": clip,
         "path": "speech/" + clip}
        for label, clips in selections for clip in clips
    ]


def select_clips(catalog):
    calibration = speech_by_word(catalog["calibration"])
    test = speech_by_word(catalog["test"])
    other_words = sorted(set(calibration) - set(TARGETS))
    rows = []
    for draw in range(CALIBRATION_DRAWS):
        rng = random.Random(CALIBRATION_SEED + draw)
        selections = [(word, prefer_distinct_speakers(calibration[word], 10, rng))
                      for word in TARGETS]
        unknown = [rng.choice(calibration[word]) for word in rng.sample(other_words, 10)]
        rows.extend(manifest_rows(selections + [("unknown", unknown)], "calibration", draw))
    rng = random.Random(TEST_SEED)
    selections = [(word, prefer_distinct_speakers(test[word], TEST_COUNT, rng))
                  for word in TARGETS]
    selections.append(("unknown", stratified_test_unknown(test, rng)))
    rows.extend(manifest_rows(selections, "test"))
    return rows


def inspect_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2):
            raise ValueError(f"Expected mono 16-bit 16 kHz PCM: {path}")
        if wav.getnframes() == 0:
            raise ValueError(f"Empty WAV: {path}")
        return wav.getnframes()


def extract_speech(path, rows, data):
    pending = {row["clip_id"] for row in rows}
    with tarfile.open(path, "r|gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name).as_posix()
            if name not in pending or not entry.isfile():
                continue
            output = data / "speech" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(entry) as source, output.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            inspect_wav(output)
            pending.remove(name)
            if not pending:
                break
    if pending:
        raise ValueError(f"{len(pending)} selected speech clips were not extracted.")


def extract_noise(path, recording, data):
    output = data / "noise" / f"{recording}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if PurePosixPath(name).name == "ch01.wav"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one ch01.wav in {path}.")
        with archive.open(candidates[0]) as source, output.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    frames = inspect_wav(output)
    if frames // 16000 < 300:
        raise ValueError(f"{recording} has fewer than 300 whole seconds.")
    return {"source_channel": "ch01.wav", "path": output.relative_to(data).as_posix(),
            "samples": frames, "sha256": digest(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Local data output directory")
    parser.add_argument("--cache", type=Path, help="Read-only directory of existing source archives")
    parser.add_argument("--download", action="store_true", help="Download missing public archives")
    args = parser.parse_args()
    protocol_lock = check_protocol_lock()
    data = args.data
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {name: pool.submit(checked_archive, name, data, args.cache, args.download)
                for name in ARCHIVES}
        archives = {name: job.result() for name, job in jobs.items()}
    catalog = speech_catalog(archives[SPEECH_ARCHIVE])
    split_speakers = {split: {speaker(clip) for clip in clips} for split, clips in catalog.items()}
    overlaps = {
        f"{a}_vs_{b}": len(a_speakers & split_speakers[b])
        for a, a_speakers in split_speakers.items()
        for b in split_speakers
        if a < b
    }
    if any(overlaps.values()):
        raise ValueError(f"Speaker overlap across official speech splits: {overlaps}")
    rows = select_clips(catalog)
    test_rows = [row for row in rows if row["partition"] == "test"]
    selected = {split: {row["clip_id"] for row in rows if row["partition"] == split}
                for split in ("calibration", "test")}
    if any(not selected[split] <= catalog[split] for split in selected):
        raise ValueError("A selected clip is outside its official split.")
    if len(test_rows) != 11 * TEST_COUNT or len(selected["test"]) != len(test_rows):
        raise ValueError("Test speech selection does not contain exactly 2,200 unique clips.")
    extract_speech(archives[SPEECH_ARCHIVE], rows, data)
    recordings = {
        split: {recording: extract_noise(archives[f"{recording}_16k.zip"], recording, data)
                for recording in split_recordings}
        for split, split_recordings in (("calibration", CALIBRATION_NOISES), ("test", TEST_NOISES))
    }
    noise_digests = {recordings[split][recording]["sha256"]
                     for split in recordings for recording in recordings[split]}
    if len(noise_digests) != len(CALIBRATION_NOISES) + len(TEST_NOISES):
        raise ValueError("Two DEMAND recordings have identical extracted WAVs.")
    manifest = data / "speech_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "protocol_lock": protocol_lock,
        "sources": {
            "Speech Commands v0.02": {
                "paper": "https://arxiv.org/abs/1804.03209",
                "license": "CC BY 4.0",
                "checksums": "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/speech_commands/checksums.tsv",
            },
            "DEMAND": {
                "record": "https://zenodo.org/records/1227121",
                "checksums": "https://zenodo.org/api/records/1227121",
                "license": "The creators’ description specifies CC BY-SA 3.0 Unported for audio. Zenodo’s structured license field says CC BY 4.0.",
            },
        },
        "source_archives": {
            name: {"url": url, "bytes": size, "checksum": f"{algorithm}:{checksum}"}
            for name, (url, size, algorithm, checksum) in ARCHIVES.items()
        },
        "official_split_clip_counts": {name: len(clips) for name, clips in catalog.items()},
        "official_split_speaker_counts": {name: len(ids) for name, ids in split_speakers.items()},
        "official_split_speaker_overlaps": overlaps,
        "speech_manifest_rows": len(rows),
        "speech_manifest_sha256": digest(manifest),
        "selected_unique_speech": {split: len(clips) for split, clips in selected.items()},
        "selected_speakers": {split: len({speaker(clip) for clip in clips})
                              for split, clips in selected.items()},
        "test_clip_counts_by_label": dict(Counter(row["label"] for row in test_rows)),
        "test_speaker_counts_by_label": {
            label: len({row["speaker"] for row in test_rows if row["label"] == label})
            for label in (*TARGETS, "unknown")
        },
        "test_unknown_word_counts": dict(sorted(
            Counter(row["word"] for row in test_rows if row["label"] == "unknown").items()
        )),
        "reserved_validation_noise_recordings": VALIDATION_NOISES,
        "noise_recordings": recordings,
    }
    (data / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
