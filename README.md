# Tiny keyword spotting under noise

Does changing the noise content of a 120-clip calibration set affect the
error introduced by int8 quantization on unfamiliar background recordings?
This CPU-only study compares three calibration mixtures for a frozen MLPerf
Tiny keyword spotter.

Each calibration draw uses the same underlying speech across the 0%, 50%, and
100% augmentation arms. The arms share the same ten background-only clips.
Added calibration noise has a full-clip SNR of 10 dB. Evaluation compares each
int8 model with the true float export on exactly the same clips.

## Study and results

The [locked protocol](PROTOCOL.md) fixes the split, sample counts, and analysis
before test evaluation. The final study uses 20 calibration draws, 200 test
utterances per speech-bearing label, and four whole held-out DEMAND recordings.
The primary contrast is 50% versus 0% augmentation, averaged over the four
recordings and three test SNRs. See the [held-out summary](results/summary.json)
and [result figure](results/heldout_results.png).

**The primary contrast was −0.009 percentage points** (95% paired confidence
interval −0.099 to +0.085). The interval lies within the locked ±0.50 point
practical margin for the 50% versus 0% comparison. Float speech accuracy
fell from 91.3% on unaugmented clips to 61.0% at 0 dB, averaged across the
four recordings. The 100% arm had the lowest observed mean added error in
the secondary comparison. See the [results and scope](results/README.md).

The earlier validation pilot used three draws, 275 speech utterances, and one
validation background. Float accuracy was 93.5% on unmodified speech and 76.4%
at 0 dB. The largest difference between the mixtures' average int8 error gaps
on the pilot's noisy conditions was 0.16 percentage points. These were
validation observations used to plan the final study.

## Reproduce the held-out run

Use Python 3.11, Git, and curl. The Python dependencies below are pinned to the
versions exercised on Apple Silicon. Run these commands from the root of this
repository. The public source and data downloads require no sign-in.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

git init -q upstream/mlperf-tiny
git -C upstream/mlperf-tiny fetch --depth=1 https://github.com/mlcommons/tiny.git 4addd0fa08d216e20637637874e084895f289da4
git -C upstream/mlperf-tiny checkout --detach FETCH_HEAD

.venv/bin/python src/prepare_data.py --data data --download
.venv/bin/python src/run_study.py --data data --output runs/final \
  --model upstream/mlperf-tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model
.venv/bin/python src/analyze.py --predictions runs/final/predictions.csv.gz \
  --output runs/final/analysis
```

The model checkpoint comes from the exact upstream commit above. The data
preparation script verifies archive checksums and selects audio from the
official Speech Commands v0.02 speaker split and the designated DEMAND
recordings. Use `--cache /path/to/existing/archives` to reuse verified
downloads, or omit `--download` when all required archives are cached.
Allow several gigabytes for the downloads and intermediate outputs.

The runner writes per-clip predictions and provenance under `runs/final`.
The analysis writes the primary contrast, condition and command tables, and
the held-out figure under `runs/final/analysis`. The tracked `results/`
directory contains selected outputs from the original run. Confidence
intervals are conditional on the four test recordings and the sampled noise
windows, so the results support claims about this evaluation setup.

## Source and data credits

The study code is available under [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) and [third-party sources and terms](THIRD_PARTY.md) for
the MLPerf Tiny checkpoint, Speech Commands, and DEMAND. Downloads, audio,
model weights, and per-clip predictions are stored locally.
