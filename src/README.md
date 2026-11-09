# Study scripts

Run these scripts from the repository root. The [reproduction instructions](../README.md)
list the required environment and commands.

- `prepare_data.py` checks the protocol lock, verifies the source archives, and builds the speaker-disjoint manifests.
- `run_study.py` mixes the audio, exports the float and int8 models, and writes matched predictions.
- `analyze.py` checks the evaluation grid and produces the prespecified metrics, interval, and figure.
- `audio.py` and `models.py` provide the waveform, feature, conversion, and inference routines.
