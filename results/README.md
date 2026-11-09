# Held-out results

This run evaluated 2,200 speech utterances from 241 held-out speakers and 400
background-only excerpts. The four test recordings and three SNR levels yield
29,000 matched inputs per model, including clean speech. The sample count for
the primary interval is based on the paired calibration draws and speaker
clusters. Multiple conditions share each utterance.

| Calibration speech augmented | Mean added int8 error on noisy speech |
| --- | ---: |
| 0% | −0.155 pp |
| 50% | −0.164 pp |
| 100% | −0.493 pp |

Added error is float accuracy minus int8 accuracy. Negative values indicate
slightly higher observed int8 accuracy on these paired inputs. The primary
50% minus 0% contrast was **−0.009 pp** (95% CI **−0.099 to +0.085 pp**).
The interval fits within the prespecified ±0.50 pp practical margin. The
100% arm had the lowest observed mean in a secondary descriptive comparison.

Float accuracy was 91.32% on clean speech, 78.31% over all noisy conditions,
and 60.99% at 0 dB averaged across the four recordings. NRIVER at 0 dB was
the lowest absolute accuracy cell for the float model, at 47.18%. The
condition tables distinguish low absolute accuracy from quantization damage.
Background-only excerpts have no injected speech, and incidental speech is
unannotated. The measured rates are nominal triggers on the selected clips.
The interval applies to this checkpoint, converter, and fixed noise protocol.

- `summary.json` contains the estimates, interval, and diagnostics.
- `conditions.csv`, `commands.csv`, and `calibration_draws.csv` give the condition, recall, and paired draw tables.
- `heldout_results.png` visualizes the speech accuracy, gaps, and primary comparison.
- `data_audit.json`, `run.json`, and the manifests document how to reproduce the run.
