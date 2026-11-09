# Held-out evaluation protocol

Frozen locally on 2026-11-09, before any final WAV extraction, final DEMAND
download, or final inference. The SHA-256 in `PROTOCOL.sha256` identifies this
version. The earlier validation pilot used DKITCHEN for calibration and DLIVING
for validation. Its observed mixture differences were small. The final design
uses 20 calibration draws, up from 10 in the initial plan, to better characterize
draw variability. The pilot also established that CPU conversion is fast enough.

## Question and primary comparison

For the frozen MLPerf Tiny DS-CNN, how does adding recorded background noise to
0%, 50%, or 100% of the speech examples in a 120-clip calibration set change the
error added by int8 quantization on held-out speakers and background recordings?

The primary contrast is 50% versus 0%. Let D be float accuracy minus int8
accuracy in percentage points on paired inputs. Average D equally over the 11
speech-bearing labels, four recordings, three SNRs, and 20 calibration draws.
Subtract D for the 0% arm from D for the 50% arm. A negative result favors the
50% arm. The primary analysis excludes clean audio and background-only clips.
The 100% versus 0% and 50% versus 100% comparisons are secondary and descriptive.

## Data and fixed sample sizes

Use Google Speech Commands v0.02 and the official validation and testing lists.
Group by the speaker ID before `_nohash_`. Audit the three complete partitions
for overlap. Use the official training split as the calibration pool. Draw ten
clips per speech label for each of 20 draws, using seed 726100 + draw index,
prioritizing distinct speakers per target label. For unknown, choose ten distinct
source words and one clip per word in each draw. Draws are sampled separately
and may share clips. The 10 target labels follow the model's published label
order. All other spoken words map to unknown.

Sample 200 distinct clips for each of the 11 speech labels from the official
test split, using seed 726200, for 2,200 unique speech clips. For each target
label, prefer a different speaker per clip and fill from unique remaining clips
if needed. Select eight clips from each of the 25 unknown words, preferring
different speakers across the unknown label. Record the resulting manifest and
unique speaker counts. Each utterance appears unaugmented once and in all 12
noisy cells.

Use only channel 1 of the 16 kHz DEMAND archives. Assign each full recording to
one split. Calibration uses DKITCHEN, OOFFICE, TCAR. Validation is reserved for
DLIVING and NPARK. Final evaluation uses PCAFETER, STRAFFIC, TMETRO, NRIVER, in
that order. The final SNRs are 20, 10, and 0 dB. These four recordings are fixed
conditions and do not represent a random sample of environments.

## Pairing and preprocessing

Each calibration draw has 110 speech examples and ten background-only examples,
one version per clip and ten examples per label. Assign each speech utterance
one 10 dB noisy counterpart. Exactly five of ten per speech label use that
counterpart in the 50% arm. Use the same counterparts in the 100% arm. Assign
the three calibration backgrounds approximately equally in both halves of the
speech sample and among the ten background-only clips. Use nonoverlapping
one-second noise windows within a draw. Keep the background-only inputs and
speech ordering identical in the three arms. Use noise selection seed 731902
plus draw index.

For each test recording, use seed 731901 (one stream across recordings) to
reserve 100 distinct nonoverlapping one-second background-only windows. Draw
speech noise windows from the remaining nonoverlapping windows, using shuffled
cycles to balance reuse. Reuse each assigned window at all three SNRs for an
utterance. The total test manifest has 28,600 speech inputs and 400
background-only inputs, for 29,000 inputs. A background-only input contains no
injected foreground speech. Incidental words in the environment are unannotated,
so report its target trigger rate as nominal, rather than verified false alarms.
The unknown-word false positive rate uses labelled out-of-vocabulary speech.

Require 16 kHz mono PCM, crop or right-pad to one second, and add noise in the
waveform domain at the declared full-clip RMS ratio. Use one gain for both
components to retain headroom, then apply the upstream positive-peak MFCC
front end with its zero guard. Abort on invalid data rather than substituting
clips or environments after observing scores. Use the same cached features for
every model. Export the pure float baseline from the released SavedModel with
optimization disabled. Export 60 full-int8 models from the same checkpoint and
converter version. For each int8 input, round the affine code to nearest with
ties to even, saturate to the int8 range, and then cast. Decode all outputs
using argmax with ties assigned to the first label in model order. Record
model hashes, quantization parameters, and input saturation.

## Inference and reporting rules

The primary two-sided 95% confidence interval is the percentile interval from
10,000 crossed bootstrap replicates using seed 12020261109. Resample the 20
calibration draws and the unique test speakers independently with replacement.
Carry matched mixture arms with their draw, and carry all examples, noise
conditions, and precision predictions with their speaker. Recompute the
equal-label mean within each replicate. Keep test recordings and noise offsets
fixed. Show all 20 paired draw estimates. The interval is conditional on the
fixed test noise protocol and does not cover unseen populations of recordings.

Predeclare ±0.50 percentage points as the practical margin for the primary
contrast. An interval entirely within that range supports a small effect for
this setup. An interval spanning both zero and practically important changes
leaves those changes unresolved. Report the signed estimate and full interval
regardless of the result. Do not reselect a winning recipe on final data.

Report ordinary top-1 accuracy on the 29,000-input manifest, overall and by
speech condition, per-command recall, target false positives on unknown speech,
and nominal trigger rates on background-only clips. Report clean results
separately. Count paired flips in both directions. Define worst absolute int8
performance as the lowest balanced speech accuracy among the 12 noisy cells,
after averaging across draws. Define worst quantization damage as the largest
signed D on that same grid. For descriptive background-only uncertainty use
within-recording 10-second blocks and paired calibration draws. Include
per-recording results and note that speech noise windows are reused.

The evaluation runs on CPU with Python 3.11, TensorFlow 2.16.2, legacy Keras
2.16.0, and NumPy 1.26.4. The upstream MLPerf Tiny source is pinned at
`4addd0fa08d216e20637637874e084895f289da4`. This is an offline clip study.
Hardware timing, energy, and real microphone behavior are outside its scope.
