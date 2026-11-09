"""Analyze paired held-out MLPerf Tiny predictions from the locked protocol."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LABELS = ("down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
          "silence", "unknown")
COMMANDS = LABELS[:10]
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}
SPEECH_LABELS = (*COMMANDS, "unknown")
RECORDINGS = ("PCAFETER", "STRAFFIC", "TMETRO", "NRIVER")
SNRS = (20, 10, 0)
MIXTURES = (0, 50, 100)
DRAWS = tuple(range(20))
PER_SPEECH_LABEL = 200
PER_BACKGROUND_RECORDING = 100
FEATURE_VALUES = 49 * 10
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 12020261109
PRACTICAL_MARGIN_PP = 0.5
CLEAN = ("", "clean")
NOISY = tuple((recording, f"{snr} dB") for recording in RECORDINGS for snr in SNRS)
BACKGROUNDS = tuple((recording, "background only") for recording in RECORDINGS)
ROWS_PER_MODEL = (1 + len(NOISY)) * len(SPEECH_LABELS) * PER_SPEECH_LABEL + (
    len(BACKGROUNDS) * PER_BACKGROUND_RECORDING
)


def model_name(draw, mixture):
    return f"draw_{draw}_mixture_{mixture}"


def read_predictions(path):
    """Read paired predictions into compact arrays and check matched inputs."""
    required = {"float": ("", "")}
    required.update({model_name(draw, mixture): (str(draw), str(mixture))
                     for draw in DRAWS for mixture in MIXTURES})
    allowed = {**required, "shipped_int8": ("", "")}
    identities, predictions, clamps = {}, {}, {}
    opened = gzip.open if path.suffix == ".gz" else open
    with opened(path, "rt", newline="") as file:
        for row in csv.DictReader(file):
            name = row["model"]
            if name not in allowed or (row["draw"], row["mixture_percent"]) != allowed[name]:
                raise ValueError(f"Unexpected model or calibration assignment: {name}")
            if name not in predictions:
                predictions[name] = np.full(ROWS_PER_MODEL, -1, dtype=np.int8)
                clamps[name] = np.zeros(ROWS_PER_MODEL, dtype=np.uint16)
            key = (row["noise_recording"], row["condition"], row["clip_id"])
            attrs = (row["true_label"], row["speaker"], row["noise_offset"],
                     row["realized_snr_db"], row["common_gain"])
            if name == "float":
                if key in identities or len(identities) == ROWS_PER_MODEL:
                    raise ValueError(f"Duplicate or unexpected float input: {key}")
                identities[key] = (len(identities), attrs)
            elif len(identities) != ROWS_PER_MODEL:
                raise ValueError("Write every float prediction before the int8 predictions.")
            i, expected_attrs = identities[key]
            if attrs != expected_attrs or predictions[name][i] != -1:
                raise ValueError(f"Changed input or duplicate prediction for {name}, {key}")
            predictions[name][i] = LABEL_INDEX[row["predicted_label"]]
            clamp_count = int(row["input_clamp_count"])
            if not 0 <= clamp_count <= FEATURE_VALUES:
                raise ValueError(f"Invalid input clamp count for {name}, {key}")
            clamps[name][i] = clamp_count
    if not required.keys() <= predictions.keys() or any(
        np.any(labels < 0) for labels in predictions.values()
    ):
        raise ValueError("Missing at least one required model or prediction.")
    return identities, predictions, clamps


def assemble_grid(identities):
    """Validate the fixed 12-cell grid, backgrounds and shared utterance order."""
    group = defaultdict(dict)
    for (recording, condition, clip_id), entry in identities.items():
        group[recording, condition][clip_id] = entry
    if group.keys() != {CLEAN, *NOISY, *BACKGROUNDS}:
        raise ValueError("Prediction conditions differ from the locked noise grid.")
    clean = group[CLEAN]
    speech_ids = sorted(clean)
    expected_counts = dict.fromkeys(SPEECH_LABELS, PER_SPEECH_LABEL)
    if Counter(attrs[0] for _, attrs in clean.values()) != expected_counts:
        raise ValueError("Unexpected clean sample size or class balance.")
    if any(not attrs[1] or attrs[2] or attrs[3] for _, attrs in clean.values()):
        raise ValueError("Clean inputs must carry speaker IDs and no noise offsets or SNR.")
    indices = {CLEAN: np.array([clean[clip_id][0] for clip_id in speech_ids])}
    offsets = {}
    for cell in NOISY:
        recording, condition = cell
        if group[cell].keys() != clean.keys():
            raise ValueError(f"Unmatched noisy speech IDs in {cell}.")
        cell_offsets = []
        for clip_id in speech_ids:
            _, attrs = group[cell][clip_id]
            if attrs[:2] != clean[clip_id][1][:2]:
                raise ValueError(f"Changed label or speaker for {clip_id} in {cell}.")
            offset = int(attrs[2])
            snr = float(attrs[3])
            if offset < 0 or abs(snr - int(condition.split()[0])) > 1e-5 or not np.isfinite(snr):
                raise ValueError(f"Invalid recorded noise offset or SNR for {clip_id} in {cell}.")
            cell_offsets.append(offset)
        if offsets.setdefault(recording, cell_offsets) != cell_offsets:
            raise ValueError(f"Noise excerpts changed across SNR conditions in {recording}.")
        indices[cell] = np.array([group[cell][clip_id][0] for clip_id in speech_ids])
    blocks = {}
    for cell in BACKGROUNDS:
        entries = sorted(group[cell].values())
        if len(entries) != PER_BACKGROUND_RECORDING:
            raise ValueError(f"Unexpected background sample size in {cell}.")
        bg_offsets = []
        for _, (label, speaker, offset, snr, _) in entries:
            if label != "silence" or speaker or snr or int(offset) < 0:
                raise ValueError(f"Invalid background entry in {cell}.")
            bg_offsets.append(int(offset))
        if any(b - a < 16_000 for a, b in zip(sorted(bg_offsets), sorted(bg_offsets)[1:])):
            raise ValueError(f"Background excerpts overlap in {cell}.")
        if set(bg_offsets) & set(offsets[cell[0]]):
            raise ValueError(f"Reserved background windows were reused for speech in {cell[0]}.")
        indices[cell] = np.array([i for i, _ in entries])
        # Resample nonempty, contiguous 10-second recording bins as clusters.
        _, blocks[cell] = np.unique(np.array(bg_offsets) // 160_000, return_inverse=True)
    target = np.empty(ROWS_PER_MODEL, dtype=np.int8)
    for i, attrs in identities.values():
        target[i] = LABEL_INDEX[attrs[0]]
    speaker_ids = np.array([clean[clip_id][1][1] for clip_id in speech_ids])
    _, speaker_index = np.unique(speaker_ids, return_inverse=True)
    return indices, target, speaker_index, blocks


def interval(samples):
    return np.quantile(samples, [0.025, 0.975]).tolist()


def primary_bootstrap(speaker_index, speech_labels, per_clip_contrast):
    """Cross speaker and draw resamples, retaining all paired noisy copies."""
    label_index = np.searchsorted(np.unique(speech_labels), speech_labels)
    counts = np.zeros((int(speaker_index.max()) + 1, len(SPEECH_LABELS)), dtype=np.int64)
    np.add.at(counts, (speaker_index, label_index), 1)
    totals = np.zeros((len(DRAWS), *counts.shape), dtype=np.float64)
    for draw, contrast in enumerate(per_clip_contrast):
        np.add.at(totals[draw], (speaker_index, label_index), contrast)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rng_draw = np.full(len(DRAWS), 1 / len(DRAWS))
    rng_speaker = np.full(len(counts), 1 / len(counts))
    estimates = []
    for start in range(0, BOOTSTRAP_ITERATIONS, 500):
        size = min(500, BOOTSTRAP_ITERATIONS - start)
        draw_weights = rng.multinomial(len(DRAWS), rng_draw, size=size)
        speaker_weights = rng.multinomial(len(counts), rng_speaker, size=size)
        denominators = speaker_weights @ counts
        if np.any(denominators == 0):
            raise ValueError("Bootstrap omitted a speech label. Review speaker coverage.")
        weights = speaker_weights[:, :, None] / denominators[:, None, :]
        by_draw = weights.reshape(size, -1) @ totals.reshape(len(DRAWS), -1).T
        estimates.extend(
            ((by_draw * draw_weights).sum(axis=1) * 100 / (len(DRAWS) * len(SPEECH_LABELS))).tolist()
        )
    point = 100 * (totals.sum(axis=1) / counts.sum(axis=0)).mean()
    return float(point), interval(estimates)


def background_bootstrap(prediction_series, indices, blocks):
    """Use paired time-block resampling for nominal background trigger rates."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    draw_weights = rng.multinomial(
        len(DRAWS), np.full(len(DRAWS), 1 / len(DRAWS)), size=BOOTSTRAP_ITERATIONS
    )
    by_environment = {name: {} for name in prediction_series}
    pooled_samples = {name: [] for name in prediction_series}
    pooled_points = {name: [] for name in prediction_series}
    for cell in BACKGROUNDS:
        recording = cell[0]
        clip_index = indices[cell]
        block_index = blocks[cell]
        block_sizes = np.bincount(block_index)
        n_blocks = len(block_sizes)
        weights = rng.multinomial(
            n_blocks, np.full(n_blocks, 1 / n_blocks), size=BOOTSTRAP_ITERATIONS
        )
        denominators = weights @ block_sizes
        for name, predictions in prediction_series.items():
            triggers = predictions[:, clip_index] < len(COMMANDS)
            hits = np.array([np.bincount(block_index, weights=row, minlength=n_blocks)
                             for row in triggers])
            by_draw = (weights @ hits.T) * 100 / denominators[:, None]
            samples = (by_draw * draw_weights).sum(axis=1) / len(DRAWS) if len(hits) > 1 else by_draw[:, 0]
            point = float(100 * triggers.mean())
            by_environment[name][recording] = {
                "rate_percent": point,
                "ci95_percent": interval(samples),
                "clips": len(clip_index),
                "nonempty_10_second_blocks": n_blocks,
            }
            pooled_samples[name].append(samples)
            pooled_points[name].append(point)
    return {
        name: {
            "pooled_equal_recording_rate_percent": float(np.mean(pooled_points[name])),
            "pooled_ci95_percent": interval(np.mean(pooled_samples[name], axis=0)),
            "by_recording": by_environment[name],
        }
        for name in prediction_series
    }


def practical_readout(ci):
    lower, upper = ci
    if -PRACTICAL_MARGIN_PP < lower and upper < PRACTICAL_MARGIN_PP:
        return "The interval is entirely inside the prespecified practical margin."
    if upper < -PRACTICAL_MARGIN_PP:
        return "The interval supports a practically meaningful reduction from 50% augmentation."
    if lower > PRACTICAL_MARGIN_PP:
        return "The interval supports a practically meaningful increase from 50% augmentation."
    return "The interval does not resolve the contrast relative to the practical margin."


def summarize(identities, predictions, clamps):
    indices, target, speaker_index, blocks = assemble_grid(identities)
    model_series = {"float": ["float"]}
    model_series.update({
        f"int8_{mixture}pct": [model_name(draw, mixture) for draw in DRAWS]
        for mixture in MIXTURES
    })
    if "shipped_int8" in predictions:
        model_series["shipped_int8"] = ["shipped_int8"]
    pred_series = {name: np.stack([predictions[model] for model in models])
                   for name, models in model_series.items()}
    clamp_series = {name: np.stack([clamps[model] for model in models])
                    for name, models in model_series.items()}
    correct = {name: values == target for name, values in pred_series.items()}
    noisy_grid = np.stack([indices[cell] for cell in NOISY])
    noisy_all = noisy_grid.ravel()
    all_idx = np.arange(len(target))
    fp_correct = correct["float"][0]
    gaps = {
        mixture: 100 * (fp_correct[noisy_all].mean() -
                        correct[f"int8_{mixture}pct"][:, noisy_all].mean(axis=1))
        for mixture in MIXTURES
    }
    paired = gaps[50] - gaps[0]
    per_clip = (
        correct["int8_0pct"][:, noisy_grid].mean(axis=1)
        - correct["int8_50pct"][:, noisy_grid].mean(axis=1)
    )
    point, ci = primary_bootstrap(speaker_index, target[indices[CLEAN]], per_clip)
    if not np.isclose(point, paired.mean(), atol=1e-10, rtol=0):
        raise RuntimeError("Paired bootstrap estimand does not match the observed grid contrast.")
    draws = [{
        "draw": draw, "int8_0pct_gap_pp": gaps[0][draw],
        "int8_50pct_gap_pp": gaps[50][draw], "int8_100pct_gap_pp": gaps[100][draw],
        "50_minus_0_pp": paired[draw],
        "100_minus_0_pp": gaps[100][draw] - gaps[0][draw],
        "50_minus_100_pp": gaps[50][draw] - gaps[100][draw],
    } for draw in DRAWS]
    cell_rows, command_rows = [], []
    worst, overall = {}, {}
    for name, values in pred_series.items():
        def metrics(idx):
            is_correct = correct[name][:, idx]
            reference = fp_correct[idx]
            is_speech = target[idx] != LABEL_INDEX["silence"]
            unknown = idx[target[idx] == LABEL_INDEX["unknown"]]
            background = idx[~is_speech]
            return {
                "unique_input_conditions": len(idx),
                "top1_accuracy_percent": float(100 * is_correct.mean()),
                "added_error_pp": float(100 * (reference.mean() - is_correct.mean())),
                "broken_float_decisions_mean": float(np.sum(reference & ~is_correct, axis=1).mean()),
                "corrected_float_decisions_mean": float(np.sum(~reference & is_correct, axis=1).mean()),
                "unknown_clips": len(unknown),
                "unknown_target_false_positives_mean": (
                    float((values[:, unknown] < len(COMMANDS)).sum(axis=1).mean()) if len(unknown) else None
                ),
                "unknown_target_false_positive_percent": (
                    float(100 * (values[:, unknown] < len(COMMANDS)).mean()) if len(unknown) else None
                ),
                "background_clips": len(background),
                "background_nominal_triggers_mean": (
                    float((values[:, background] < len(COMMANDS)).sum(axis=1).mean()) if len(background) else None
                ),
                "background_nominal_trigger_percent": (
                    float(100 * (values[:, background] < len(COMMANDS)).mean()) if len(background) else None
                ),
                "input_clamped_values_percent": float(100 * clamp_series[name][:, idx].mean() / FEATURE_VALUES),
            }

        overall[name] = metrics(all_idx)
        rows = {}
        for cell, idx in [*indices.items(), (("ALL", "noisy pooled"), noisy_all)]:
            recording, condition = cell
            row = {"series": name, "recording": recording, "condition": condition, **metrics(idx)}
            rows[cell] = row
            cell_rows.append(row)
            if cell in BACKGROUNDS:
                continue
            for label in COMMANDS:
                label_idx = idx[target[idx] == LABEL_INDEX[label]]
                recall = correct[name][:, label_idx].mean()
                command_rows.append({
                    "series": name, "recording": recording, "condition": condition, "command": label,
                    "clips": len(label_idx), "recall_percent": float(100 * recall),
                    "recall_change_vs_float_pp": float(100 * (recall - fp_correct[label_idx].mean())),
                })
        lowest_cell = min(NOISY, key=lambda cell: rows[cell]["top1_accuracy_percent"])
        highest_gap = max(NOISY, key=lambda cell: rows[cell]["added_error_pp"])
        worst[name] = {
            "lowest_absolute_accuracy": rows[lowest_cell],
            "largest_float_to_int8_gap": rows[highest_gap],
        }
    summary = {
        "scope": "Held-out evaluation on four fixed DEMAND recordings and official Speech Commands test speakers.",
        "design": {
            "calibration_draws": len(DRAWS),
            "mixtures_percent_speech_augmented": list(MIXTURES),
            "held_out_recordings": list(RECORDINGS),
            "snr_db": list(SNRS),
            "speech_clips_per_label": PER_SPEECH_LABEL,
            "background_clips_per_recording": PER_BACKGROUND_RECORDING,
            "unique_test_utterances": len(speaker_index),
            "unique_test_speakers": int(speaker_index.max()) + 1,
            "input_conditions_per_model": len(target),
        },
        "primary": {
            "estimand": "50% minus 0% in float-minus-int8 accuracy gap, in percentage points. "
                        "Average equally over calibration draws, 11 speech labels, four recordings, and three SNRs.",
            "estimate_pp": point,
            "ci95_pp": ci,
            "practical_margin_pp": PRACTICAL_MARGIN_PP,
            "interpretation": practical_readout(ci),
            "bootstrap": {
                "method": "Crossed paired percentile bootstrap over calibration draws and test speakers.",
                "iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED,
                "scope": "Conditional on fixed recordings and assigned noise excerpts.",
            },
        },
        "secondary_descriptive": {
            "noisy_float_accuracy_percent": float(100 * fp_correct[noisy_all].mean()),
            "noisy_mean_added_error_pp": {str(m): float(gaps[m].mean()) for m in MIXTURES},
            "100_minus_0_pp": float((gaps[100] - gaps[0]).mean()),
            "50_minus_100_pp": float((gaps[50] - gaps[100]).mean()),
            "calibration_draw_points": draws,
            "overall_12_label_top1": overall,
            "worst_cases_on_fixed_noisy_grid": worst,
            "background_triggers": {
                "scope": "Target predictions on excerpts with no injected speech. Incidental speech is unannotated. "
                         "Intervals resample 10-second time bins within each fixed recording and calibration draws. "
                         "Degenerate zero intervals do not establish a zero deployment false-positive rate.",
                "by_model_series": background_bootstrap(pred_series, indices, blocks),
            },
        },
    }
    return summary, cell_rows, command_rows, draws


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import PercentFormatter

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.085, right=0.95, top=0.83, bottom=0.12, hspace=0.6, wspace=0.31)
    fig.suptitle("Int8 calibration under held-out noise", y=0.98, weight="bold", fontsize=18)
    fig.text(0.5, 0.925, "20 paired calibration draws · 2,200 utterances · four held-out recordings",
             ha="center", fontsize=11, color="#445266")
    lookup = {(row["series"], row["recording"], row["condition"]): row for row in rows}
    colors = {"float": "#202C3C", "int8_0pct": "#136D9D", "int8_50pct": "#BC541C",
              "int8_100pct": "#27796B"}
    markers = ("x", "o", "s", "^")
    conditions = ["clean", *(f"{snr} dB" for snr in SNRS)]
    for i, (name, color) in enumerate(colors.items()):
        label = "Float" if name == "float" else f"{MIXTURES[i - 1]}% speech augmented"
        cells = [[lookup[name, *CLEAN]], *(
            [lookup[name, r, c] for r in RECORDINGS] for c in conditions[1:]
        )]
        accuracy = [np.mean([cell["top1_accuracy_percent"] for cell in group]) for group in cells]
        gap = [np.mean([cell["added_error_pp"] for cell in group]) for group in cells]
        axs[0, 0].plot(conditions, accuracy, color=color, marker=markers[i], label=label,
                       linestyle="--" if i == 0 else "-", linewidth=1.35)
        if i:
            axs[0, 1].plot(conditions, gap, color=color, marker=markers[i], linewidth=1.35)
    axs[0, 0].set(title="Speech accuracy", ylabel="Accuracy", ylim=(0, 100))
    axs[0, 0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axs[0, 1].set(title="Quantization gap", ylabel="Float minus int8 accuracy (pp)")
    axs[0, 1].axhline(0, color="#8895A5", linewidth=0.8)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.903), ncol=4, frameon=False, fontsize=9)

    primary = summary["primary"]
    draws = summary["secondary_descriptive"]["calibration_draw_points"]
    contrast = axs[1, 0]
    contrast.axhspan(-PRACTICAL_MARGIN_PP, PRACTICAL_MARGIN_PP, color="#DFECE9", alpha=0.8)
    contrast.axhline(0, color="#8895A5", linewidth=0.8)
    contrast.scatter(np.arange(1, len(DRAWS) + 1), [row["50_minus_0_pp"] for row in draws],
                     color="#355976", s=19, zorder=3)
    lo, hi = primary["ci95_pp"]
    mean = primary["estimate_pp"]
    contrast.vlines(len(DRAWS) + 2, lo, hi, color="#BC541C", linewidth=1.6)
    contrast.scatter([len(DRAWS) + 2], [mean], color="#BC541C", marker="D", s=32, zorder=4)
    contrast.set_xticks([1, 5, 10, 15, 20, 22], ["1", "5", "10", "15", "20", "Mean"])
    contrast.set(title="Primary paired contrast", xlabel="Calibration draw",
                 ylabel="50% minus 0% added error (pp)")
    contrast.set_xlim(0, 23.5)

    heat = axs[1, 1]
    matrix = np.array([[lookup[f"int8_{m}pct", *cell]["added_error_pp"] for cell in NOISY]
                       for m in MIXTURES])
    limit = max(0.1, float(np.abs(matrix).max()))
    im = heat.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit))
    for j, recording in enumerate(RECORDINGS):
        heat.text(j * len(SNRS) + 1, 1.04, recording, ha="center", va="bottom", fontsize=8,
                  transform=heat.get_xaxis_transform())
        if j:
            heat.axvline(j * len(SNRS) - 0.5, color="white", linewidth=1)
    heat.set_xticks(np.arange(len(NOISY)), [str(snr) for _ in RECORDINGS for snr in SNRS])
    heat.set_yticks(np.arange(len(MIXTURES)), [f"{m}%" for m in MIXTURES])
    heat.set(xlabel="Signal-to-noise ratio (dB)", ylabel="Speech augmented")
    heat.set_title("Gap by held-out condition", y=1.20)
    fig.colorbar(im, ax=heat, fraction=0.05, pad=0.04).ax.set_title("pp", fontsize=9)

    for axis in (axs[0, 0], axs[0, 1], contrast):
        axis.grid(axis="y", color="#DDE3E9", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(0.085, 0.035, "Positive gaps mean extra int8 error. Shading shows ±0.5 pp. "
             "The interval holds recordings and noise offsets fixed.", fontsize=9, color="#445266")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="Paired .csv or .csv.gz")
    parser.add_argument("--output", required=True, type=Path, help="Directory for tables and plots")
    args = parser.parse_args()
    summary, cells, commands, draws = summarize(*read_predictions(args.predictions))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "conditions.csv", cells)
    write_csv(args.output / "commands.csv", commands)
    write_csv(args.output / "calibration_draws.csv", draws)
    plot_summary(summary, cells, args.output / "heldout_results.png")
    print(f"Wrote held-out analysis to {args.output}")


if __name__ == "__main__":
    main()
