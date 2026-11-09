"""Convert and evaluate the frozen MLPerf Tiny keyword spotter on CPU.

Use TF_USE_LEGACY_KERAS=1 with the pilot's pinned TensorFlow environment.
"""

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf


DEFAULT_SAVED_MODEL = (
    Path(__file__).resolve().parents[1]
    / "mlperf-tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model"
)
FEATURE_SHAPE = (49, 10, 1)
CALIBRATION_COUNT = 120
CLASS_COUNT = 12


def _feature_batch(features):
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 4 or features.shape[1:] != FEATURE_SHAPE or not len(features):
        raise ValueError(f"Expected a nonempty feature batch with shape (N, {FEATURE_SHAPE}).")
    if not np.isfinite(features).all():
        raise ValueError("Features contain a non-finite value.")
    return features


def convert_float(saved_model=DEFAULT_SAVED_MODEL):
    """Export the SavedModel without quantization or weight compression."""
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = []
    return converter.convert()


def convert_int8(features, saved_model=DEFAULT_SAVED_MODEL):
    """Export a fully integer model using one 120-example calibration set."""
    features = _feature_batch(features)
    if len(features) != CALIBRATION_COUNT:
        raise ValueError(f"Expected exactly {CALIBRATION_COUNT} representative examples.")

    def representative_data():
        for example in features:
            yield [example[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _predict_scores(model_bytes, features, *, legacy_cast=False):
    features = _feature_batch(features)
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    if tuple(inp["shape"]) != (1, *FEATURE_SHAPE) or tuple(out["shape"]) != (1, CLASS_COUNT):
        raise ValueError("Model has unexpected input or output dimensions.")

    clamp_counts = np.zeros(len(features), dtype=np.int32)
    if inp["dtype"] == np.int8:
        scale, zero_point = inp["quantization"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Model input does not have a valid int8 scale.")
        codes = np.rint(features / scale) + zero_point
        clamp_counts = ((codes < -128) | (codes > 127)).sum(axis=(1, 2, 3), dtype=np.int32)
        if legacy_cast:
            # Match the upstream evaluator when auditing the historical artifact.
            encoded = (features / scale + zero_point).astype(np.int8)
            encoding = "upstream direct cast"
        else:
            encoded = np.clip(codes, -128, 127).astype(np.int8)
            encoding = "nearest-even rounding and saturation"
    elif inp["dtype"] == np.float32:
        encoded = features
        encoding = "float32"
    else:
        raise ValueError(f"Unsupported model input type: {inp['dtype']}")

    scores = np.empty((len(features), CLASS_COUNT), dtype=out["dtype"])
    for i in range(len(features)):
        interpreter.set_tensor(inp["index"], encoded[i:i + 1])
        interpreter.invoke()
        scores[i] = interpreter.get_tensor(out["index"])[0]
    if not np.isfinite(scores).all():
        raise ValueError("Model output contains a non-finite value.")

    metadata = {
        "bytes": len(model_bytes),
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "input_shape": inp["shape"].tolist(),
        "output_shape": out["shape"].tolist(),
        "input_dtype": np.dtype(inp["dtype"]).name,
        "output_dtype": np.dtype(out["dtype"]).name,
        "input_quantization": list(inp["quantization"]),
        "output_quantization": list(out["quantization"]),
        "input_encoding": encoding,
        "tensor_dtype_counts": dict(Counter(
            np.dtype(tensor["dtype"]).name for tensor in interpreter.get_tensor_details()
        )),
    }
    return scores, clamp_counts, metadata


def predict(model_bytes, features, *, legacy_cast=False):
    """Return label indices, per-clip values needing saturation, and model metadata.

    With legacy_cast, the saturation counts still describe the values that
    need clamping, even though the historical evaluator did not clamp them.
    """
    scores, clamp_counts, metadata = _predict_scores(
        model_bytes, features, legacy_cast=legacy_cast
    )
    # The output quantization scale is positive, so argmax is unchanged by dequantizing.
    return np.argmax(scores, axis=1), clamp_counts, metadata


def check_float_parity(model_bytes, features, saved_model=DEFAULT_SAVED_MODEL):
    """Compare the true float export with the original serving signature."""
    features = _feature_batch(features)
    scores, _, metadata = _predict_scores(model_bytes, features)
    if metadata["input_dtype"] != "float32" or metadata["output_dtype"] != "float32":
        raise ValueError("Float parity requires a model with float32 inputs and outputs.")
    with tf.device("/CPU:0"):
        serving = tf.saved_model.load(str(saved_model)).signatures["serving_default"]
        input_name, = serving.structured_input_signature[1]
        expected, = serving(**{input_name: tf.convert_to_tensor(features)}).values()
    expected = expected.numpy()
    return {
        "clips": len(features),
        "max_abs_score_error": float(np.max(np.abs(scores - expected))),
        "argmax_disagreements": int(np.sum(
            np.argmax(scores, axis=1) != np.argmax(expected, axis=1)
        )),
        "allclose": bool(np.allclose(scores, expected, rtol=1e-4, atol=1e-5)),
    }
