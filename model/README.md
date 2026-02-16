# SignSync Model Documentation

## Overview

This directory contains the machine learning model for American Sign Language (ASL) classification.

## Current Model

**File:** `signsync_model.pkl`
**Type:** Custom neural network
**Format:** Pickle serialized dictionary
**Architecture:** 784 -> 128 -> 64 -> 24 (fully connected)
**Test Accuracy:** 85%

## Supported Model Types

SignSync supports multiple ML frameworks. Set the `MODEL_TYPE` environment variable accordingly:

| Type | Extension | Library | Environment Variable |
|------|-----------|---------|---------------------|
| `custom` | `.pkl` | Custom neural network (default) | `MODEL_TYPE=custom` |
| `sklearn` | `.pkl` | scikit-learn | `MODEL_TYPE=sklearn` |
| `tensorflow` | `.h5` | TensorFlow/Keras | `MODEL_TYPE=tensorflow` |
| `onnx` | `.onnx` | ONNX Runtime | `MODEL_TYPE=onnx` |

## Input Specifications

| Property | Value |
|----------|-------|
| Input Size | 28x28 pixels (configurable via `MODEL_INPUT_SIZE`) |
| Color Mode | Grayscale (1 channel) |
| Normalization | 0-1 range (pixel / 255.0) |
| Data Type | float32 |

### Input Shape by Model Type

- **custom:** Column vector `(784, 1)` for 28x28 images
- **sklearn:** Row vector `(1, 784)` for 28x28 images
- **tensorflow:** 4D tensor `(1, 28, 28, 1)`
- **onnx:** 4D tensor `(1, 28, 28, 1)`

## Output Specifications

| Property | Value |
|----------|-------|
| Classes | 24 ASL letters (A-Y, excluding J and Z) |
| Output | Probability distribution over classes |
| Confidence | Highest probability value (0.0 - 1.0) |

### Class Labels

```
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
```

Note: J and Z are excluded because they require motion to form, which cannot be captured in a static image.

## Configuration

Set these environment variables in your `.env` file:

```bash
MODEL_PATH=model/signsync_model.pkl
MODEL_TYPE=custom
MODEL_INPUT_SIZE=28
```

## Adding a New Model

1. Place your model file in this directory
2. Update `MODEL_PATH` in your environment
3. Set the correct `MODEL_TYPE`
4. Ensure input size matches via `MODEL_INPUT_SIZE`

### Example: Using a TensorFlow Model

```bash
MODEL_PATH=model/asl_cnn.h5
MODEL_TYPE=tensorflow
MODEL_INPUT_SIZE=224
```

## Model Training

If you need to train a new model, ensure:

1. Training data uses the same preprocessing (grayscale, normalized)
2. Input size is consistent across training and inference
3. Output classes match the `ASL_LABELS` in `app.py`

## Performance Notes

- Sklearn models load fastest and have lowest memory footprint
- TensorFlow models may require GPU for optimal performance
- ONNX provides good balance of speed and compatibility

## Troubleshooting

**Model not loading:**
- Check file path is correct
- Verify MODEL_TYPE matches the file format
- Check logs for specific error messages

**Poor accuracy:**
- Ensure input images are well-lit
- Hand should be centered in frame
- Background should be contrasting

**Memory issues:**
- Consider using ONNX quantized models
- Reduce MODEL_INPUT_SIZE if possible
