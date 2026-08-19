# Face Detection

Browser-based face detection using ONNX Runtime Web and WebGPU, with YOLOv8-based face detection models.

```javascript
import { FaceDetector } from "yolov8-face-detector";

// Create the detector
const detector = new FaceDetector({
    model: "standard",
    executionProviders: ["webgpu", "wasm"]
});

// Load the model
await detector.load();

// Detect faces
const result =
    await detector.detect(image);
```

## Models

The package supports three model-loading strategies.

### Standard model

```js
const detector =
    new FaceDetector({
        model: "standard"
    });
```
Uses the standard `yolov8n` model.
`/models/yolov8n-face.onnx`

### Lite model
```js
const detector =
    new FaceDetector({
        model: "lite"
    });
```
Uses the lightweight `yolov8-lite-s` model.
`/models/yolov8-lite-s-face.onnx`

### Custom model
```js
const detector =
    new FaceDetector({
        publicPath:
            "/models/custom-model.onnx"
    });
```
Uses a custom model specified by the `publicPath`.
`/models/custom-model.onnx`

Models can be loaded from:

* Local files
* Remote URLs
* CDN URLs or CDNs

## Local hosting
```text
public/
└── models/
    ├── yolov8n-face.onnx
    ├── yolov8-lite-s.onnx
    └── custom-model.onnx
```

## Model Performance Study

This project supports multiple YOLOv8-based face detection models.

The purpose of this study is to compare the standard `yolov8n` model with the lightweight `yolov8-lite-s` model for browser-based face detection using ONNX Runtime Web and WebGPU.

The primary goal is to determine whether the lightweight model provides a useful efficiency advantage for resource-constrained environments while maintaining comparable detection behavior.

> **Important:** The benchmark in this document is a local performance study, not a formal accuracy evaluation. Model accuracy claims are based on the reported WIDERFace results provided with the respective models.

---

## Models

| Model | Input Size | Model Size | Intended Role |
|---|---:|---:|---|
| YOLOv8n | 640 × 640 | ~12 MB | Standard / quality |
| YOLOv8-Lite-S | 640 × 640 | ~7.4 MB | Lightweight / resource-constrained |

### Reported WIDERFace Performance

The reported WIDERFace validation results are:

| Model | Test Size | Easy | Medium | Hard | Model Size |
|---|---:|---:|---:|---:|---:|
| YOLOv8-Lite-S | 640 | 93.4 | 91.2 | 78.6 | 7.4 MB |
| YOLOv8n | 640 | 94.6 | 92.3 | 79.6 | 12 MB |

> Easy, Medium, and Hard refer to mAP on the WIDERFace validation subsets.

The reported results show that YOLOv8n has higher WIDERFace mAP, while YOLOv8-Lite-S has a substantially smaller model footprint.

---

# Benchmark Test Environment

All browser benchmark results documented below were collected on the following system.

## Hardware

| Component | Specification |
|---|---|
| **CPU** | Intel Core i3-9100 @ 3.60 GHz |
| **CPU Cores / Threads** | 4 cores / 4 threads |
| **GPU** | MSI NVIDIA GeForce GTX 1050 OC |
| **VRAM** | 2 GB |
| **RAM** | 16 GB ADATA XPG |
| **RAM Speed** | ~2444 MHz effective operating speed |
| **Motherboard** | ASUS EX-H310M-V3 R2.0 |
| **GPU Acceleration** | WebGPU |

> The RAM is operating below its rated speed due to the platform/motherboard configuration. The benchmark was performed using the system's actual configuration rather than an overclocked or otherwise optimized configuration.

## Software

| Component | Specification |
|---|---|
| **Operating System** | Windows 11 |
| **Browser** | Google Chrome |
| **Inference Runtime** | ONNX Runtime Web |
| **Execution Provider** | WebGPU |
| **Input Resolution** | 640 × 640 |
| **Benchmark Iterations** | 50 |
| **Warmup** | 1 inference |

> The exact Chrome and ONNX Runtime Web versions should be recorded here for future reproducibility.

## WebGPU Hardware Detection

During the benchmark, Chrome successfully exposed the NVIDIA GPU through WebGPU.

The application reported:

```text
WebGPU available: true
GPU adapter: NVIDIA GPUAdapter
Vendor: NVIDIA
Architecture: Pascal
````

The NVIDIA GTX 1050 has 2 GB of VRAM according to the hardware configuration listed above.

The benchmark therefore represents GPU-accelerated browser inference rather than CPU-only inference.

## Benchmark Context

These results should be interpreted specifically within the following environment:

```text
CPU:         Intel Core i3-9100 @ 3.60 GHz
Cores:       4
Threads:     4
GPU:         MSI NVIDIA GTX 1050 OC
VRAM:        2 GB
RAM:         16 GB ADATA XPG
RAM Speed:   ~2444 MHz
Motherboard: ASUS EX-H310M-V3 R2.0
OS:          Windows 11
Browser:     Google Chrome
Runtime:     ONNX Runtime Web
Backend:     WebGPU
Input:       640 × 640
Iterations:  50
Warmup:      1 inference
```

Because browser/WebGPU performance depends on the GPU, browser version, driver, CPU, memory configuration, operating system, and runtime implementation, these measurements should not be treated as universal performance numbers.

They should instead be considered a **reference benchmark for this specific test system and software environment**.

---

# Browser WebGPU Benchmark

The models were benchmarked using the project's JavaScript implementation with:

* ONNX Runtime Web
* WebGPU execution provider
* 640 × 640 input
* Browser-based inference
* 50 measured inference iterations
* One warmup inference
* End-to-end `detector.detect()` timing

The benchmark measures the complete detection operation rather than only the raw ONNX `session.run()` call.

Therefore, the measured time represents the application's detection pipeline, including:

```text
Image
  ↓
Preprocessing
  ↓
ONNX Runtime / WebGPU
  ↓
Postprocessing
  ↓
NMS
  ↓
Detection result
```

This makes the benchmark useful for evaluating the actual browser detection experience, although it should not be interpreted as pure neural-network inference latency.

---

## Benchmark Results

### YOLOv8n

```text
Iterations: 50
Average: 103.85 ms
Min: 92.30 ms
Max: 157.30 ms
```

### YOLOv8-Lite-S

```text
Iterations: 50
Average: 98.49 ms
Min: 87.30 ms
Max: 151.60 ms
```

## Comparison

| Metric                   |   YOLOv8n | YOLOv8-Lite-S |         Difference |
| ------------------------ | --------: | ------------: | -----------------: |
| Average                  | 103.85 ms |  **98.49 ms** | **5.36 ms faster** |
| Minimum                  |  92.30 ms |  **87.30 ms** | **5.00 ms faster** |
| Maximum                  | 157.30 ms | **151.60 ms** |  **5.70 ms lower** |
| Model Size               |    ~12 MB |   **~7.4 MB** |   **~38% smaller** |
| Detections on test image |       233 |           232 |            1 fewer |

> The detection count was obtained from the same test image and should not be interpreted as a formal accuracy measurement.

---

## Average Performance

YOLOv8-Lite-S achieved an average end-to-end detection time of:

```text
98.49 ms
```

compared with:

```text
103.85 ms
```

for YOLOv8n.

The measured difference was:

```text
103.85 - 98.49 = 5.36 ms
```

Relative to YOLOv8n:

```text
(103.85 - 98.49) / 103.85 × 100
≈ 5.16%
```

Therefore, YOLOv8-Lite-S was approximately **5.16% faster** in this particular 50-iteration browser benchmark.

This result applies specifically to the reference test environment and should not be generalized to other GPUs or mobile devices without additional testing.

---

## Model Size

Using the reported model sizes:

```text
YOLOv8n       ≈ 12 MB
YOLOv8-Lite-S ≈ 7.4 MB
```

The lightweight model is approximately **38% smaller** based on these rounded model-size figures.

A smaller model can potentially reduce:

* Initial download size
* Network bandwidth usage
* Browser storage requirements
* Model loading overhead
* Memory pressure

Actual model loading and memory behavior should be benchmarked separately.

---

# Detection Comparison

On the same benchmark image, the models produced:

```text
YOLOv8n       → 233 detections
YOLOv8-Lite-S → 232 detections
```

The difference was one detected face.

This result is useful as an observation from the test image, but it **does not establish equivalent accuracy**.

A single image cannot determine whether one model is more accurate than another.

The reported WIDERFace results indicate that YOLOv8n has higher mAP than YOLOv8-Lite-S.

Therefore:

* YOLOv8n has higher reported WIDERFace accuracy.
* YOLOv8-Lite-S has a substantially smaller model.
* YOLOv8-Lite-S was slightly faster in the local WebGPU benchmark.
* Both models produced nearly identical detection counts on the benchmark image.

---

# Performance Interpretation

## YOLOv8n

### Advantages

* Higher reported WIDERFace mAP
* Better suited to applications prioritizing detection quality
* Suitable as the standard model

### Disadvantages

* Larger model footprint
* Slightly slower in the current WebGPU benchmark
* Higher download and storage requirements

## YOLOv8-Lite-S

### Advantages

* Approximately 38% smaller based on the reported model sizes
* Approximately 5% faster in the current benchmark
* Very similar detection count on the test image
* Better suited to resource-constrained environments

### Disadvantages

* Lower reported WIDERFace mAP
* May have reduced performance on difficult faces
* Has not yet been comprehensively tested on actual mobile hardware

---

# Preliminary Model Strategy

Based on the current evidence, the project should retain both models rather than replacing YOLOv8n entirely.

```text
                    Face Detection
                          │
              ┌───────────┴───────────┐
              │                       │
        Quality-oriented        Efficiency-oriented
              │                       │
              ▼                       ▼
          YOLOv8n               YOLOv8-Lite-S
              │                       │
       Desktop / higher          Mobile /
       accuracy priority         constrained devices
```

## Proposed Roles

| Environment                 | Recommended Model | Reason                      |
| --------------------------- | ----------------- | --------------------------- |
| Desktop                     | YOLOv8n           | Higher reported accuracy    |
| High-performance device     | YOLOv8n           | Accuracy priority           |
| Mobile                      | YOLOv8-Lite-S     | Smaller and slightly faster |
| Resource-constrained device | YOLOv8-Lite-S     | Lower model footprint       |
| Poor network conditions     | YOLOv8-Lite-S     | Smaller download            |

> These are preliminary recommendations. Actual mobile-device benchmarking is required before making a definitive mobile model recommendation.

---

# WebGPU Execution

The models are executed through ONNX Runtime Web using the WebGPU execution provider when WebGPU is available.

The browser test confirmed:

```text
WebGPU available: true
GPU adapter: NVIDIA GPUAdapter
```

ONNX Runtime also reported that some nodes were not assigned to the preferred execution provider.

This does **not** necessarily indicate that WebGPU failed.

Certain operations, including shape-related operations, may be assigned to CPU while other operations use the preferred execution provider.

The project should therefore distinguish between:

```text
WebGPU unavailable
```

and:

```text
WebGPU available with some CPU-assigned operations
```

---

# Benchmark Limitations

The current benchmark is an initial performance study rather than a formal model evaluation.

## Hardware

The benchmark was performed on a desktop system using an NVIDIA GTX 1050 through Chrome/WebGPU.

Performance on other GPUs and mobile devices may differ substantially.

## Dataset

The detection-count comparison was performed using a single test image.

Therefore:

```text
233 vs 232
```

does not establish model accuracy equivalence.

## Browser and Runtime

The results are specific to the tested:

* Browser
* Browser version
* WebGPU implementation
* GPU driver
* ONNX Runtime Web version
* Operating system
* Hardware configuration
* Application implementation

Different environments may produce different results.

## Benchmark Scope

The current benchmark measures end-to-end `detector.detect()` time.

It does not separately measure:

* Model loading time
* Preprocessing time
* Raw ONNX inference time
* Postprocessing time
* NMS time
* Memory consumption

These should be measured separately in future benchmarking.

## Sample Size

The current performance benchmark uses 50 measured inference iterations after one warmup inference.

Although this provides a more useful sample than a very small benchmark, additional runs and devices are required for a broader performance study.

---

# Future Benchmarking

Future benchmarks should collect:

```text
Average detection time
Median detection time
P95 detection time
Minimum detection time
Maximum detection time

Model loading time
Preprocessing time
Raw inference time
Postprocessing time
NMS time

Memory usage
Model download size
Detection count
```

Testing should also include multiple image categories:

```text
Single face
Small faces
Large faces
Crowded scenes
Low-resolution images
Side profiles
Partially occluded faces
Difficult lighting
```

Hardware testing should eventually include:

```text
Desktop NVIDIA GPU
Desktop integrated GPU
Laptop integrated GPU
Android devices
iOS devices
Low-end mobile hardware
```

This will provide a stronger basis for automatic model selection.

---

# Current Conclusion

The current browser benchmark shows that **YOLOv8-Lite-S is a promising lightweight model for this project**.

Compared with YOLOv8n, it:

* Uses approximately **38% less model storage** based on the reported model sizes
* Was approximately **5.16% faster** in the 50-run WebGPU benchmark
* Had a slightly lower minimum and maximum measured detection time
* Produced **232 detections compared with 233** on the test image

However, YOLOv8n has higher reported WIDERFace mAP, so YOLOv8-Lite-S should **not currently be considered an accuracy-equivalent replacement**.

The current architecture should therefore retain both models:

```text
YOLOv8n
    ↓
Standard / quality-oriented model

YOLOv8-Lite-S
    ↓
Lightweight / resource-constrained model
```

Future versions can implement automatic model selection based on device capabilities after additional hardware testing.
