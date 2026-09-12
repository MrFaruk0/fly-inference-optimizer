# TensorFly

TensorFly is a Google Colab-first experiment asking whether an engineered controller operating on reconstructed MaleCNS v1.0 structure can select real runtime settings for `Qwen/Qwen3.5-9B`.

> TensorFly uses reconstructed MaleCNS v1.0 anatomy, neuron identities, morphology and structural connectivity. Neural dynamics, inference-metric encoding, reward modulation and mappings from neural activity to Qwen runtime actions are engineered experimental approximations.

It does not claim a biologically complete fly brain, that a fly understands Qwen, or that measured fly dopamine/physiology optimizes Transformers.

## Run in Colab

Open [TensorFly_Colab.ipynb](TensorFly_Colab.ipynb) in a CUDA Google Colab runtime and use **Runtime → Run all**. There is no local desktop, Docker, Blender, neuVid, MuJoCo, Flybody, FlyGym, or NeuroMechFly dependency.

The notebook clones this repository, installs current Hugging Face Transformers (Qwen3.5's documented requirement), reports GPU/RAM/Torch/CUDA, prepares MaleCNS, runs the experiment and baselines, serves the Three.js viewer, and exposes browser-native video capture.

`Qwen/Qwen3.5-9B` is requested by default. A100 is the primary target, L4 is supported, and a T4 is supported with an explicit visible fallback to `Qwen/Qwen3.5-4B` when its 16 GB-class VRAM makes direct 9B loading impractical. The actual model ID, revision when supplied, dtype, GPU, VRAM, Torch, Transformers and CUDA are persisted in benchmark rows. No model substitution is silent.

```python
import tensorfly

prepared = tensorfly.prepare()  # official sources only; fail closed
experiment = tensorfly.TensorFlyExperiment(model="Qwen/Qwen3.5-9B")
results = experiment.run(prompt_corpus=tensorfly.DEFAULT_PROMPTS, trials=20)
baselines = experiment.compare_baselines(prompt_corpus=tensorfly.DEFAULT_PROMPTS)
experiment.replay()
experiment.export_video()  # writes viewer/tensorfly_replay.json
```

`export_video()` deliberately exports measured replay data rather than trying to render on the benchmark GPU. In the served viewer use **Cinematic Replay**, **Record Demo**, **Stop Recording**, and **Download Video**. It uses `canvas.captureStream()` and `MediaRecorder` to create `tensorfly_demo.webm`. Recording happens after Qwen measurements and cannot perturb them.

## Biological data and provenance

Normal preparation is fail-closed:

```text
cache → SHA-256 validation → otherwise official Janelia download
      → schema/filter validation → real compact graph + manifest
```

The pinned [official MaleCNS v1.0 download release](https://male-cns.janelia.org/download/) supplies curated annotations, neurotransmitter predictions, the full weighted connectivity table, and selected body-ID-addressed SWC skeletons. The source is CC-BY. Source URL, exact byte count and SHA-256 are stored for each bulk table; selected skeleton object URL, size and SHA-256 are stored independently.

The preparation manifest records the release, preprocessing version and time, source annotation/edge rows, retained neuron/edge/contact counts, exclusions, isolates, retention policy, and source hashes. The documented policy keeps nonempty, non-Glia superclass annotations and all released directed edges between those bodies. Its locked official result is exactly 166,700 retained neurons and 25,582,938 directed edges; it is not a rounded integrity check.

Biological body IDs remain `uint64` in Python. They are sorted into reversible compact graph indices, and the browser receives decimal strings so JavaScript never rounds them through `Number`.

There is no automatic synthetic fallback. `prepare_malecns()` raises on a download, hash, schema, accounting, preprocessing, or source-load failure. `synthetic_dev=True` is an explicit tiny developer/test fixture only.

## Populations, simulation, and controller

Sensory, dopaminergic/modulatory, and controller/readout registries are built from actual annotation fields and save both real body IDs and compact indices. The selection criteria and source transmitter metadata are written to `populations.json`; no positional graph slices are used. The explicit criteria select R1–R8/annotated sensory entries, dopamine/DA or PAM/PPL annotations, and named descending/motor cell types or superclass terms.

The deterministic `SensoryEncoder` normalizes measured prefill latency, true TTFT, TPOT, throughput, allocated VRAM and reserved VRAM, then drives only resolved real sensory indices. An absent stimulus is exactly zero drive.

```text
InferenceConfig → measured Qwen benchmark → SensoryEncoder
→ real MaleCNS graph/LIF approximation → real controller population readout
→ actual next Qwen config → measured reward → real dopamine-population current
```

The action decoder can change actual generation-time batch size or KV-cache usage. Attention implementation and `torch.compile` are only exposed when a caller has validated them for the current hardware; they are not decorative metadata. Every trial stores input/next config, raw and normalized metrics, sensory drive, neural state, readout, modulatory target/current, action, reward, and fixed-workload signature.

## Measurement and comparison

The Qwen path uses the current official `AutoProcessor` + `AutoModelForMultimodalLM` loading surface, left padding, greedy generation, warmup, synchronization, CUDA events for prefill, and a first-generated-token streamer for true TTFT. It records prefill latency, TTFT, decode latency, TPOT, tokens/sec, end-to-end latency, generated-token count, and peak CUDA allocated/reserved memory. Visualization is not active while those timings are recorded.

`compare_baselines()` gives fixed default, seeded random search, simple hill climbing and TensorFly the same model, prompt corpus, warmup, trial budget, and `max_new_tokens`. It reports all raw rows for best/median score, TTFT, TPOT, throughput, memory, and evaluations-to-best; it does not conceal a TensorFly loss.

## Three.js replay

`prepare()` selects a deterministic real-morphology subset in this order: sensory, dopamine, controller, then evenly spaced retained context bodies. It downloads official SWCs, performs source-vertex-only decimation while preserving branch ancestry, and writes `viewer/real-morphology.json`.

The browser renders this subset in population batches using `BufferGeometry`, typed arrays, dynamic vertex colours, and additive blending. No production geometry, soma cloud, fibre, link, or activity is random or generated. Actual recorded activity, sensory current, and modulatory current are keyed by real body ID, so the matching real skeleton brightens. Camera movement is aesthetic; neural events are replayed recorded state.

The independent [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) was reviewed for browser replay/UI ideas but is not a dependency or copied code; TensorFly needs real skeletons and avoids inheriting its custom license. [AxonWeave](https://github.com/dhakalnirajan/axonweave) was evaluated as a potential substrate but is not required: its public description did not establish a Colab-verifiable, checksum-pinned official MaleCNS provisioning and morphology path matching this experiment. TensorFly therefore uses the smaller direct Janelia preparation layer. NAVis remains optional because the shipped SWC parser is sufficient.

## Tests

CI uses only small fixtures; it never downloads the 1.1 GB graph or Qwen. It covers fail-closed preparation, exact body-ID round trips, annotation-backed population selection, zero unstimulated drive, deterministic encoding, real config mutation, fixed workloads, true TTFT behavior, recorded replay metrics, and rejection of procedural/synthetic viewer anatomy.
