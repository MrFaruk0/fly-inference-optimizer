# TensorFly

TensorFly is an experimental closed-loop optimizer. It benchmarks
`Qwen/Qwen3.5-9B`, encodes measured serving metrics into a simulation whose
identities, connectivity, transmitter annotations and visible morphology come
from MaleCNS v1.0, then uses an engineered neural readout to select the next
real inference configuration.

> TensorFly uses reconstructed MaleCNS v1.0 anatomy, neuron identities,
> morphology, and synaptic connectivity. Neural dynamics, inference-metric
> encoding, reward modulation, plasticity, and the mapping from neural
> activity to Qwen runtime actions are engineered experimental approximations.

It does **not** claim that a fly brain understands or naturally optimizes
Transformer inference. Connectivity `weight` is a synaptic contact count, not
a calibrated electrophysiological conductance; transmitter-based effects are
an engineered receptor-agnostic approximation.

## Colab

Open `TensorFly_Colab.ipynb` in a paid CUDA Colab runtime and run all cells.
The normal path downloads, checksums and caches the official files itself—no
manual upload or environment-variable data path:

```python
import tensorfly

tensorfly.prepare()
experiment = tensorfly.TensorFlyExperiment(model="Qwen/Qwen3.5-9B")
experiment.run(prompt_corpus=tensorfly.DEFAULT_PROMPTS, trials=20)
experiment.compare_baselines()
experiment.replay()
experiment.export_video()
```

The initial preparation downloads approximately 1.1 GB of official source
tables and needs a high-RAM Colab. Subsequent runs reuse checksum-verified
sources and derived arrays. Qwen is deliberately lazy: importing or preparing
the dataset never loads model weights.

## Data provenance

Normal execution is fail-closed. `prepare_malecns()` fetches the version-pinned
Janelia bulk files, verifies the following byte counts and SHA-256 digests,
and rejects missing, corrupt, or unprovenanced data:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| body annotations | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| body neurotransmitters | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| connectome weights | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

The default retention policy is the documented fly-wirehead-compatible policy:
nonempty `super_class`, excluding `Glia`, retaining every released directed
edge between retained bodies. It produces exactly 166,700 neurons and
25,582,938 directed edges for the official inputs. The preparation report
records source/retained rows, exclusions, contact count, isolates, policy and
source hashes. It preserves `uint64` biological body IDs and writes a sorted,
reversible compact mapping alongside `pre_index`, `post_index`, and
`synapse_count` arrays.

The upstream source is [Janelia's MaleCNS v1.0 download page](https://male-cns.janelia.org/download/),
which documents the [CC-BY licence](https://creativecommons.org/licenses/by/4.0/).
The lock values and retention architecture are attributed to
[`mattyhempstead/fly-wirehead`](https://github.com/mattyhempstead/fly-wirehead);
TensorFly independently implements them rather than copying unlicensed code.

Synthetic data is only available as `synthetic_dev=True` / `--synthetic-dev`
for tests and developer diagnostics. It is never a fallback for normal runs.

## Viewer and replay

`prepare_malecns()` downloads selected official SWC skeleton objects, records
their individual checksums, and converts them into
`tensorfly-real-morphology/1`. The Three.js viewer uses typed
`BufferGeometry` line segments built from those coordinates; it refuses
synthetic/procedural geometry and keeps body IDs as strings in JavaScript to
avoid `uint64` precision loss. Optional connection lines are emitted only from
actual retained edges.

Replay frames are produced by the experiment log, not fabricated from spikes:
they contain the recorded Qwen metrics, configuration transitions, and
simulation activity keyed by real body ID.

## Fair experiment protocol

Every comparison uses the same model revision, prompt corpus, warmup,
evaluation count and fixed `max_new_tokens`. The reported methods are fixed
default configuration, seeded random search, hill climbing and TensorFly.
TTFT is timestamped at the first generated token; it is not total generation
time. Prefill uses a full prompt forward pass, decode/TPOT excludes first-token
time, and CUDA peak allocated/reserved memory is sampled after synchronization.
