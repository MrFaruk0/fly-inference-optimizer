# TensorFly — MaleCNS-scale connectome inference on Colab

TensorFly scaffolds MaleCNS-scale connectome simulation + Qwen inference benchmarking,
with a standalone Three.js viewer for measured replays.

Start here: open **`TensorFly_Colab.ipynb`** in Colab and run top-to-bottom.

## Paid Colab profiles

| Colab GPU | TensorFly profile | dtype | `batch_size` | Qwen model |
|---|---|---|---|---|
| A100 40/80GB | `A100` | bfloat16 | 1024 | `Qwen/Qwen3.5-9B` |
| L4 24GB | `L4` | float16 | 512 | `Qwen/Qwen3.5-9B` |
| T4 16GB | `T4` | float16 | 256 | `Qwen/Qwen3.5-4B` (fallback notice) |
| CPU / other | `CPU` | float32 | 64 | `Qwen/Qwen3.5-4B` (fallback notice) |

The notebook's startup cell prints GPU name, VRAM, system RAM, CUDA version,
PyTorch version, profile, model, and device (`Runtime.startup_summary`).

## Explicit fallback (printed, never silent)

- Known name (`A100` / `L4` / `T4` substring) → that profile.
- Unknown CUDA GPU name → prints
  `[tensorfly.runtime] Unknown GPU ...` and uses the **`T4`** profile
  (safe lowest-common GPU denominator).
- No CUDA GPU → prints `[tensorfly.runtime] No CUDA GPU detected.`
  and uses the **`CPU`** profile.
- Qwen model: `A100`/`L4` with ≥ 20 GB VRAM (or unknown VRAM) → `Qwen/Qwen3.5-9B`;
  anything else (`T4`/`CPU`/unknown/low VRAM) → `Qwen/Qwen3.5-4B` with the exact line:

```text
TensorFly fallback: Qwen3.5-4B because current GPU memory is insufficient for the 9B benchmark profile.
```

## Real MaleCNS source requirement / honesty

Full network target: **166,700 neurons / 25,600,000 edges** (MaleCNS v1.0).

- Provide a real export via `MALECNS_EDGE_SOURCE`:
  - `.npz` CSR dump (`row_ptr`/`col_idx`/`weights` or `indptr`/`indices`/`data`),
    or edgelist arrays (`sources`/`targets`[, `weights`]);
  - or `.csv` edgelist with `source,target[,weight]` header.
- Pass it to `MaleCNSSimulation(config, edge_source=...)`.
- Without it, `build()` creates a **synthetic scaffold** explicitly flagged
  `is_synthetic=True`, `data_source="synthetic-scaffold (NOT real MaleCNS v1.0)"`.
  A missing/unreadable path logs `missing-source` / `source-load-failed` and stays
  synthetic. Synthetic output is only for memory/layout benchmarking — never
  presented as real MaleCNS data.

## Commands

```bash
# install (Colab cell 0 does this)
pip install -e .[inference]

# smoke benchmark (fast, runs automatically in the notebook)
python -c "from tensorfly import *; s=MaleCNSSimulation(SimulationConfig(num_neurons=5000,num_edges=20000)); s.build(); print(run_benchmark(s, steps=5, repeats=2).to_dict())"

# full-target build (gated; needs a high-RAM paid runtime + real source)
export MALECNS_EDGE_SOURCE=/path/to/malecns_edges.npz

# serve the viewer (must be HTTP, not file://)
python -m http.server 8000 --directory viewer
# open http://localhost:8000/?replay=./tensorfly_replay.json
```

The notebook separates **benchmark mode** (`run_benchmark`, timed `sim.step()`
loop; build excluded) from **viewer/replay mode** (`ReplayRecorder` snapshots
→ `viewer/tensorfly_replay.json`). The full-target cell defines the memmap/cache
build (`cache_dir="cache/malecns"`, `use_memmap=True`) but does not execute it
unless `RUN_FULL_TARGET = True`.

## Video output workflow

1. Generate `viewer/tensorfly_replay.json` (notebook section 6, real
   simulation-derived activity).
2. Serve `viewer/` and open `?replay=./tensorfly_replay.json`
   (header flips from `SAMPLE SCAFFOLD · not measured` to
   `LIVE REPLAY · measured snapshots`).
3. Press **Record demo** (records `canvas.captureStream(60)` video + ~4 Hz
   state JSON), then **Stop** → **Download Video** (`.webm`) /
   **Download JSON** (reloadable `fly-cns-replay/1`).
4. Optional MP4: set the endpoint field (or `window.__MP4_CONVERT_URL__`);
   the viewer POSTs the WebM and downloads the MP4 response, keeping WebM
   as fallback.

See `viewer/README.md` for the replay schema, controls, and MP4 hook details.
