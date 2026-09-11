# MaleCNS Viewer (standalone Three.js frontend)

Polished, dark cinematic 3D viewer for MaleCNS inference runs. No build step, no backend
dependency. All neuron state lives in batched `BufferGeometry` typed buffers
(3 `Points` draw calls + lines + dust) — there are no per-neuron JS objects.

## Run it (must be served over HTTP)

ES modules + `canvas.captureStream` require `http(s)`, not `file://`.

```bash
# from the repo root
python -m http.server 8000 --directory viewer
# open http://localhost:8000/
```

Alternatives: `npx serve viewer -l 8000`, or VS Code "Live Server" on `viewer/index.html`.

Deep-link a replay:

```text
http://localhost:8000/?replay=./sample-replay.json
http://localhost:8000/?replay=https://example.com/runs/trial-3.json
```

## Replay JSON schema (`fly-cns-replay/1`)

Top-level object:

```json
{
  "schema": "fly-cns-replay/1",
  "meta": {
    "model": "fly-qwen-7b",
    "prompt": "describe the optic lobe response",
    "populations": { "sensory": 3800, "dopamine": 550, "controller": 1300 },
    "config": { "id": "cfg-A", "lr": 0.001 },
    "best": { "config": "cfg-C", "reward": 0.91 }
  },
  "layout": {
    "positions": [[x, y, z], "..."],
    "regions": [0, 1, 2, "..."]
  },
  "frames": [
    {
      "t": 0.0,
      "trial": 0,
      "tokens": 12,
      "throughput_tps": 42.5,
      "ttft_ms": 118.2,
      "decode_ms_per_token": 23.1,
      "reward": 0.62,
      "config_id": "cfg-A",
      "config": { "id": "cfg-A", "lr": 0.001 },
      "event": { "type": "config_change", "label": "cfg-A -> cfg-B" },
      "sensory": [0.0, 0.31, "..."],
      "dopamine": [0.5, "..."],
      "controller": [0.1, "..."]
    }
  ]
}
```

Rules:

| field | required | notes |
|---|---|---|
| `frames` | yes | non-empty, sorted by `t` (viewer sorts anyway). Snapshot-driven: **every** visual/metric value is interpolated from these. |
| `frames[].t` | yes | seconds, monotonic. Playback interpolates linearly between neighbors. |
| `frames[].sensory / dopamine / controller` | one form required | dense arrays in `[0,1]` (clamped). Lengths should match `meta.populations`. Shorter arrays are **zero-filled (tails stay 0, never edge-repeated or stale scaffold)** and the measured prefix per frame (`sN`/`dN`/`cN`) is tracked; longer ones truncated. Interpolation only touches the measured prefix and explicitly zeroes tails. |
| `frames[].activity` | alternative | single global array split sequentially `sensory \| dopamine \| controller`. Short tails stay zero; measured prefix tracked as above. |
| `frames[].dopamine_mean` | alternative | scalar broadcast fill for dopamine (slow global signal). Counts as full dopamine coverage for that frame. |
| `frames[].event` | no | `{ "type": "config_change" \| "best" \| "trial_end", "label": "..." }`. **Only** these create event banners / log entries. Nothing is fabricated. |
| `meta.populations` | recommended | defaults to the viewer's built-in counts (3800/550/1300). `meta.num_neurons` accepted as fallback (split 66/10/24%). |
| `layout` | no | `positions` (`N×3`) + `regions` (`0=sensory, 1=dopamine, 2=controller`). If absent, the viewer uses its deterministic built-in anatomy-inspired layout. |

Population visual signatures:

- **sensory** (cyan): small points, fast flicker — driven by high-frequency content in `sensory[]`.
- **dopamine** (amber): large soft points, slow broadcast — mean of `dopamine[]` also drives fog density + link-line opacity.
- **controller** (magenta): medium points, rhythmic bursts — `controller[]` burst structure.

A minimal valid replay is `{ "frames": [{ "t": 0, "sensory": [...], ... }] }`.

See [`sample-replay.json`](./sample-replay.json) (48 frames, small populations, clearly-labeled sample data).

## Honesty rule: sample scaffold vs measured replay

- With no replay loaded, the viewer generates a **procedural sample scaffold** (seeded, deterministic)
  and is **always** badged `SAMPLE SCAFFOLD · not measured` in the header.
- Loading any replay JSON (file, URL, or `?replay=`) switches the badge to
  `LIVE REPLAY · measured snapshots` **only when coverage is full** (every frame measured
  at least the built-in buffer widths 3800/550/1300). Smaller or ragged replays (e.g.
  `sample-replay.json` with 120/24/48) show a qualified
  `REPLAY · partial coverage s:120/3800 d:24/550 c:48/1300 — only first N measured` badge,
  and the run panel lists `measured s:…/… d:…/… c:…/…` plus `provenance: replay:<label>`.
  All HUD values come from interpolated snapshots; unmeasured tails render as zero activity,
  never as stale scaffold.
- The recorder's JSON take re-attaches only events present in the source frames — it never invents them.
  Recorded `sensory`/`dopamine`/`controller` arrays contain **only measured lengths**
  (sliced to the replay coverage, or full buffers for the scaffold), and the take carries
  `meta.populations` = measured widths plus `meta.coverage` / `meta.provenance` and
  `source.coverage` / `source.provenance` for reload-time verification.

## Controls

| control | what it does |
|---|---|
| Play / Pause (`Space`) | start/stop snapshot playback (rendering/orbit always live) |
| Replay (`R`) | restart from `t=0`, rebuild event log deterministically |
| Reset view (`V`) | fly back to dorsal default |
| Presets `1–4` | Dorsal / Frontal / Lateral / VNC close-up camera tweens |
| Cinematic (`C`) | auto-orbit + letterbox + slow preset tour (respects `prefers-reduced-motion`) |
| Speed | 0.25×–4× playback rate |
| Scrub slider | seek; event log rebuilds deterministically (no phantom events) |
| Load file / Load URL | parse + validate replay JSON; errors shown as toast, scaffold retained |
| Record demo | starts `canvas.captureStream(60)` + `MediaRecorder` **and** ~4 Hz deterministic state capture |
| Stop | stops both; enables Video + JSON downloads |
| Download Video | saves the take as `.webm` (browser-native) |
| Download JSON | saves the collected states as a reloadable `fly-cns-replay/1` file |
| MP4 hook (optional) | POSTs the WebM take to your endpoint, downloads the MP4 response; WebM remains the fallback |

## Video / MP4 notes

- Recording uses `canvas.captureStream(60)` + `MediaRecorder`, picking the first supported
  mime in `video/webm;codecs=vp9` → `vp8` → `video/webm`. `preserveDrawingBuffer: true` is set
  so captures are reliable.
- For MP4: enter an endpoint URL (or set `window.__MP4_CONVERT_URL__` before `app.js` loads).
  The viewer `POST`s multipart `file` (WebM) + `config` JSON and downloads whatever blob the
  hook returns as `fly-cns-demo.mp4`. Any hook failure keeps the WebM take intact.
- Recorded JSON takes store activity rounded to 3 decimals at ~4 Hz so they stay small and
  reloadable via Load file / `?replay=`.

## Accessibility & states

- All controls are native buttons/inputs with labels, visible focus, and keyboard shortcuts
  (`Space`, `R`, `V`, `C`, `1–4`).
- `aria-live` status line + event log; loading overlay with spinner; errors via `role="alert"` toast.
- Layout collapses gracefully under 1080 px / 760 px (side event panels hide on small screens).

## Files

```text
viewer/
  index.html          # structure + overlays + controls
  styles.css          # dark cinematic theme
  app.js              # Three.js scene, playback, capture (type="module")
  sample-replay.json  # small labeled sample replay (see schema above)
  README.md           # this file
```

Python backend is untouched — this viewer is fully standalone.
