# TensorFly real MaleCNS viewer

Run `tensorfly.prepare()` first. It writes `real-morphology.json` from a
deterministic subset of downloaded official MaleCNS SWC skeletons, retaining
the original body ID as a decimal string. It also captures every downloaded
object URL, byte size and SHA-256 in the skeleton manifest.

Then run an experiment and call `experiment.export_video()` to write
`tensorfly_replay.json`. Serve this directory over HTTP:

```bash
python -m http.server 8000 --directory viewer
```

The viewer uses Three.js `BufferGeometry` / typed arrays and real SWC line
segments only. It rejects absent, malformed or synthetic morphology/replay
provenance instead of inventing fallback point clouds, shells, or links.
Activity values are keyed by real body-ID strings; optional visual connection
lines are a deterministic cap of actual retained graph edges.
