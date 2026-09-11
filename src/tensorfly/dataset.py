"""Checksum-locked MaleCNS v1.0 acquisition and preparation.

The release files are the official Janelia Google Cloud objects used by
``mattyhempstead/fly-wirehead``.  This module deliberately has no synthetic
fallback: a synthetic fixture is available only when ``synthetic_dev=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

RELEASE = "MaleCNS v1.0"
EXPECTED_RETAINED_NEURONS = 166_700
EXPECTED_RETAINED_EDGES = 25_582_938
SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "annotations.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 14_483_314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "neurotransmitters.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
        "bytes": 43_282_834,
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
    "edges.feather": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 1_051_241_946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}

# These are documented by the Janelia MaleCNS download page.  Skeleton files
# are object-addressed by biological body ID; this URI is recorded in every
# manifest so a viewer can never mistake generated coordinates for anatomy.
SKELETON_SWCS_URI = (
    "gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/"
)
SKELETON_SWCS_HTTPS = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-swc/"
)


class DatasetPreparationError(RuntimeError):
    """Raised for missing, corrupt, or schema-incompatible release data."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: str | Path, spec: Mapping[str, Any]) -> None:
    path = Path(path)
    size = path.stat().st_size
    if size != int(spec["bytes"]):
        raise DatasetPreparationError(
            f"{path.name}: byte size {size} != pinned {spec['bytes']}"
        )
    actual = sha256_file(path)
    if actual != str(spec["sha256"]):
        raise DatasetPreparationError(
            f"{path.name}: SHA-256 {actual} != pinned {spec['sha256']}"
        )


def _download_verified(path: Path, spec: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with urllib.request.urlopen(str(spec["url"])) as response, temporary.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        verify_source(temporary, spec)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def exact_ids(values: Any) -> np.ndarray:
    """Convert IDs without ever accepting a lossy floating-point value."""
    array = np.asarray(values)
    if array.dtype.kind == "f":
        raise ValueError("MaleCNS body IDs must be integers or decimal strings, not floats")
    if array.dtype.kind in "iu":
        if np.any(array < 0):
            raise ValueError("MaleCNS body IDs cannot be negative")
        return array.astype(np.uint64, copy=False)
    out: list[int] = []
    for value in array.tolist():
        text = str(value)
        if not text.isascii() or not text.isdecimal():
            raise ValueError(f"Invalid biological body ID: {value!r}")
        out.append(int(text))
    return np.asarray(out, dtype=np.uint64)


def _exact_counts(values: Any) -> np.ndarray:
    """Parse contact counts from Arrow/CSV without silently truncating data."""
    array = np.asarray(values)
    if array.dtype.kind in "iu":
        result = array.astype(np.uint64, copy=False)
    elif array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
            raise DatasetPreparationError("MaleCNS synapse counts must be finite integers")
        result = array.astype(np.uint64)
    else:
        parsed: list[int] = []
        for value in array.tolist():
            text = str(value).strip()
            if not text.isascii() or not text.isdecimal():
                raise DatasetPreparationError(f"Invalid synapse contact count: {value!r}")
            parsed.append(int(text))
        result = np.asarray(parsed, dtype=np.uint64)
    if np.any(result < 1) or np.any(result > np.iinfo(np.uint32).max):
        raise DatasetPreparationError("MaleCNS synapse counts must be positive uint32 values")
    return result.astype(np.uint32)


def _read_table(path: Path) -> Any:
    try:
        import pyarrow.feather as feather  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]

        if path.suffix.lower() in {".feather", ".arrow"}:
            return feather.read_table(path)
        if path.suffix.lower() == ".parquet":
            return parquet.read_table(path)
        import pyarrow.csv as csv  # type: ignore[import-not-found]

        return csv.read_csv(path)
    except ImportError:
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DatasetPreparationError("Reading MaleCNS files requires pyarrow or pandas") from exc
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_feather(path) if path.suffix.lower() == ".feather" else pd.read_csv(path)


def _column_names(table: Any) -> list[str]:
    if hasattr(table, "column_names"):
        return [str(c) for c in table.column_names]
    return [str(c) for c in table.columns]


def _column(table: Any, name: str) -> list[Any]:
    if hasattr(table, "column"):
        return table.column(name).to_pylist()
    return table[name].tolist()


def _find_column(columns: Sequence[str], *names: str) -> str:
    lower = {name.lower(): name for name in columns}
    for candidate in names:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise DatasetPreparationError(f"Required MaleCNS column missing; expected one of {names}")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _records(table: Any) -> list[dict[str, Any]]:
    names = _column_names(table)
    cols = {name: _column(table, name) for name in names}
    return [
        {name: _json_value(cols[name][i]) for name in names}
        for i in range(len(next(iter(cols.values()), [])))
    ]


def _edge_batches(path: Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield edge batches; Arrow IPC avoids loading the 1.1GB file at once."""
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.ipc as ipc  # type: ignore[import-not-found]

        reader = ipc.open_file(pa.memory_map(str(path), "r"))
        columns = [str(c) for c in reader.schema.names]
        pre_name = _find_column(columns, "body_pre", "pre", "source", "pre_id")
        post_name = _find_column(columns, "body_post", "post", "target", "post_id")
        weight_name = _find_column(columns, "weight", "synapse_count", "count")
        for number in range(reader.num_record_batches):
            batch = reader.get_batch(number)
            arrays = {
                name: batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                for name in (pre_name, post_name, weight_name)
            }
            yield exact_ids(arrays[pre_name]), exact_ids(arrays[post_name]), _exact_counts(arrays[weight_name])
        return
    except (ImportError, ValueError, OSError):
        table = _read_table(path)
        names = _column_names(table)
        pre_name = _find_column(names, "body_pre", "pre", "source", "pre_id")
        post_name = _find_column(names, "body_post", "post", "target", "post_id")
        weight_name = _find_column(names, "weight", "synapse_count", "count")
        yield exact_ids(_column(table, pre_name)), exact_ids(_column(table, post_name)), _exact_counts(_column(table, weight_name))


@dataclass
class MaleCNSDataset:
    """Prepared graph and annotation catalog with biological-ID mappings."""

    body_ids: np.ndarray
    pre_index: np.ndarray
    post_index: np.ndarray
    synapse_count: np.ndarray
    annotations: list[dict[str, Any]]
    transmitters: list[dict[str, Any]]
    report: dict[str, Any]
    cache_dir: Path
    skeleton_manifest: dict[str, Any] | None = None

    @property
    def num_neurons(self) -> int:
        return int(self.body_ids.size)

    @property
    def num_edges(self) -> int:
        return int(self.pre_index.size)

    @property
    def synaptic_contacts(self) -> int:
        return int(self.synapse_count.astype(np.uint64).sum(dtype=np.uint64))

    def index_for_body_id(self, body_id: int | str) -> int:
        value = exact_ids([body_id])[0]
        index = int(np.searchsorted(self.body_ids, value))
        if index >= self.body_ids.size or self.body_ids[index] != value:
            raise KeyError(f"Unknown MaleCNS body ID: {body_id}")
        return index

    def body_id_for_index(self, internal_index: int) -> int:
        if internal_index < 0 or internal_index >= self.body_ids.size:
            raise IndexError(internal_index)
        return int(self.body_ids[internal_index])


def _cache_load(root: Path, source_hashes: Mapping[str, Any]) -> MaleCNSDataset | None:
    meta_path = root / "prepared.json"
    arrays_path = root / "graph.npz"
    catalog_path = root / "catalog.json"
    if not (meta_path.exists() and arrays_path.exists() and catalog_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_hashes") != source_hashes:
        return None
    with np.load(arrays_path, allow_pickle=False) as arrays:
        body_ids = arrays["body_ids"].astype(np.uint64, copy=False)
        pre = arrays["pre_index"].astype(np.uint32, copy=False)
        post = arrays["post_index"].astype(np.uint32, copy=False)
        counts = arrays["synapse_count"].astype(np.uint32, copy=False)
    if body_ids.size != EXPECTED_RETAINED_NEURONS or pre.size != EXPECTED_RETAINED_EDGES:
        raise DatasetPreparationError("Cached derived arrays do not satisfy pinned MaleCNS v1.0 accounting")
    if not np.all(body_ids[:-1] < body_ids[1:]) or pre.size != post.size or pre.size != counts.size:
        raise DatasetPreparationError("Cached MaleCNS graph mapping is malformed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return MaleCNSDataset(body_ids, pre, post, counts, catalog["annotations"], catalog["transmitters"], meta, root)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_arrays_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _normalize_annotations(raw: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not raw:
        raise DatasetPreparationError("MaleCNS annotation table is empty")
    names = list(raw[0])
    id_name = _find_column(names, "bodyId", "body_id", "body", "bodyid")
    superclass_name = _find_column(names, "superclass", "class", "super_class")
    ids = exact_ids([row[id_name] for row in raw])
    if np.unique(ids).size != ids.size:
        raise DatasetPreparationError("Duplicate body IDs in MaleCNS annotations")
    # fly-wirehead-compatible policy: an assigned superclass, excluding the
    # release's explicit Glia superclass.  Do not infer biology from row order.
    keep = np.asarray(
        [bool(str(row.get(superclass_name) or "").strip()) and str(row.get(superclass_name) or "").strip().lower() != "glia" for row in raw],
        dtype=bool,
    )
    retained = [dict(row, body_id=int(ids[i]), retained=bool(keep[i])) for i, row in enumerate(raw) if keep[i]]
    retained.sort(key=lambda row: int(row["body_id"]))
    return np.sort(ids[keep]), retained


def prepare_malecns(
    cache_dir: str | Path | None = None,
    *,
    synthetic_dev: bool = False,
) -> MaleCNSDataset:
    """Download, verify, preprocess, and cache the official MaleCNS graph.

    ``synthetic_dev`` is intentionally explicit and produces a tiny schema-
    compatible fixture for unit tests.  No download failure can enter it.
    """
    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "tensorfly" / "malecns-v1.0"
    root.mkdir(parents=True, exist_ok=True)
    if synthetic_dev:
        body_ids = np.asarray([11, 23, 47, 89], dtype=np.uint64)
        pre = np.asarray([0, 1, 2, 3], dtype=np.uint32)
        post = np.asarray([1, 2, 3, 0], dtype=np.uint32)
        counts = np.asarray([2, 1, 3, 1], dtype=np.uint32)
        report = {"release": RELEASE, "synthetic_dev": True, "source_hashes": {}, "source_annotation_rows": 4, "retained_neurons": 4, "source_edge_rows": 4, "retained_edge_rows": 4, "excluded_edges": 0, "synaptic_contact_count": 7, "isolated_neurons": 0, "retention_policy": "explicit development fixture"}
        return MaleCNSDataset(body_ids, pre, post, counts, [{"body_id": int(i), "superclass": "synthetic-dev", "type": "fixture", "retained": True} for i in body_ids], [{"body": int(i), "consensus_nt": "unknown"} for i in body_ids], report, root)

    source_paths: dict[str, Path] = {}
    for name, spec in SOURCE_SPECS.items():
        path = root / name
        if not path.exists():
            _download_verified(path, spec)
        else:
            verify_source(path, spec)
        source_paths[name] = path
    source_hashes = {name: {**spec} for name, spec in SOURCE_SPECS.items()}
    cached = _cache_load(root, source_hashes)
    if cached is not None:
        return cached

    annotation_table = _read_table(source_paths["annotations.feather"])
    annotation_raw = _records(annotation_table)
    body_ids, retained_annotations = _normalize_annotations(annotation_raw)
    annotation_by_id = {int(row["body_id"]): row for row in retained_annotations}
    nt_table = _read_table(source_paths["neurotransmitters.feather"])
    transmitters = _records(nt_table)
    nt_names = list(transmitters[0]) if transmitters else []
    nt_id_name = _find_column(nt_names, "body", "bodyId", "body_id") if nt_names else "body"
    nt_by_id = {int(exact_ids([row[nt_id_name]])[0]): row for row in transmitters}
    for body_id, row in annotation_by_id.items():
        row["neurotransmitter"] = nt_by_id.get(body_id, {}).get("consensus_nt")
        row["neurotransmitter_source"] = "source_consensus_prediction_or_ground_truth" if body_id in nt_by_id else "missing"
    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    source_edges = source_contacts = 0
    index = {int(value): i for i, value in enumerate(body_ids)}
    for pre_ids, post_ids, weights in _edge_batches(source_paths["edges.feather"]):
        if len(pre_ids) != len(post_ids) or len(pre_ids) != len(weights):
            raise DatasetPreparationError("MaleCNS edge columns have different lengths")
        weights = _exact_counts(weights)
        source_edges += len(pre_ids)
        source_contacts += int(np.asarray(weights, dtype=np.uint64).sum(dtype=np.uint64))
        i = np.searchsorted(body_ids, pre_ids)
        j = np.searchsorted(body_ids, post_ids)
        keep = (i < body_ids.size) & (j < body_ids.size)
        keep &= body_ids[np.minimum(i, body_ids.size - 1)] == pre_ids
        keep &= body_ids[np.minimum(j, body_ids.size - 1)] == post_ids
        pre_parts.append(i[keep].astype(np.uint32))
        post_parts.append(j[keep].astype(np.uint32))
        count_parts.append(weights[keep])
    pre = np.concatenate(pre_parts) if pre_parts else np.empty(0, dtype=np.uint32)
    post = np.concatenate(post_parts) if post_parts else np.empty(0, dtype=np.uint32)
    counts = np.concatenate(count_parts) if count_parts else np.empty(0, dtype=np.uint32)
    # This source lock intentionally adopts fly-wirehead's published policy;
    # a count mismatch means schema/policy/source drift, not a graph we may
    # casually label MaleCNS v1.0.
    if body_ids.size != EXPECTED_RETAINED_NEURONS or pre.size != EXPECTED_RETAINED_EDGES:
        raise DatasetPreparationError(
            "Pinned MaleCNS v1.0 retention accounting mismatch: "
            f"got {body_ids.size} neurons / {pre.size} edges, expected "
            f"{EXPECTED_RETAINED_NEURONS} / {EXPECTED_RETAINED_EDGES}"
        )
    incoming = np.bincount(post, weights=counts.astype(np.float64), minlength=body_ids.size)
    outgoing = np.bincount(pre, weights=counts.astype(np.float64), minlength=body_ids.size)
    report = {
        "dataset_id": "malecns_v1",
        "release": RELEASE,
        "source_hashes": source_hashes,
        "source_annotation_rows": len(annotation_raw),
        "retained_neurons": int(body_ids.size),
        "source_edge_rows": source_edges,
        "retained_edge_rows": int(pre.size),
        "excluded_edges": source_edges - int(pre.size),
        "synaptic_contact_count": int(counts.astype(np.uint64).sum(dtype=np.uint64)),
        "source_synaptic_contact_count": source_contacts,
        "excluded_synaptic_contact_count": source_contacts - int(counts.astype(np.uint64).sum(dtype=np.uint64)),
        "isolated_neurons": int(np.count_nonzero((incoming == 0) & (outgoing == 0))),
        "retention_policy": "assigned superclass and non-Glia status; all released edges between retained entries; no weight threshold; autapses retained",
        "synaptic_weights_are_contact_counts": True,
        "physiology_note": "Wiring and transmitter predictions are source data; membrane dynamics and transmitter effects are engineered approximations.",
    }
    _write_arrays_atomic(root / "graph.npz", body_ids=body_ids, pre_index=pre, post_index=post, synapse_count=counts)
    _write_json_atomic(root / "catalog.json", {"annotations": list(annotation_by_id.values()), "transmitters": transmitters})
    _write_json_atomic(root / "prepared.json", report)
    return MaleCNSDataset(body_ids, pre, post, counts, list(annotation_by_id.values()), transmitters, report, root)


def _parse_swc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    node_ids: list[int] = []
    parent_ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise DatasetPreparationError(f"Malformed SWC row in {path}")
        node_ids.append(int(fields[0]))
        coords.append((float(fields[2]), float(fields[3]), float(fields[4])))
        parent_ids.append(int(fields[6]))
    if not coords:
        raise DatasetPreparationError(f"Empty SWC skeleton: {path}")
    local = {node_id: index for index, node_id in enumerate(node_ids)}
    parents = np.asarray([local.get(parent, -1) for parent in parent_ids], dtype=np.int32)
    return np.asarray(coords, dtype=np.float32), parents


def prepare_skeleton_manifest(
    selected_body_ids: Iterable[int | str],
    skeleton_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pack real SWC skeleton coordinates into an efficient typed-array cache."""
    root = Path(skeleton_dir)
    ids = np.sort(exact_ids(list(selected_body_ids)))
    chunks: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    offsets = [0]
    for body_id in ids:
        path = root / f"{int(body_id)}.swc"
        if not path.exists():
            raise DatasetPreparationError(f"Missing official skeleton for body ID {int(body_id)}: {path}")
        coords, parent = _parse_swc(path)
        chunks.append(coords)
        parents.append(parent)
        offsets.append(offsets[-1] + len(coords))
        entries.append({"body_id": str(int(body_id)), "path": path.name, "sha256": sha256_file(path), "coordinate_space": "MaleCNS EM, 8nm"})
    output = Path(output_path) if output_path is not None else root / "selected-skeletons.npz"
    _write_arrays_atomic(output, body_ids=ids, offsets=np.asarray(offsets, dtype=np.uint32), coordinates=np.concatenate(chunks), parent=np.concatenate(parents))
    manifest = {"release": RELEASE, "source": SKELETON_SWCS_URI, "body_ids": [str(int(i)) for i in ids], "entries": entries, "geometry_is_source_derived": True, "procedural_geometry": False, "binary_path": str(output)}
    _write_json_atomic(output.with_suffix(".json"), manifest)
    return manifest


def download_selected_skeletons(selected_body_ids: Iterable[int | str], cache_dir: str | Path) -> Path:
    """Fetch selected official SWCs and verify cached objects against manifest.

    Janelia publishes a directory of body-ID-addressed SWCs rather than an
    aggregate archive/hash.  Each selected immutable object URL, byte count
    and SHA-256 is therefore captured and rechecked locally.  No generated
    coordinates can enter this path.
    """
    root = Path(cache_dir) / "skeletons-swc"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download-manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"entries": {}}
    entries: dict[str, Any] = dict(old.get("entries", {}))
    for body_id in np.sort(exact_ids(list(selected_body_ids))):
        key = str(int(body_id)); path = root / f"{key}.swc"; url = f"{SKELETON_SWCS_HTTPS}{key}.swc"
        prior = entries.get(key)
        if path.exists() and prior and prior.get("sha256") == sha256_file(path) and prior.get("bytes") == path.stat().st_size:
            continue
        fd, name = tempfile.mkstemp(prefix=f".{key}.", suffix=".partial", dir=root); os.close(fd)
        temporary = Path(name)
        try:
            with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024): out.write(chunk)
            temporary.replace(path)
            entries[key] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path), "body_id": key}
        finally:
            temporary.unlink(missing_ok=True)
    _write_json_atomic(manifest_path, {"release": RELEASE, "source_directory": SKELETON_SWCS_URI, "entries": entries})
    return root


def export_viewer_morphology(dataset: MaleCNSDataset, selected_body_ids: Iterable[int | str], skeleton_dir: str | Path, output_path: str | Path, population_by_id: Mapping[str, str] | None = None) -> Path:
    """Emit browser-efficient real coordinates and real-edge context JSON."""
    manifest = prepare_skeleton_manifest(selected_body_ids, skeleton_dir, Path(skeleton_dir) / "selected-skeletons.npz")
    with np.load(manifest["binary_path"], allow_pickle=False) as arrays:
        ids, offsets, coordinates, parents = arrays["body_ids"], arrays["offsets"], arrays["coordinates"], arrays["parent"]
    skeletons: list[dict[str, Any]] = []
    points: dict[int, list[float]] = {}
    for index, body_id in enumerate(ids):
        start, end = int(offsets[index]), int(offsets[index + 1]); coords = coordinates[start:end]
        segments: list[int] = []
        for local, parent in enumerate(parents[start:end]):
            if parent >= 0 and parent < len(coords): segments.extend((local, int(parent)))
        bid = int(body_id); points[bid] = coords[0].astype(float).tolist()
        skeletons.append({"body_id": str(bid), "positions": coords.astype(float).reshape(-1).tolist(), "segments": segments, "population": (population_by_id or {}).get(str(bid), "context")})
    selected = set(points); edge_rows: list[dict[str, Any]] = []
    # Deterministic capped context lines correspond to actual retained edges.
    for pre, post in zip(dataset.pre_index, dataset.post_index):
        a, b = int(dataset.body_ids[pre]), int(dataset.body_ids[post])
        if a in selected and b in selected:
            edge_rows.append({"pre_body_id": str(a), "post_body_id": str(b), "positions": points[a] + points[b]})
            if len(edge_rows) >= 2_000: break
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, {"schema": "tensorfly-real-morphology/1", "provenance": {"release": RELEASE, "is_synthetic": bool(dataset.report.get("synthetic_dev", False)), "retention_policy": dataset.report["retention_policy"], "skeleton_manifest": manifest}, "skeletons": skeletons, "connectivity_edges": edge_rows})
    return destination


__all__ = ["RELEASE", "EXPECTED_RETAINED_NEURONS", "EXPECTED_RETAINED_EDGES", "SOURCE_SPECS", "SKELETON_SWCS_URI", "SKELETON_SWCS_HTTPS", "DatasetPreparationError", "MaleCNSDataset", "exact_ids", "sha256_file", "verify_source", "prepare_malecns", "prepare_skeleton_manifest", "download_selected_skeletons", "export_viewer_morphology"]
