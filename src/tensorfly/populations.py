"""Deterministic, annotation-backed MaleCNS population registries.

Population membership is a scientific selection policy, not a positional slice
of the graph.  The selected biological IDs and criteria are serializable so a
run can be audited or replayed exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dataset import MaleCNSDataset, exact_ids

DEFAULT_SENSORY_TYPES = frozenset({"R1-R6", "R7", "R8"})
DEFAULT_CONTROLLER_TYPES = frozenset({"DNa02", "DNp09", "MDN", "MN9", "DNp20", "DNpe017"})


def _text(row: Mapping[str, Any], *names: str) -> str:
    keys = {str(key).lower(): key for key in row}
    for name in names:
        key = keys.get(name.lower())
        if key is not None and row[key] is not None:
            return str(row[key]).strip()
    return ""


@dataclass(frozen=True)
class PopulationCriteria:
    """Explicit predicates used to select each neural population."""

    sensory_types: tuple[str, ...] = tuple(sorted(DEFAULT_SENSORY_TYPES))
    sensory_superclass_terms: tuple[str, ...] = ("sensory", "optic lobe")
    dopamine_transmitters: tuple[str, ...] = ("dopamine", "da")
    dopamine_type_prefixes: tuple[str, ...] = ("PAM", "PPL")
    controller_types: tuple[str, ...] = tuple(sorted(DEFAULT_CONTROLLER_TYPES))
    controller_superclass_terms: tuple[str, ...] = ("descending", "motor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensory_types": list(self.sensory_types),
            "sensory_superclass_terms": list(self.sensory_superclass_terms),
            "dopamine_transmitters": list(self.dopamine_transmitters),
            "dopamine_type_prefixes": list(self.dopamine_type_prefixes),
            "controller_types": list(self.controller_types),
            "controller_superclass_terms": list(self.controller_superclass_terms),
        }


@dataclass
class PopulationRegistry:
    """Selected body IDs plus compact graph indices and auditable criteria."""

    sensory_body_ids: np.ndarray
    dopamine_body_ids: np.ndarray
    controller_body_ids: np.ndarray
    sensory_indices: np.ndarray
    dopamine_indices: np.ndarray
    controller_indices: np.ndarray
    criteria: PopulationCriteria
    annotation_criteria: dict[str, str]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def body_ids(self) -> dict[str, np.ndarray]:
        return {"sensory": self.sensory_body_ids, "dopamine": self.dopamine_body_ids, "controller": self.controller_body_ids}

    @property
    def indices(self) -> dict[str, np.ndarray]:
        return {"sensory": self.sensory_indices, "dopamine": self.dopamine_indices, "controller": self.controller_indices}

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_ids": {key: [str(int(value)) for value in values] for key, values in self.body_ids.items()},
            "internal_indices": {key: [int(value) for value in values] for key, values in self.indices.items()},
            "criteria": self.criteria.to_dict(),
            "annotation_criteria": dict(self.annotation_criteria),
            "provenance": dict(self.provenance),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination


def _matches(row: Mapping[str, Any], criteria: PopulationCriteria) -> tuple[bool, bool, bool]:
    cell_type = _text(row, "type", "cell_type", "cellType").lower()
    superclass = _text(row, "superclass", "class", "super_class").lower()
    transmitter = _text(row, "neurotransmitter", "consensus_nt", "nt").lower()
    sensory = cell_type.upper() in {value.upper() for value in criteria.sensory_types} or any(term.lower() in superclass for term in criteria.sensory_superclass_terms)
    dopamine = transmitter in {value.lower() for value in criteria.dopamine_transmitters} or any(cell_type.upper().startswith(prefix.upper()) for prefix in criteria.dopamine_type_prefixes)
    controller = cell_type.upper() in {value.upper() for value in criteria.controller_types} or any(term.lower() in superclass for term in criteria.controller_superclass_terms)
    return sensory, dopamine, controller


def build_population_registry(
    dataset: MaleCNSDataset | Sequence[Mapping[str, Any]],
    *,
    criteria: PopulationCriteria | None = None,
    persist_path: str | Path | None = None,
    require_nonempty: bool = True,
) -> PopulationRegistry:
    """Resolve populations from real annotations and body IDs.

    ``dataset`` should normally be the object returned by
    :func:`tensorfly.dataset.prepare_malecns`; a sequence is accepted for
    lightweight schema fixtures and must contain ``body_id``/annotation fields.
    """
    policy = criteria or PopulationCriteria()
    if isinstance(dataset, MaleCNSDataset):
        rows = dataset.annotations
        body_ids = dataset.body_ids
        release = dataset.report.get("release", "unknown")
        source_hashes = dataset.report.get("source_hashes", {})
        synthetic = bool(dataset.report.get("synthetic_dev", False))
    else:
        rows = [dict(row) for row in dataset]
        if not rows:
            raise ValueError("Population annotation fixture is empty")
        body_ids = np.sort(exact_ids([row.get("body_id", row.get("bodyId")) for row in rows]))
        release = "fixture"
        source_hashes = {}
        synthetic = True
    index = {int(value): i for i, value in enumerate(body_ids)}
    selected: dict[str, list[int]] = {"sensory": [], "dopamine": [], "controller": []}
    for row in rows:
        raw = row.get("body_id", row.get("bodyId"))
        if raw is None:
            continue
        body_id = int(exact_ids([raw])[0])
        if body_id not in index:
            continue
        sensory, dopamine, controller = _matches(row, policy)
        if sensory:
            selected["sensory"].append(body_id)
        if dopamine:
            selected["dopamine"].append(body_id)
        if controller:
            selected["controller"].append(body_id)
    for key in selected:
        selected[key] = sorted(set(selected[key]))
    if require_nonempty:
        missing = [key for key, values in selected.items() if not values]
        if missing:
            raise ValueError(f"No real annotated neurons matched population(s): {', '.join(missing)}")
    ids = {key: np.asarray(values, dtype=np.uint64) for key, values in selected.items()}
    indices = {key: np.asarray([index[int(value)] for value in values], dtype=np.uint32) for key, values in ids.items()}
    registry = PopulationRegistry(
        ids["sensory"], ids["dopamine"], ids["controller"],
        indices["sensory"], indices["dopamine"], indices["controller"],
        policy,
        {
            "sensory": "cell type in R1-R6/R7/R8 OR annotation superclass contains sensory/optic lobe",
            "dopamine": "source consensus transmitter is dopamine/DA OR annotated cell type prefix PAM/PPL",
            "controller": "annotated cell type in DNa02/DNp09/MDN/MN9/DNp20/DNpe017 OR superclass contains descending/motor",
        },
        {"release": release, "source_hashes": source_hashes, "selection_is_deterministic": True, "synthetic": synthetic},
    )
    destination = persist_path
    if destination is None and isinstance(dataset, MaleCNSDataset):
        # Population membership is part of dataset provenance, not ephemeral
        # runtime state.  Keep the exact IDs/criteria beside the derived graph.
        destination = dataset.cache_dir / "populations.json"
    if destination is not None:
        registry.save(destination)
    return registry


__all__ = ["PopulationCriteria", "PopulationRegistry", "build_population_registry", "DEFAULT_SENSORY_TYPES", "DEFAULT_CONTROLLER_TYPES"]
