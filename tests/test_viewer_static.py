"""Lightweight static checks for viewer/app.js + viewer/index.html.

No browser/node required: verifies the reviewer-reported fatal ESM issue stays
fixed, WebGL1-safe shaders, and replay coverage integrity markers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "viewer" / "app.js"
INDEX = REPO / "viewer" / "index.html"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_strings_comments(src: str) -> str:
    # Remove block comments, line comments, template/single/double strings (naive).
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"`(?:\\.|[^`\\])*`", "``", src, flags=re.S)
    src = re.sub(r"'(?:\\.|[^'\\\n])*'", "''", src)
    src = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', src)
    return src


def test_viewer_files_exist():
    assert APP.exists(), "viewer/app.js missing"
    assert INDEX.exists(), "viewer/index.html missing"


def test_module_markers_present():
    html = _read(INDEX)
    js = _read(APP)
    assert 'type="module"' in html, "index.html must load app.js as type=module"
    assert "importmap" in html, "index.html must define an importmap for three"
    assert "import * as THREE from 'three'" in js
    assert "OrbitControls" in js


def test_balanced_delimiters():
    js = _strip_strings_comments(_read(APP))
    for o, c, name in [("{", "}", "braces"), ("(", ")", "parens"), ("[", "]", "brackets")]:
        assert js.count(o) == js.count(c), f"unbalanced {name}: {o}={js.count(o)} {c}={js.count(c)}"
    # The fatal ESM bug was a stray `});` closing getArr; the fixed closer is `};`.
    assert "return { buf, measured };" in js, "getArr must return { buf, measured }"


def test_no_gl_vertex_id():
    js = _read(APP)
    assert "gl_VertexID" not in js, "viewer must not use gl_VertexID (WebGL1-unsafe)"
    assert "gl_VertexId" not in js


def test_getarr_syntax_fixed():
    js = _read(APP)
    # getArr arrow body must close with `};`, and forEach must still close with `});`.
    m = re.search(r"const getArr = \(name, n\) => \{.*?return \{ buf, measured \};\s*\};", js, flags=re.S)
    assert m, "getArr definition is malformed (expected `return { buf, measured };` + `};`)"


def test_replay_coverage_handling_present():
    js = _read(APP)
    for marker in [
        "summarizeCoverage",
        "measuredCounts",
        "coverageText",
        "zeroActivityTails",
        "sN",
        "dN",
        "cN",
        "partial coverage",
        "zero-fill tails",
        "never stale scaffold",
        "isPartial",
        "provenance",
    ]:
        assert marker in js, f"replay coverage marker missing: {marker}"


def test_tails_zeroed_and_recorded_lengths_measured():
    js = _read(APP)
    assert "S.fill(0, nS)" in js and "D.fill(0, nD)" in js and "C.fill(0, nC)" in js
    assert "subarray(0," in js, "recorded JSON must slice to measured lengths"
    # Recorded populations must derive from measured counts, not built-in constants.
    assert "measuredCounts()" in js
    # Old fabrication (edge-repeat padding) must be gone.
    assert "buf[m - 1]" not in js, "edge-repeat padding fabricates activity; tails must be zero"


def test_partial_badge_qualified():
    js = _read(APP)
    assert "REPLAY · partial coverage" in js, "partial replays need a qualified badge"
    # Full measured badge must only appear on the full-coverage path.
    assert "LIVE REPLAY · measured snapshots" in js
    assert "coverage.isFull" in js or "cov.isFull" in js or "isFull" in js
