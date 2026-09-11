import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ============================================================
   MaleCNS viewer — batched point-clouds, snapshot-driven.
   No per-neuron JS objects: all state lives in typed buffers.
   ============================================================ */

const $ = (id) => document.getElementById(id);
const els = {
  canvas: $('scene'), viewport: $('viewport'),
  loading: $('loading'), loadingText: $('loadingText'),
  toast: $('toast'), live: $('live'), status: $('statusLine'),
  badge: $('sourceBadge'), recBadge: $('recBadge'), recTime: $('recTime'),
  fps: $('fpsBadge'), modelLine: $('modelLine'),
  hTrial: $('hTrial'), hTokens: $('hTokens'), hTime: $('hTime'),
  hTps: $('hTps'), hTtft: $('hTtft'), hDecode: $('hDecode'), hReward: $('hReward'),
  hConfig: $('hConfig'), hPrompt: $('hPrompt'), hBest: $('hBest'),
  progress: $('playProgress'), scrub: $('scrub'), spark: $('spark'),
  log: $('eventLog'), banner: $('eventBanner'),
  dModel: $('dModel'), dSource: $('dSource'), dFrames: $('dFrames'),
  dDur: $('dDur'), dNeurons: $('dNeurons'), dConfig: $('dConfig'),
  btnPlay: $('btnPlay'), btnReplay: $('btnReplay'), btnReset: $('btnResetView'),
  btnCine: $('btnCinematic'), speed: $('speed'),
  file: $('fileInput'), url: $('urlInput'), btnUrl: $('btnLoadUrl'), btnSample: $('btnSample'),
  btnRecord: $('btnRecord'), btnStop: $('btnStop'), btnVideo: $('btnVideo'), btnJSON: $('btnJSON'),
  mp4Url: $('mp4Url'), btnMp4: $('btnMp4'),
};

const POP = {
  sensory:   { count: 3800, color: new THREE.Color('#22d3ee'), size: 2.1 },
  dopamine:  { count: 550,  color: new THREE.Color('#fbbf24'), size: 4.2 },
  controller:{ count: 1300, color: new THREE.Color('#e879f9'), size: 2.8 },
};
const TOTAL = POP.sensory.count + POP.dopamine.count + POP.controller.count;

/* ---------- deterministic RNG ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- three setup ---------- */
const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x05070d, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070d, 0.0085);

const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 600);
const DEFAULT_CAM = { pos: new THREE.Vector3(30, 20, 42), tgt: new THREE.Vector3(0, -2, 0) };
camera.position.copy(DEFAULT_CAM.pos);

const controls = new OrbitControls(camera, els.canvas);
controls.target.copy(DEFAULT_CAM.tgt);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 8;
controls.maxDistance = 180;
controls.autoRotateSpeed = 0.9;

scene.add(new THREE.AmbientLight(0x334155, 0.7));
const key = new THREE.DirectionalLight(0x88ccff, 1.1); key.position.set(20, 30, 25); scene.add(key);
const rim = new THREE.DirectionalLight(0xe879f9, 0.5); rim.position.set(-25, -10, -20); scene.add(rim);

/* brain shell wireframes (static decoration, not data) */
{
  const shellMat = new THREE.MeshBasicMaterial({ color: 0x1e3a5f, wireframe: true, transparent: true, opacity: 0.16 });
  const mk = (rx, ry, rz, x, y, z) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 14), shellMat);
    m.scale.set(rx, ry, rz); m.position.set(x, y, z); scene.add(m);
  };
  mk(7.5, 9.5, 8.5, -13.5, 4, 0);
  mk(7.5, 9.5, 8.5, 13.5, 4, 0);
  mk(8.5, 6.5, 7.5, 0, 2, -1);
  const grid = new THREE.GridHelper(120, 40, 0x164e63, 0x0b1a2e);
  grid.position.y = -26; grid.material.transparent = true; grid.material.opacity = 0.35; scene.add(grid);
}

/* ---------- batched neuron buffers ---------- */
// One BufferGeometry per population (3 draw calls total). No per-neuron objects.
function makePopulation(count, baseColor, baseSize) {
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const act = new Float32Array(count);       // activity in [0,1], updated per frame
  const siz = new Float32Array(count);       // static size jitter
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aActivity', new THREE.BufferAttribute(act, 1));
  geo.setAttribute('aSize', new THREE.BufferAttribute(siz, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uPixelRatio: { value: renderer.getPixelRatio() }, uTime: { value: 0 } },
    vertexShader: `
      attribute vec3 aColor; attribute float aActivity; attribute float aSize;
      uniform float uPixelRatio; uniform float uTime;
      varying vec3 vColor; varying float vAct;
      void main() {
        vAct = aActivity;
        float seed = dot(position, vec3(12.9898, 78.233, 37.719));
        float twinkle = 0.92 + 0.08 * sin(uTime * 2.0 + seed);
        vColor = aColor * (0.22 + 1.9 * aActivity) * twinkle;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float px = aSize * (0.55 + 1.9 * aActivity);
        gl_PointSize = px * uPixelRatio * (160.0 / max(1.0, -mv.z));
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor; varying float vAct;
      void main() {
        vec2 uv = gl_PointCoord - vec2(0.5);
        float d = length(uv);
        if (d > 0.5) discard;
        float core = smoothstep(0.5, 0.05, d);
        float glow = smoothstep(0.5, 0.2, d) * 0.35;
        float a = clamp(core + glow * (0.3 + vAct), 0.0, 1.0) * (0.35 + 0.65 * clamp(vAct + 0.15, 0.0, 1.0));
        gl_FragColor = vec4(vColor, a);
      }`,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  scene.add(points);
  // cache base color into col
  for (let i = 0; i < count; i++) { col[i * 3] = baseColor.r; col[i * 3 + 1] = baseColor.g; col[i * 3 + 2] = baseColor.b; }
  return { geo, points, mat, pos, col, act, siz, count };
}

const pSens = makePopulation(POP.sensory.count, POP.sensory.color, POP.sensory.size);
const pDopa = makePopulation(POP.dopamine.count, POP.dopamine.color, POP.dopamine.size);
const pCtrl = makePopulation(POP.controller.count, POP.controller.color, POP.controller.size);

/* deterministic anatomically-inspired layout into the typed position buffers */
function buildLayout(seed = 1337) {
  const rng = mulberry32(seed);
  const jit = (arr, i, v) => { arr[i] = v; };
  // sensory: two optic lobes + antennal patch
  for (let i = 0; i < pSens.count; i++) {
    const side = i % 2 === 0 ? -1 : 1;
    const lobe = rng() < 0.82;
    let x, y, z;
    if (lobe) {
      x = side * 13.5 + gauss(rng) * 3.4;
      y = 4 + gauss(rng) * 4.4;
      z = gauss(rng) * 4.2;
    } else {
      x = side * 4 + gauss(rng) * 2.2;
      y = -4 + gauss(rng) * 2.0;
      z = 6 + gauss(rng) * 2.0;
    }
    jit(pSens.pos, i * 3, x); jit(pSens.pos, i * 3 + 1, y); jit(pSens.pos, i * 3 + 2, z);
    pSens.siz[i] = POP.sensory.size * (0.7 + rng() * 0.7);
  }
  // dopamine: central mushroom-body-ish cluster (slow broadcast look)
  for (let i = 0; i < pDopa.count; i++) {
    const x = gauss(rng) * 3.6, y = 2.5 + gauss(rng) * 2.6, z = -1.5 + gauss(rng) * 3.0;
    pDopa.pos[i * 3] = x; pDopa.pos[i * 3 + 1] = y; pDopa.pos[i * 3 + 2] = z;
    pDopa.siz[i] = POP.dopamine.size * (0.8 + rng() * 0.6);
  }
  // controller: descending tract + VNC tube
  for (let i = 0; i < pCtrl.count; i++) {
    const f = rng();
    let x, y, z;
    if (f < 0.35) { // neck connective
      const t = rng();
      x = gauss(rng) * 1.6; y = 0 - t * 8; z = -1 + gauss(rng) * 1.6;
    } else { // VNC tube
      y = -8 - rng() * 15;
      const r = 2.6 + rng() * 1.6, a = rng() * Math.PI * 2;
      x = Math.cos(a) * r * 0.7; z = Math.sin(a) * r * 0.7;
    }
    pCtrl.pos[i * 3] = x; pCtrl.pos[i * 3 + 1] = y; pCtrl.pos[i * 3 + 2] = z;
    pCtrl.siz[i] = POP.controller.size * (0.7 + rng() * 0.7);
  }
  for (const p of [pSens, pDopa, pCtrl]) {
    p.geo.attributes.position.needsUpdate = true;
    p.geo.attributes.aSize.needsUpdate = true;
    p.geo.attributes.aColor.needsUpdate = true;
  }
}

/* faint static connectivity (LineSegments, single draw call) */
let linkMat;
{
  const rng = mulberry32(99);
  const N = 700;
  const lp = new Float32Array(N * 6);
  const pick = (p) => {
    const i = Math.floor(rng() * p.count) * 3;
    return [p.pos[i], p.pos[i + 1], p.pos[i + 2]];
  };
  for (let e = 0; e < N; e++) {
    const src = rng();
    const A = src < 0.5 ? pick(pSens) : src < 0.7 ? pick(pDopa) : pick(pCtrl);
    const B = src < 0.5 ? pick(pSens) : src < 0.7 ? pick(pCtrl) : pick(pCtrl);
    lp.set(A, e * 6); lp.set(B, e * 6 + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(lp, 3));
  linkMat = new THREE.LineBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.07, blending: THREE.AdditiveBlending, depthWrite: false });
  const lines = new THREE.LineSegments(g, linkMat);
  lines.frustumCulled = false;
  scene.add(lines);
}

/* ambient dust (static, 1 draw call) */
{
  const rng = mulberry32(7);
  const N = 700, pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos[i * 3] = (rng() - 0.5) * 160;
    pos[i * 3 + 1] = (rng() - 0.5) * 100;
    pos[i * 3 + 2] = (rng() - 0.5) * 160;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.PointsMaterial({ color: 0x3b82f6, size: 0.55, transparent: true, opacity: 0.4, sizeAttenuation: true, depthWrite: false });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  scene.add(pts);
}

buildLayout();

/* ============================================================
   Replay model — snapshot-driven, interpolated
   ============================================================ */
const player = {
  mode: 'sample',          // 'sample' | 'replay'
  meta: null,
  frames: [],              // normalized frames
  duration: 1,
  time: 0,
  playing: false,
  speed: 1,
  lastEventKey: -1,
  sourceLabel: 'sample scaffold',
  coverage: null,          // replay coverage vs built-in buffers (null for full scaffold)
  provenance: 'sample-scaffold',
};

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

/* ---------- replay coverage integrity ---------- */
// Built-in buffers are fixed (POP.*). Loaded replays may declare smaller (or
// ragged) populations. We never fabricate tails: missing entries stay 0, stale
// scaffold values are explicitly cleared, and recorded JSON only carries
// measured lengths. Partial coverage also downgrades the badge (see setSourceBadge).
function summarizeCoverage(frames, exp) {
  const buffered = { sensory: POP.sensory.count, dopamine: POP.dopamine.count, controller: POP.controller.count };
  const expected = {
    sensory: Number(exp?.sensory ?? buffered.sensory) || 0,
    dopamine: Number(exp?.dopamine ?? buffered.dopamine) || 0,
    controller: Number(exp?.controller ?? buffered.controller) || 0,
  };
  const minMeasured = { sensory: Infinity, dopamine: Infinity, controller: Infinity };
  for (const f of frames) {
    const sN = Number(f.sN ?? f.s?.length ?? 0);
    const dN = Number(f.dN ?? f.d?.length ?? 0);
    const cN = Number(f.cN ?? f.c?.length ?? 0);
    if (sN < minMeasured.sensory) minMeasured.sensory = sN;
    if (dN < minMeasured.dopamine) minMeasured.dopamine = dN;
    if (cN < minMeasured.controller) minMeasured.controller = cN;
  }
  if (!frames.length) {
    minMeasured.sensory = 0; minMeasured.dopamine = 0; minMeasured.controller = 0;
  }
  if (!Number.isFinite(minMeasured.sensory)) minMeasured.sensory = 0;
  if (!Number.isFinite(minMeasured.dopamine)) minMeasured.dopamine = 0;
  if (!Number.isFinite(minMeasured.controller)) minMeasured.controller = 0;
  const part = (k) => {
    const measured = minMeasured[k];
    const expN = expected[k];
    const bufN = buffered[k];
    // Full only when every replay frame measured at least the buffered width.
    const full = measured >= bufN && expN >= bufN;
    return { measured, expected: expN, buffered: bufN, full };
  };
  const sensory = part('sensory'), dopamine = part('dopamine'), controller = part('controller');
  const isFull = sensory.full && dopamine.full && controller.full;
  return { sensory, dopamine, controller, isFull, isPartial: !isFull, expected };
}

function coverageText(cov) {
  if (!cov) return '';
  const f = (p) => `${p.measured}/${p.buffered}`;
  return `s:${f(cov.sensory)} d:${f(cov.dopamine)} c:${f(cov.controller)}`;
}

function measuredCounts() {
  const cov = player.coverage;
  if (player.mode !== 'replay' || !cov) return { s: pSens.count, d: pDopa.count, c: pCtrl.count };
  return {
    s: Math.max(0, Math.min(pSens.count, cov.sensory.measured)),
    d: Math.max(0, Math.min(pDopa.count, cov.dopamine.measured)),
    c: Math.max(0, Math.min(pCtrl.count, cov.controller.measured)),
  };
}

// Explicitly clear activity tails beyond measured widths so a smaller replay
// can never display stale scaffold/sample values.
function zeroActivityTails() {
  const m = measuredCounts();
  if (player.mode !== 'replay' || !player.coverage || player.coverage.isFull) return;
  pSens.act.fill(0, m.s);
  pDopa.act.fill(0, m.d);
  pCtrl.act.fill(0, m.c);
  pSens.geo.attributes.aActivity.needsUpdate = true;
  pDopa.geo.attributes.aActivity.needsUpdate = true;
  pCtrl.geo.attributes.aActivity.needsUpdate = true;
}

function normalizeFrames(raw, metaPop) {
  const exp = metaPop || { sensory: POP.sensory.count, dopamine: POP.dopamine.count, controller: POP.controller.count };
  const out = [];
  const sorted = [...raw].sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
  sorted.forEach((f, idx) => {
    const getArr = (name, n) => {
      let a = f[name];
      if (!a && f.activity && name === 'sensory') a = f.activity.slice(0, n);
      if (a && !Array.isArray(a) && ArrayBuffer.isView(a)) a = Array.from(a);
      const buf = new Float32Array(n);
      let measured = 0;
      if (Array.isArray(a)) {
        const m = Math.min(n, a.length);
        measured = m;
        for (let i = 0; i < m; i++) buf[i] = clamp01(Number(a[i]) || 0);
        // zero-fill tails: never fabricate activity beyond measured lengths
        for (let i = m; i < n; i++) buf[i] = 0;
      } else if (name === 'dopamine') {
        // scalar broadcast allowed: { dopamine_mean: x }
        const raw = f.dopamine_mean ?? f.dopamineMean;
        if (raw !== undefined && raw !== null) {
          const v = clamp01(Number(raw) || 0);
          buf.fill(v);
          measured = n;
        } else {
          measured = 0;
        }
      } else {
        measured = 0;
      }
      return { buf, measured };
    };
    // support global activity split s|d|c
    let s, d, c, sN = 0, dN = 0, cN = 0;
    if (!f.sensory && !f.dopamine && !f.controller && Array.isArray(f.activity)) {
      const g = f.activity;
      s = new Float32Array(exp.sensory); d = new Float32Array(exp.dopamine); c = new Float32Array(exp.controller);
      // measured = entries actually present; tails stay zero (never stale/fabricated)
      const avail = g.length;
      let o = 0;
      const mS = Math.max(0, Math.min(exp.sensory, avail - o));
      for (let i = 0; i < mS; i++, o++) s[i] = clamp01(Number(g[o]) || 0);
      sN = mS;
      const mD = Math.max(0, Math.min(exp.dopamine, avail - o));
      for (let i = 0; i < mD; i++, o++) d[i] = clamp01(Number(g[o]) || 0);
      dN = mD;
      const mC = Math.max(0, Math.min(exp.controller, avail - o));
      for (let i = 0; i < mC; i++, o++) c[i] = clamp01(Number(g[o]) || 0);
      cN = mC;
    } else {
      const rs = getArr('sensory', exp.sensory);
      const rd = getArr('dopamine', exp.dopamine);
      const rc = getArr('controller', exp.controller);
      s = rs.buf; sN = rs.measured;
      d = rd.buf; dN = rd.measured;
      c = rc.buf; cN = rc.measured;
    }
    out.push({
      key: idx,
      t: Number(f.t ?? idx * 0.1),
      trial: Number(f.trial ?? 0),
      tokens: Number(f.tokens ?? 0),
      tps: Number(f.throughput_tps ?? f.throughput ?? 0),
      ttft: Number(f.ttft_ms ?? f.ttft ?? 0),
      decode: Number(f.decode_ms_per_token ?? f.decode ?? 0),
      reward: Number(f.reward ?? 0),
      configLabel: String(f.config_id ?? f.config?.id ?? f.config ?? `cfg-${idx}`),
      configObj: (f.config && typeof f.config === 'object') ? f.config : { id: String(f.config_id ?? f.config ?? `cfg-${idx}`) },
      event: f.event && f.event.type ? { type: String(f.event.type), label: String(f.event.label || f.event.type) } : null,
      s, d, c, sN, dN, cN,
    });
  });
  return out;
}

function validateReplay(json) {
  if (!json || typeof json !== 'object') throw new Error('Replay JSON must be an object.');
  if (!Array.isArray(json.frames) || json.frames.length === 0) throw new Error('Replay JSON needs a non-empty "frames" array.');
  return true;
}

/* clearly-labeled procedural fallback — never presented as measured */
function buildSampleScaffold() {
  const rng = mulberry32(20260911);
  const N = 240, dt = 0.1;
  const phaseS = new Float32Array(POP.sensory.count);
  const phaseC = new Float32Array(POP.controller.count);
  for (let i = 0; i < phaseS.length; i++) phaseS[i] = rng() * Math.PI * 2;
  for (let i = 0; i < phaseC.length; i++) phaseC[i] = rng() * Math.PI * 2;
  const frames = [];
  let tokens = 0, best = -Infinity, bestCfg = 'scaffold-A';
  const cfgs = ['scaffold-A', 'scaffold-B', 'scaffold-C'];
  for (let k = 0; k < N; k++) {
    const t = k * dt;
    const trial = Math.floor(k / 48);
    tokens += 2 + Math.floor(rng() * 3);
    const tps = 34 + 9 * Math.sin(t * 1.3) + rng() * 3;
    const ttft = 110 + 22 * Math.sin(t * 0.5 + 1) + rng() * 6;
    const decode = 24 + 5 * Math.sin(t * 0.9) + rng() * 2;
    const reward = clamp01(0.55 + 0.3 * Math.sin(t * 0.55) + (rng() - 0.5) * 0.08 + k / N * 0.12);
    const cfg = cfgs[Math.floor(k / 80) % cfgs.length];
    if (reward > best) { best = reward; bestCfg = cfg; }
    const s = new Float32Array(POP.sensory.count);
    const d = new Float32Array(POP.dopamine.count);
    const cc = new Float32Array(POP.controller.count);
    const dMean = clamp01(0.45 + 0.4 * Math.sin(t * 0.8));
    for (let i = 0; i < s.length; i++) s[i] = clamp01(0.5 + 0.5 * Math.sin(t * 7 + phaseS[i]) * Math.sin(t * 1.7 + i * 0.01));
    for (let i = 0; i < d.length; i++) d[i] = clamp01(dMean + 0.12 * Math.sin(t * 1.1 + i * 0.05));
    const burst = Math.pow(Math.max(0, Math.sin(t * 2.2)), 3);
    for (let i = 0; i < cc.length; i++) cc[i] = clamp01(0.25 + 0.65 * burst * (0.5 + 0.5 * Math.sin(phaseC[i] + t * 3)));
    let event = null;
    if (k === 80) event = { type: 'config_change', label: 'scaffold-A → scaffold-B (sample)' };
    if (k === 160) event = { type: 'config_change', label: 'scaffold-B → scaffold-C (sample)' };
    if (k === N - 1) event = { type: 'best', label: `best ${bestCfg} r=${best.toFixed(2)} (sample)` };
    frames.push({
      key: k, t, trial, tokens, tps, ttft, decode, reward,
      configLabel: cfg, configObj: { id: cfg, note: 'procedural sample scaffold' },
      event, s, d, c: cc,
      sN: s.length, dN: d.length, cN: cc.length,
    });
  }
  player.mode = 'sample';
  player.meta = {
    model: 'sample-scaffold', prompt: 'procedural sample — no measured data',
    config: { id: 'scaffold-A' }, best: { config: bestCfg, reward: Number(best.toFixed(3)) },
  };
  player.frames = frames;
  player.duration = frames[frames.length - 1].t || 1;
  player.time = 0; player.lastEventKey = -1;
  player.sourceLabel = 'sample scaffold';
  player.provenance = 'sample-scaffold';
  player.coverage = {
    sensory: { measured: POP.sensory.count, expected: POP.sensory.count, buffered: POP.sensory.count, full: true },
    dopamine: { measured: POP.dopamine.count, expected: POP.dopamine.count, buffered: POP.dopamine.count, full: true },
    controller: { measured: POP.controller.count, expected: POP.controller.count, buffered: POP.controller.count, full: true },
    isFull: true, isPartial: false,
    expected: { sensory: POP.sensory.count, dopamine: POP.dopamine.count, controller: POP.controller.count },
  };
  setSourceBadge(false, null);
  refreshMetaUI();
  rebuildEventLog();
  announce('Sample scaffold loaded. Clearly labeled procedural data, not measured.');
}

function setSourceBadge(isLive, coverage) {
  const cov = coverage ?? player.coverage;
  if (!isLive) {
    els.badge.className = 'badge badge-sample';
    els.badge.textContent = 'SAMPLE SCAFFOLD · not measured';
    els.badge.title = 'procedural sample scaffold (provenance: sample-scaffold)';
    return;
  }
  // Never show an unqualified "measured" badge when replay coverage is partial.
  if (cov && cov.isPartial) {
    els.badge.className = 'badge badge-live';
    els.badge.textContent = `REPLAY · partial coverage ${coverageText(cov)} — only first N measured`;
    els.badge.title = `partial replay coverage ${coverageText(cov)} (provenance: replay file; tails zero, not measured)`;
    return;
  }
  els.badge.className = 'badge badge-live';
  els.badge.textContent = 'LIVE REPLAY · measured snapshots';
  els.badge.title = cov ? `full replay coverage ${coverageText(cov)} (provenance: replay file)` : 'measured replay snapshots';
}

async function loadReplayFromObject(json, label) {
  validateReplay(json);
  showLoading('Validating replay snapshots…');
  await new Promise((r) => setTimeout(r, 30));
  const meta = json.meta || {};
  const pop = meta.populations || (meta.num_neurons ? splitCounts(meta.num_neurons) : null);
  const frames = normalizeFrames(json.frames, pop);
  const exp = pop || { sensory: POP.sensory.count, dopamine: POP.dopamine.count, controller: POP.controller.count };
  const coverage = summarizeCoverage(frames, exp);
  // Optional embedded layout: positions Nx3 + regions N (0=sensory,1=dopamine,2=controller)
  if (json.layout && Array.isArray(json.layout.positions)) {
    applyEmbeddedLayout(json.layout);
  }
  player.mode = 'replay';
  player.meta = {
    model: String(meta.model || 'replay-model'),
    prompt: String(meta.prompt || '(prompt not provided)'),
    config: meta.config || {},
    best: meta.best || findBest(frames),
  };
  player.frames = frames;
  player.duration = frames[frames.length - 1].t || 1;
  player.time = 0; player.lastEventKey = -1;
  player.sourceLabel = label || 'replay';
  player.coverage = coverage;
  player.provenance = `replay:${player.sourceLabel}`;
  // Clear any stale scaffold/sample activity before the first interpolated frame.
  pSens.act.fill(0); pDopa.act.fill(0); pCtrl.act.fill(0);
  setSourceBadge(true, coverage);
  refreshMetaUI();
  rebuildEventLog();
  hideLoading();
  const covMsg = coverage.isFull
    ? `${frames.length} snapshots, ${fmtDur(player.duration)}`
    : `${frames.length} snapshots, ${fmtDur(player.duration)}, partial coverage ${coverageText(coverage)}`;
  setStatus(`replay loaded — ${covMsg}`);
  announce(coverage.isFull
    ? `Replay loaded. ${frames.length} snapshots.`
    : `Replay loaded with partial coverage ${coverageText(coverage)}. Only measured prefix shown; tails are zero.`);
}

function splitCounts(n) {
  // deterministic fallback split matching viewer proportions
  const s = Math.round(n * 0.66), d = Math.round(n * 0.1);
  return { sensory: s, dopamine: d, controller: Math.max(0, n - s - d) };
}

function findBest(frames) {
  let b = frames[0];
  for (const f of frames) if (f.reward > b.reward) b = f;
  return { config: b.configLabel, reward: Number(b.reward.toFixed(4)), t: b.t };
}

function applyEmbeddedLayout(layout) {
  try {
    const P = layout.positions, R = layout.regions;
    if (!Array.isArray(P) || P.length < 10) return;
    const buckets = [[], [], []];
    for (let i = 0; i < P.length; i++) {
      const r = Array.isArray(R) ? (R[i] | 0) : (i % 3 === 0 ? 0 : i % 3 === 1 ? 2 : 1);
      buckets[Math.min(2, Math.max(0, r))].push(P[i]);
    }
    const fill = (pop, list) => {
      const n = Math.min(pop.count, list.length);
      for (let i = 0; i < n; i++) {
        const p = list[i];
        pop.pos[i * 3] = Number(p[0]) || 0;
        pop.pos[i * 3 + 1] = Number(p[1]) || 0;
        pop.pos[i * 3 + 2] = Number(p[2]) || 0;
      }
      pop.geo.attributes.position.needsUpdate = true;
    };
    // regions: 0 sensory, 1 dopamine, 2 controller
    fill(pSens, buckets[0]); fill(pDopa, buckets[1]); fill(pCtrl, buckets[2]);
    toast(`Embedded layout applied (${P.length} positions).`);
  } catch (e) { console.warn('layout ignored', e); }
}

/* ---------- interpolation: write snapshots into typed buffers ---------- */
const cur = { t: 0, trial: 0, tokens: 0, tps: 0, ttft: 0, decode: 0, reward: 0, cfg: '', cfgObj: {} };

function sampleAt(time) {
  const F = player.frames;
  if (!F.length) return;
  const t = Math.max(0, Math.min(player.duration, time));
  let lo = 0, hi = F.length - 1;
  if (t <= F[0].t) { lo = 0; hi = 0; }
  else if (t >= F[hi].t) { lo = hi; hi = hi; }
  else {
    lo = 0;
    while (lo < hi - 1) {
      const m = (lo + hi) >> 1;
      if (F[m].t <= t) lo = m; else hi = m;
    }
    hi = lo + 1;
  }
  const A = F[lo], B = F[hi];
  const span = Math.max(1e-6, B.t - A.t);
  const a = hi === lo ? 0 : clamp01((t - A.t) / span);
  const L = (x, y) => x + (y - x) * a;
  cur.t = t; cur.trial = a < 0.5 ? A.trial : B.trial;
  cur.tokens = Math.round(L(A.tokens, B.tokens));
  cur.tps = L(A.tps, B.tps); cur.ttft = L(A.ttft, B.ttft);
  cur.decode = L(A.decode, B.decode); cur.reward = L(A.reward, B.reward);
  cur.cfg = a < 0.5 ? A.configLabel : B.configLabel;
  cur.cfgObj = a < 0.5 ? A.configObj : B.configObj;
  // lerp activity vectors directly into the GPU attribute arrays (no objects)
  // coverage integrity: interpolate only the measured prefix; zero tails so
  // smaller replays never show stale scaffold values.
  const S = pSens.act, D = pDopa.act, C = pCtrl.act;
  const mS = Math.min(A.sN ?? A.s.length, B.sN ?? B.s.length);
  const mD = Math.min(A.dN ?? A.d.length, B.dN ?? B.d.length);
  const mC = Math.min(A.cN ?? A.c.length, B.cN ?? B.c.length);
  const nS = Math.min(S.length, A.s.length, B.s.length, mS);
  const nD = Math.min(D.length, A.d.length, B.d.length, mD);
  const nC = Math.min(C.length, A.c.length, B.c.length, mC);
  for (let i = 0; i < nS; i++) S[i] = A.s[i] + (B.s[i] - A.s[i]) * a;
  for (let i = 0; i < nD; i++) D[i] = A.d[i] + (B.d[i] - A.d[i]) * a;
  for (let i = 0; i < nC; i++) C[i] = A.c[i] + (B.c[i] - A.c[i]) * a;
  // Tails beyond measured coverage are explicitly zero — never stale scaffold.
  if (nS < S.length) S.fill(0, nS);
  if (nD < D.length) D.fill(0, nD);
  if (nC < C.length) C.fill(0, nC);
  pSens.geo.attributes.aActivity.needsUpdate = true;
  pDopa.geo.attributes.aActivity.needsUpdate = true;
  pCtrl.geo.attributes.aActivity.needsUpdate = true;
  // dopamine mean drives global glow + link opacity (distinct broadcast signature)
  let dm = 0;
  for (let i = 0; i < nD; i++) dm += D[i];
  dm /= Math.max(1, nD);
  linkMat.opacity = 0.04 + dm * 0.12;
  scene.fog.density = 0.007 + dm * 0.004;
  return { idx: hi, alpha: a };
}

/* ---------- HUD / overlays (from snapshots only) ---------- */
let lastHud = 0;
function updateHUD(force) {
  const now = performance.now();
  if (!force && now - lastHud < 90) {
    els.progress.style.width = `${(player.time / player.duration) * 100}%`;
    return;
  }
  lastHud = now;
  const totalTrials = player.frames.length ? (player.frames[player.frames.length - 1].trial + 1) : 0;
  els.hTrial.textContent = `${cur.trial} / ${Math.max(0, totalTrials - 1)}`;
  els.hTokens.textContent = `${cur.tokens}`;
  els.hTime.textContent = `${cur.t.toFixed(1)}s / ${player.duration.toFixed(1)}s`;
  els.hTps.textContent = cur.tps ? `${cur.tps.toFixed(1)} tok/s` : '—';
  els.hTtft.textContent = cur.ttft ? `${cur.ttft.toFixed(0)} ms` : '—';
  els.hDecode.textContent = cur.decode ? `${cur.decode.toFixed(1)} ms/tok` : '—';
  els.hReward.textContent = Number.isFinite(cur.reward) ? cur.reward.toFixed(3) : '—';
  els.hConfig.textContent = cur.cfg || '—';
  els.progress.style.width = `${(player.time / player.duration) * 100}%`;
  if (document.activeElement !== els.scrub) els.scrub.value = String(Math.round((player.time / player.duration) * 1000));
  drawSpark();
}

function refreshMetaUI() {
  const m = player.meta || {};
  els.modelLine.textContent = `${m.model || '—'} · ${(m.prompt || '').slice(0, 140)}`;
  els.hPrompt.textContent = m.prompt || '—';
  els.dModel.textContent = m.model || '—';
  const cov = player.coverage;
  if (player.mode === 'replay' && cov) {
    const prov = player.provenance || 'replay file';
    els.dSource.textContent = cov.isFull
      ? `${player.sourceLabel} · measured (provenance: ${prov})`
      : `${player.sourceLabel} · PARTIAL ${coverageText(cov)} (provenance: ${prov}; tails zero)`;
    els.dNeurons.textContent = cov.isFull
      ? `${TOTAL} (s:${pSens.count} d:${pDopa.count} c:${pCtrl.count}) · full coverage`
      : `measured s:${cov.sensory.measured}/${pSens.count} d:${cov.dopamine.measured}/${pDopa.count} c:${cov.controller.measured}/${pCtrl.count} · buffers s:${pSens.count} d:${pDopa.count} c:${pCtrl.count}`;
  } else {
    els.dSource.textContent = `${player.sourceLabel} (provenance: sample-scaffold)`;
    els.dNeurons.textContent = `${TOTAL} (s:${pSens.count} d:${pDopa.count} c:${pCtrl.count})`;
  }
  els.dFrames.textContent = `${player.frames.length}`;
  els.dDur.textContent = fmtDur(player.duration);
  els.dConfig.textContent = JSON.stringify(m.config ?? {}, null, 2);
  const best = m.best;
  els.hBest.textContent = best ? `${best.config ?? best.id ?? '—'} · r=${best.reward ?? '—'}` : '—';
}

function rebuildEventLog() {
  els.log.innerHTML = '';
  const upto = currentFrameIndex();
  for (let i = 0; i <= upto; i++) {
    const e = player.frames[i]?.event;
    if (e) appendLog(e, player.frames[i]);
  }
  if (!els.log.children.length) {
    const li = document.createElement('li');
    li.textContent = player.mode === 'sample'
      ? 'sample scaffold — procedural markers only'
      : 'no config_change / best events in snapshots yet';
    els.log.appendChild(li);
  }
}

function appendLog(e, frame) {
  const only = els.log.querySelector('li:only-child');
  if (only && (only.textContent.includes('no config') || only.textContent.includes('procedural markers'))) only.remove();
  const li = document.createElement('li');
  li.className = e.type === 'best' ? 'best' : 'cfg';
  li.textContent = `t=${frame.t.toFixed(1)}s · ${e.type}: ${e.label}`;
  els.log.prepend(li);
  while (els.log.children.length > 12) els.log.lastChild.remove();
}

function currentFrameIndex() {
  const F = player.frames;
  let idx = 0;
  for (let i = 0; i < F.length; i++) { if (F[i].t <= player.time + 1e-9) idx = i; else break; }
  return idx;
}

function checkEvents() {
  const idx = currentFrameIndex();
  if (idx === player.lastEventKey) return;
  // deterministic rebuild when scrubbing backwards
  if (idx < player.lastEventKey) { player.lastEventKey = idx; rebuildEventLog(); return; }
  for (let i = player.lastEventKey + 1; i <= idx; i++) {
    const f = player.frames[i];
    if (f?.event) {
      appendLog(f.event, f);
      flashBanner(f.event);
      announce(`${f.event.type}: ${f.event.label}`);
    }
  }
  player.lastEventKey = idx;
}

let bannerTimer = 0;
function flashBanner(e) {
  els.banner.hidden = false;
  els.banner.className = 'event-banner' + (e.type === 'config_change' ? ' cfg' : '');
  els.banner.textContent = `${e.type === 'best' ? '★ BEST' : '⚙ CONFIG'} · ${e.label}`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { els.banner.hidden = true; }, 3200);
}

/* throughput / reward sparkline from snapshots */
function drawSpark() {
  const c = els.spark, ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const F = player.frames;
  if (F.length < 2) return;
  let mx = 0; for (const f of F) mx = Math.max(mx, f.tps);
  mx = Math.max(1, mx);
  const X = (t) => (t / player.duration) * W;
  ctx.strokeStyle = 'rgba(34,211,238,0.9)'; ctx.lineWidth = 1.5; ctx.beginPath();
  F.forEach((f, i) => { const x = X(f.t), y = H - 6 - (f.tps / mx) * (H - 14); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.strokeStyle = 'rgba(251,191,36,0.9)'; ctx.lineWidth = 1.5; ctx.beginPath();
  F.forEach((f, i) => { const x = X(f.t), y = H - 6 - clamp01(f.reward) * (H - 14); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.fillRect(X(player.time) - 1, 0, 2, H);
}

/* ---------- camera presets / tween ---------- */
const PRESETS = {
  dorsal:  { pos: [0, 52, 12],   tgt: [0, 0, 0] },
  frontal: { pos: [0, 6, 58],    tgt: [0, 0, 0] },
  lateral: { pos: [58, 10, 4],   tgt: [0, 0, 0] },
  vnc:     { pos: [12, -14, 30], tgt: [0, -13, 0] },
};
let camTween = null;
function flyTo(name) {
  const p = PRESETS[name];
  if (!p) return;
  camTween = {
    t: 0, dur: 1.1,
    p0: camera.position.clone(), p1: new THREE.Vector3(...p.pos),
    t0: controls.target.clone(), t1: new THREE.Vector3(...p.tgt),
  };
  announce(`Camera preset ${name}.`);
}
const ease = (x) => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;

/* ---------- cinematic mode ---------- */
let cine = false, cineTimer = 0, cineIdx = 0;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function setCine(on) {
  cine = on;
  document.body.classList.toggle('cine', on);
  controls.autoRotate = on;
  els.btnCine.setAttribute('aria-pressed', String(on));
  els.btnCine.textContent = on ? '⏸ Cinematic' : '🎬 Cinematic';
  if (on && !player.playing) setPlaying(true);
  announce(on ? 'Cinematic mode on.' : 'Cinematic mode off.');
}

/* ---------- transport ---------- */
function setPlaying(on) {
  player.playing = on;
  els.btnPlay.textContent = on ? '⏸ Pause' : '▶ Play';
  els.btnPlay.setAttribute('aria-pressed', String(on));
}
function replay() {
  player.time = 0; player.lastEventKey = -1;
  rebuildEventLog(); setPlaying(true);
  announce('Replaying from start.');
}

/* ---------- loading / toast / status ---------- */
function showLoading(msg) { els.loading.hidden = false; if (msg) els.loadingText.textContent = msg; }
function hideLoading() { els.loading.hidden = true; }
function toast(msg, ms = 5200) {
  els.toast.hidden = false; els.toast.textContent = msg;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { els.toast.hidden = true; }, ms);
}
function setStatus(s) { els.status.textContent = s; }
function announce(s) { els.live.textContent = s; }
const fmtDur = (s) => `${Number(s || 0).toFixed(1)}s`;

/* ---------- load paths: file / URL / ?replay= ---------- */
async function loadFromURL(url) {
  showLoading(`Fetching replay JSON…`);
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
    const json = await res.json();
    await loadReplayFromObject(json, url);
  } catch (e) {
    hideLoading();
    toast(`Replay load failed: ${e.message}. Still showing labeled sample scaffold.`);
    setStatus('replay load failed');
  }
}

/* ---------- capture: video + deterministic state collection ---------- */
let mediaRecorder = null, chunks = [], takeURL = null, takeBlob = null;
let recStart = 0, recTick = 0;
const take = { active: false, states: [], lastPush: 0 };

function pickMime() {
  const c = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  for (const m of c) { try { if (window.MediaRecorder?.isTypeSupported(m)) return m; } catch { /* noop */ } }
  return '';
}

function startRecording() {
  if (take.active) return;
  if (!('MediaRecorder' in window) || !els.canvas.captureStream) {
    toast('MediaRecorder / canvas.captureStream not supported in this browser. State-JSON capture still available — use Download JSON.');
  }
  take.active = true; take.states = []; take.lastPush = 0;
  chunks = [];
  try {
    const stream = els.canvas.captureStream(60);
    const mime = pickMime();
    mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : undefined);
    mediaRecorder.ondataavailable = (e) => { if (e.data?.size) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      takeBlob = new Blob(chunks, { type: chunks[0]?.type || 'video/webm' });
      if (takeURL) URL.revokeObjectURL(takeURL);
      takeURL = URL.createObjectURL(takeBlob);
      els.btnVideo.disabled = false; els.btnMp4.disabled = false;
      setStatus(`take ready — ${(takeBlob.size / 1e6).toFixed(2)} MB WebM, ${take.states.length} states`);
    };
    mediaRecorder.start(250);
  } catch (e) {
    toast(`Video recording could not start (${e.message}). Continuing state capture.`);
    mediaRecorder = null;
  }
  recStart = performance.now();
  els.recBadge.hidden = false;
  recTick = setInterval(() => {
    const s = Math.floor((performance.now() - recStart) / 1000);
    els.recTime.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }, 500);
  els.btnRecord.disabled = true; els.btnStop.disabled = false; els.btnJSON.disabled = true;
  if (!player.playing) setPlaying(true);
  announce('Recording demo: video plus deterministic replay states.');
}

function stopRecording() {
  if (!take.active) return;
  take.active = false;
  clearInterval(recTick);
  els.recBadge.hidden = true;
  try { mediaRecorder && mediaRecorder.state !== 'inactive' && mediaRecorder.stop(); } catch { /* noop */ }
  els.btnRecord.disabled = false; els.btnStop.disabled = true;
  els.btnJSON.disabled = take.states.length === 0;
  // video button is enabled in MediaRecorder.onstop once the blob exists
  setStatus(`recording stopped — ${take.states.length} states collected`);
  announce(`Recording stopped. ${take.states.length} replay states collected.`);
}

function pushRecordState() {
  // ~4 Hz deterministic snapshots (rounded) so the JSON take stays reloadable
  // Coverage integrity: record only measured lengths (never padded/stale tails).
  if (!take.active || player.time - take.lastPush < 0.25) return;
  take.lastPush = player.time;
  const m = measuredCounts();
  const r = (a, n) => Array.from(a.subarray(0, n), (v) => Math.round(v * 1000) / 1000);
  take.states.push({
    t: Math.round(player.time * 1000) / 1000,
    trial: cur.trial, tokens: cur.tokens,
    throughput_tps: Math.round(cur.tps * 100) / 100,
    ttft_ms: Math.round(cur.ttft * 10) / 10,
    decode_ms_per_token: Math.round(cur.decode * 100) / 100,
    reward: Math.round(cur.reward * 10000) / 10000,
    config_id: cur.cfg, config: cur.cfgObj,
    event: null, // crossed live events are appended below, never invented
    sensory: r(pSens.act, m.s), dopamine: r(pDopa.act, m.d), controller: r(pCtrl.act, m.c),
  });
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

function downloadTakeJSON() {
  if (!take.states.length) { toast('No recorded states yet — press Record demo first.'); return; }
  // re-attach crossed events deterministically from the source frames
  const withEvents = take.states.map((s) => {
    const src = player.frames.find((f) => Math.abs(f.t - s.t) < 0.13 && f.event);
    return src ? { ...s, event: src.event } : s;
  });
  const doc = {
    schema: 'fly-cns-replay/1',
    generator: 'viewer take recorder (snapshot-driven, interpolated states)',
    source: { mode: player.mode, label: player.sourceLabel, model: player.meta?.model ?? null, prompt: player.meta?.prompt ?? null, provenance: player.provenance ?? player.mode, coverage: player.coverage ?? undefined },
    meta: {
      model: player.meta?.model ?? 'recorded-take',
      prompt: player.meta?.prompt ?? '',
      populations: (() => { const mc = measuredCounts(); return { sensory: mc.s, dopamine: mc.d, controller: mc.c }; })(),
      config: player.meta?.config ?? {},
      best: player.meta?.best ?? findBest(player.frames),
      coverage: player.coverage ?? undefined,
      provenance: player.provenance ?? player.mode,
    },
    frames: withEvents,
  };
  download(new Blob([JSON.stringify(doc)], { type: 'application/json' }), 'fly-cns-take.json');
  setStatus(`replay JSON downloaded — ${withEvents.length} frames`);
}

/* optional MP4 conversion hook: POSTs WebM, expects MP4 back; WebM stays the fallback */
async function convertToMp4() {
  const endpoint = (els.mp4Url.value || window.__MP4_CONVERT_URL__ || '').trim();
  if (!endpoint) { toast('No MP4 hook configured — the WebM download is the final artifact. Add a hook URL to convert.'); return; }
  if (!takeBlob) { toast('No recorded take yet.'); return; }
  try {
    setStatus('converting to MP4 via hook…');
    const fd = new FormData();
    fd.append('file', takeBlob, 'take.webm');
    fd.append('config', JSON.stringify({ fps: 60 }));
    const res = await fetch(endpoint, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`hook HTTP ${res.status}`);
    const mp4 = new Blob([await res.blob()], { type: 'video/mp4' });
    download(mp4, 'fly-cns-demo.mp4');
    setStatus('MP4 downloaded via conversion hook');
  } catch (e) {
    toast(`MP4 hook failed (${e.message}). WebM fallback retained — use Download Video.`);
    setStatus('mp4 hook failed, webm fallback');
  }
}

/* ---------- events wiring ---------- */
els.btnPlay.onclick = () => setPlaying(!player.playing);
els.btnReplay.onclick = replay;
els.btnReset.onclick = () => flyTo('dorsal');
els.btnCine.onclick = () => setCine(!cine);
document.querySelectorAll('[data-preset]').forEach((b) => { b.onclick = () => flyTo(b.dataset.preset); });
els.speed.onchange = () => { player.speed = Number(els.speed.value) || 1; };
els.scrub.oninput = () => {
  player.time = (Number(els.scrub.value) / 1000) * player.duration;
  sampleAt(player.time); checkEvents(); updateHUD(true);
};
els.file.onchange = async () => {
  const f = els.file.files?.[0];
  if (!f) return;
  try {
    showLoading(`Reading ${f.name}…`);
    const json = JSON.parse(await f.text());
    await loadReplayFromObject(json, f.name);
  } catch (e) {
    hideLoading();
    toast(`Could not parse replay file: ${e.message}`);
  } finally { els.file.value = ''; }
};
els.btnUrl.onclick = () => {
  const u = els.url.value.trim() || './sample-replay.json';
  loadFromURL(u);
};
els.btnSample.onclick = () => { buildSampleScaffold(); setStatus('sample scaffold regenerated'); };
els.btnRecord.onclick = startRecording;
els.btnStop.onclick = stopRecording;
els.btnVideo.onclick = () => {
  if (!takeBlob) { toast('No video take yet — record first.'); return; }
  download(takeBlob, 'fly-cns-demo.webm');
};
els.btnJSON.onclick = downloadTakeJSON;
els.btnMp4.onclick = convertToMp4;

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!player.playing); }
  else if (e.key === 'r' || e.key === 'R') replay();
  else if (e.key === 'v' || e.key === 'V') flyTo('dorsal');
  else if (e.key === 'c' || e.key === 'C') setCine(!cine);
  else if (e.key === '1') flyTo('dorsal');
  else if (e.key === '2') flyTo('frontal');
  else if (e.key === '3') flyTo('lateral');
  else if (e.key === '4') flyTo('vnc');
});

function resize() {
  const w = els.viewport.clientWidth, h = els.viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
  for (const p of [pSens, pDopa, pCtrl]) p.mat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
}
window.addEventListener('resize', resize);

/* ---------- main loop ---------- */
const clock = new THREE.Clock();
let fpsAcc = 0, fpsN = 0, fpsLast = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, clock.getDelta());
  const t = clock.elapsedTime;

  if (player.playing && player.frames.length) {
    player.time += dt * player.speed;
    if (player.time >= player.duration) {
      player.time = player.duration;
      setPlaying(false);
      setStatus(player.mode === 'replay' ? 'replay complete' : 'sample loop complete');
    }
  }
  if (player.frames.length) {
    sampleAt(player.time);
    checkEvents();
    pushRecordState();
  }
  updateHUD(false);

  for (const p of [pSens, pDopa, pCtrl]) p.mat.uniforms.uTime.value = t;

  if (camTween) {
    camTween.t += dt;
    const k = ease(Math.min(1, camTween.t / camTween.dur));
    camera.position.lerpVectors(camTween.p0, camTween.p1, k);
    controls.target.lerpVectors(camTween.t0, camTween.t1, k);
    if (camTween.t >= camTween.dur) camTween = null;
  }
  if (cine && !reduceMotion) {
    cineTimer += dt;
    if (cineTimer > 14) {
      cineTimer = 0;
      const order = ['dorsal', 'lateral', 'frontal', 'vnc'];
      flyTo(order[cineIdx++ % order.length]);
    }
  }
  controls.update();
  renderer.render(scene, camera);

  fpsAcc += dt; fpsN++;
  const now = performance.now();
  if (now - fpsLast > 600) {
    const fps = fpsN / Math.max(1e-6, fpsAcc);
    els.fps.textContent = `${fps.toFixed(0)} fps · ${(TOTAL / 1000).toFixed(1)}k pts · ${renderer.info.render.calls} calls`;
    fpsAcc = 0; fpsN = 0; fpsLast = now;
  }
}

/* ---------- boot ---------- */
(async function boot() {
  showLoading('Initializing batched buffers…');
  resize();
  buildSampleScaffold();
  hideLoading();
  setPlaying(true);
  animate();
  const params = new URLSearchParams(location.search);
  const q = params.get('replay');
  if (q) { els.url.value = q; loadFromURL(q); }
  setStatus('ready — sample scaffold playing');
})();
