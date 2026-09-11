/* TensorFly real-MaleCNS viewer.  It renders only downloaded skeleton data. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.querySelector('#c');
const status = document.querySelector('#status');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x05070d, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1e8);
camera.position.set(0, 0, 90000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.25));
const group = new THREE.Group(); scene.add(group);
const params = new URLSearchParams(location.search);
const morphologyUrl = params.get('morphology') || './real-morphology.json';
const replayUrl = params.get('replay') || './tensorfly_replay.json';
let replay = null;
let geometries = new Map();

function bodyKey(id) { return String(id); } // IDs deliberately remain strings in JS.
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) {
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
}
function colorFor(population) {
  return ({ sensory: 0x35c8ff, dopamine: 0xff4bb7, controller: 0xffcb4d })[population] || 0x64748b;
}
function validateMorphology(doc) {
  if (!doc || doc.schema !== 'tensorfly-real-morphology/1' || !Array.isArray(doc.skeletons)) {
    throw new Error('Expected tensorfly-real-morphology/1 generated from downloaded MaleCNS SWC files.');
  }
  if (!doc.provenance || doc.provenance.is_synthetic) throw new Error('Synthetic morphology is rejected.');
}
function addSkeleton(skeleton) {
  if (!Array.isArray(skeleton.positions) || !Array.isArray(skeleton.segments)) return;
  const positions = new Float32Array(skeleton.positions);
  const segments = new Uint32Array(skeleton.segments);
  if (!positions.length || segments.length % 2) return;
  const expanded = new Float32Array(segments.length * 3);
  for (let i = 0; i < segments.length; i++) {
    const at = segments[i] * 3;
    if (at + 2 >= positions.length) return;
    expanded[i * 3] = positions[at]; expanded[i * 3 + 1] = positions[at + 1]; expanded[i * 3 + 2] = positions[at + 2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(expanded, 3));
  const base = new THREE.Color(colorFor(skeleton.population));
  const colors = new Float32Array((expanded.length / 3) * 3);
  for (let i = 0; i < colors.length; i += 3) { colors[i] = base.r; colors[i + 1] = base.g; colors[i + 2] = base.b; }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.34, blending: THREE.AdditiveBlending });
  const lines = new THREE.LineSegments(geometry, material); group.add(lines);
  geometries.set(bodyKey(skeleton.body_id), { geometry, material, base });
}
function addActualEdges(doc) {
  // Edges are an optional, deterministic subset emitted by preparation.  No RNG.
  for (const edge of doc.connectivity_edges || []) {
    if (!Array.isArray(edge.positions) || edge.positions.length !== 6) continue;
    if (!geometries.has(bodyKey(edge.pre_body_id)) || !geometries.has(bodyKey(edge.post_body_id))) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edge.positions), 3));
    group.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x214a64, transparent: true, opacity: 0.12 })));
  }
}
function applyFrame(frame) {
  if (!frame) return;
  const activity = frame.activity_by_body_id || {};
  for (const [id, entry] of geometries) {
    const v = Math.max(0, Math.min(1, Number(activity[id] || 0)));
    entry.material.opacity = 0.20 + 0.80 * v;
    entry.material.color.copy(entry.base).multiplyScalar(0.35 + 2.5 * v);
  }
}
async function load() {
  const [morphologyResponse, replayResponse] = await Promise.all([fetch(morphologyUrl), fetch(replayUrl)]);
  if (!morphologyResponse.ok) throw new Error(`Morphology unavailable (${morphologyResponse.status}); run tensorfly.prepare() first.`);
  const morphology = await morphologyResponse.json(); validateMorphology(morphology);
  morphology.skeletons.forEach(addSkeleton); addActualEdges(morphology);
  if (replayResponse.ok) {
    replay = await replayResponse.json();
    if (replay.meta?.provenance?.is_synthetic) throw new Error('Synthetic replay is rejected by the production viewer.');
  }
  status.textContent = `REAL MALECNS · ${morphology.skeletons.length} skeletons · ${morphology.provenance.retention_policy}`;
}
let started = performance.now();
function render(now) {
  resize(); controls.update();
  if (replay?.frames?.length) {
    const index = Math.floor((now - started) / 500) % replay.frames.length;
    applyFrame(replay.frames[index]);
  }
  renderer.render(scene, camera); requestAnimationFrame(render);
}
load().catch(error => { status.textContent = `VIEWER BLOCKED: ${error.message}`; console.error(error); });
requestAnimationFrame(render);
