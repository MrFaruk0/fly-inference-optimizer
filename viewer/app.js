/* TensorFly browser-native MaleCNS replay viewer. Real SWC coordinates only. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.querySelector('#c');
const status = document.querySelector('#status');
const playButton = document.querySelector('#play');
const cinematicButton = document.querySelector('#cinematic');
const recordButton = document.querySelector('#record');
const stopButton = document.querySelector('#stop');
const download = document.querySelector('#download');
const resetButton = document.querySelector('#reset');
const timeline = document.querySelector('#timeline');
const metrics = document.querySelector('#metrics');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setClearColor(0x03050b, 1);
const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2e8);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0x9ecbff, 0.28)); const neurons = new THREE.Group(); scene.add(neurons);
const params = new URLSearchParams(location.search); const morphologyUrl = params.get('morphology') || './real-morphology.json'; const replayUrl = params.get('replay') || './tensorfly_replay.json';
const colors = { sensory: new THREE.Color(0x31c6ff), dopamine: new THREE.Color(0xff4da6), controller: new THREE.Color(0xffc94d), context: new THREE.Color(0x58708e) };
const batches = new Map(); let replay = null; let running = true; let cinematic = false; let frameCursor = 0; let demoStart = 0;
let recorder = null; let recorded = []; let downloadUrl = null; let centre = new THREE.Vector3(); let radius = 90000;

function bodyKey(id) { return String(id); } // Never round uint64 body IDs in JavaScript.
function validateMorphology(doc) {
  if (!doc || doc.schema !== 'tensorfly-real-morphology/1' || !Array.isArray(doc.skeletons)) throw new Error('Expected source-derived tensorfly-real-morphology/1.');
  if (!doc.provenance || doc.provenance.is_synthetic || doc.provenance.procedural_geometry) throw new Error('Synthetic or procedural anatomy is rejected.');
}
function batchFor(name) { if (!batches.has(name)) batches.set(name, { positions: [], colors: [], ranges: new Map(), base: colors[name] || colors.context }); return batches.get(name); }
function ingestSkeleton(skeleton) {
  if (!Array.isArray(skeleton.positions) || !Array.isArray(skeleton.segments)) return;
  const points = skeleton.positions; const segments = skeleton.segments; if (!points.length || segments.length % 2) return;
  const batch = batchFor(skeleton.population || 'context'); const start = batch.colors.length / 3;
  for (const node of segments) {
    const at = Number(node) * 3; if (at < 0 || at + 2 >= points.length) return;
    batch.positions.push(points[at], points[at + 1], points[at + 2]); batch.colors.push(batch.base.r * .35, batch.base.g * .35, batch.base.b * .35);
  }
  batch.ranges.set(bodyKey(skeleton.body_id), { start, count: segments.length, base: batch.base });
}
function finishBatches() {
  for (const batch of batches.values()) {
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(batch.positions, 3));
    const color = new THREE.Float32BufferAttribute(batch.colors, 3); color.setUsage(THREE.DynamicDrawUsage); geometry.setAttribute('color', color); geometry.computeBoundingSphere();
    neurons.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: .88, blending: THREE.AdditiveBlending, depthWrite: false })));
    batch.geometry = geometry; batch.color = color;
  }
  const box = new THREE.Box3().setFromObject(neurons); box.getCenter(centre); radius = Math.max(1, box.getSize(new THREE.Vector3()).length() * .62); resetView();
}
function addActualEdges(doc) {
  const positions = []; for (const edge of doc.connectivity_edges || []) if (Array.isArray(edge.positions) && edge.positions.length === 6) positions.push(...edge.positions);
  if (!positions.length) return; const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  neurons.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x1d5574, transparent: true, opacity: .11, blending: THREE.AdditiveBlending, depthWrite: false })));
}
function valueFor(frame, id) { return Math.max(0, Math.min(1, Number(frame?.activity_by_body_id?.[id] || 0) + Number(frame?.sensory_drive_by_body_id?.[id] || 0) + Math.abs(Number(frame?.modulatory_drive_by_body_id?.[id] || 0)))); }
function applyFrame(frame, next = frame, mix = 0) {
  if (!frame) return;
  for (const batch of batches.values()) {
    const data = batch.color.array;
    for (const [id, range] of batch.ranges) {
      const gain = .22 + 2.7 * THREE.MathUtils.lerp(valueFor(frame, id), valueFor(next, id), mix);
      for (let vertex = range.start; vertex < range.start + range.count; vertex++) { const at = vertex * 3; data[at] = range.base.r * gain; data[at + 1] = range.base.g * gain; data[at + 2] = range.base.b * gain; }
    }
    batch.color.needsUpdate = true;
  }
  const raw = frame.raw_inference_metrics || frame;
  metrics.innerHTML = `<b>TRIAL ${Number(frame.trial ?? 0) + 1}</b><span>TTFT ${Number(raw.ttft_ms || 0).toFixed(1)} ms</span><span>TPOT ${Number(raw.tpot_ms || raw.decode_ms_per_token || 0).toFixed(1)} ms</span><span>${Number(raw.throughput_tps || 0).toFixed(2)} tok/s</span><span>reward ${Number(frame.reward || 0).toFixed(3)}</span><em>${frame.chosen_action || frame.events?.configuration_change?.action || 'replay'}</em>`;
}
function resetView() { camera.position.copy(centre).add(new THREE.Vector3(0, radius * .15, radius * 2.2)); controls.target.copy(centre); controls.update(); }
function updateCinematic(now) { const a = ((now - demoStart) / 1000) * .34; camera.position.set(centre.x + Math.cos(a) * radius * 2.05, centre.y + radius * (.28 + .15 * Math.sin(a * .7)), centre.z + Math.sin(a) * radius * 2.05); camera.lookAt(centre); }
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); } }
function render(now) {
  resize(); if (cinematic) updateCinematic(now); else controls.update();
  if (replay?.frames?.length) { if (running) frameCursor = ((now - demoStart) / 1800) % replay.frames.length; const low = Math.floor(frameCursor); const high = (low + 1) % replay.frames.length; applyFrame(replay.frames[low], replay.frames[high], frameCursor - low); timeline.value = String(low); }
  renderer.render(scene, camera); requestAnimationFrame(render);
}
function startRecording() {
  if (!window.MediaRecorder) throw new Error('This browser does not support MediaRecorder.'); recorded = []; if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  const type = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm'; recorder = new MediaRecorder(canvas.captureStream(60), { mimeType: type });
  recorder.ondataavailable = event => { if (event.data.size) recorded.push(event.data); }; recorder.onstop = () => { downloadUrl = URL.createObjectURL(new Blob(recorded, { type: 'video/webm' })); download.href = downloadUrl; download.hidden = false; };
  recorder.start(1000); recordButton.disabled = true; stopButton.disabled = false; status.textContent = 'RECORDING REAL REPLAY · browser WebM';
}
function stopRecording() { if (recorder?.state === 'recording') recorder.stop(); recordButton.disabled = false; stopButton.disabled = true; }
async function load() {
  const [morphologyResponse, replayResponse] = await Promise.all([fetch(morphologyUrl), fetch(replayUrl)]);
  if (!morphologyResponse.ok) throw new Error(`Morphology unavailable (${morphologyResponse.status}); run tensorfly.prepare() first.`);
  const morphology = await morphologyResponse.json(); validateMorphology(morphology); morphology.skeletons.forEach(ingestSkeleton); finishBatches(); addActualEdges(morphology);
  if (!replayResponse.ok) throw new Error(`Replay unavailable (${replayResponse.status}); run experiment.export_video() first.`);
  replay = await replayResponse.json(); if (replay.meta?.provenance?.is_synthetic) throw new Error('Synthetic replay is rejected.'); timeline.max = String(Math.max(0, replay.frames.length - 1));
  status.textContent = `REAL MALECNS · ${morphology.skeletons.length} SWC skeletons · ${replay.frames.length} measured trials`; demoStart = performance.now();
}
playButton.onclick = () => { running = !running; playButton.textContent = running ? 'Pause Replay' : 'Replay'; demoStart = performance.now() - frameCursor * 1800; };
cinematicButton.onclick = () => { cinematic = !cinematic; cinematicButton.classList.toggle('active', cinematic); demoStart = performance.now(); };
resetButton.onclick = resetView; recordButton.onclick = () => { cinematic = true; startRecording(); }; stopButton.onclick = stopRecording;
timeline.oninput = () => { running = false; frameCursor = Number(timeline.value); playButton.textContent = 'Replay'; };
load().catch(error => { status.textContent = `VIEWER BLOCKED: ${error.message}`; console.error(error); }); requestAnimationFrame(render);
