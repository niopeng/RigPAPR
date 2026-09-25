// Interactive rig viewer: PAPR points coloured by skinning weights, the skeleton, and the fitted motion.
// Data (static/rig/<name>.{json,bin}) is written by
// fast-papr-ys/motion_transfer_local/video_maker/web/export_rig_viewer.py: rest points, per-point top-k
// joint indices/weights (initial = Puppeteer transfer, refined = Phase 2), rest joint positions, and
// per-frame 3x4 joint deformation matrices. Skinning runs here: v' = sum_j w_j M_j v.
// The motion always uses the refined weights (the fitted result); the weight toggle changes only the colouring.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const GREY = new THREE.Color(0xd0d0d0);

function jointPalette(n) {
  const cols = [];
  for (let j = 0; j < n; j++) cols.push(new THREE.Color().setHSL((j * 0.618034) % 1, 0.65, 0.5));
  return cols;
}

// Heat colour for one joint's weight in [0, 1]: grey -> orange -> dark red.
function heat(w, out) {
  const a = new THREE.Color(0xd0d0d0), b = new THREE.Color(0xf39c12), c = new THREE.Color(0x8b0000);
  if (w < 0.5) out.copy(a).lerp(b, w * 2); else out.copy(b).lerp(c, (w - 0.5) * 2);
  return out;
}

async function load(base) {
  const meta = await (await fetch(base + '.json')).json();
  const buf = await (await fetch(base + '.bin')).arrayBuffer();
  const view = (key) => {
    const a = meta.arrays[key], n = a.shape.reduce((x, y) => x * y, 1);
    return a.dtype === 'uint8' ? new Uint8Array(buf, a.offset, n) : new Float32Array(buf, a.offset, n);
  };
  return { meta, points: view('points'), joints: view('joints'), matrices: view('matrices'),
           weights: { refined: [view('idx_refined'), view('w_refined')],
                      initial: [view('idx_initial'), view('w_initial')] } };
}

export async function mountRigViewer(el, base) {
  const data = await load(base);
  const { num_points: N, num_joints: J, num_frames: T, topk: K, parents, fps } = data.meta;

  // ---- scene
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xffffff);
  el.querySelector('.rig-canvas').appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.up.set(0, 0, 1);                                   // subjects are z-up
  const controls = new OrbitControls(camera, renderer.domElement);

  // ---- points
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const cloud = new THREE.Points(geom, new THREE.PointsMaterial({ size: 0.012, vertexColors: true }));
  scene.add(cloud);

  // ---- skeleton: bones as lines, joints as pickable dots
  const bonePairs = [];
  parents.forEach((p, j) => { if (p >= 0) bonePairs.push([j, p]); });
  const bonePos = new Float32Array(bonePairs.length * 6);
  const boneGeom = new THREE.BufferGeometry();
  boneGeom.setAttribute('position', new THREE.BufferAttribute(bonePos, 3));
  const bones = new THREE.LineSegments(boneGeom, new THREE.LineBasicMaterial({ color: 0x222222, depthTest: false }));
  bones.renderOrder = 1;
  scene.add(bones);
  const jointPos = new Float32Array(J * 3), jointCol = new Float32Array(J * 3);
  const jointGeom = new THREE.BufferGeometry();
  jointGeom.setAttribute('position', new THREE.BufferAttribute(jointPos, 3));
  jointGeom.setAttribute('color', new THREE.BufferAttribute(jointCol, 3));
  const jointDots = new THREE.Points(jointGeom, new THREE.PointsMaterial({ size: 9, sizeAttenuation: false,
    vertexColors: true, depthTest: false }));
  jointDots.renderOrder = 2;
  scene.add(jointDots);

  // ---- state
  const palette = jointPalette(J);
  let frame = 0, playing = true, weightSet = 'refined', selected = -1, showSkeleton = true;
  const [idxR, wR] = data.weights.refined;

  function pose(f) {
    const M = data.matrices, P = data.points, off = f * J * 12;
    for (let i = 0; i < N; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      let ox = 0, oy = 0, oz = 0;
      for (let k = 0; k < K; k++) {
        const w = wR[i * K + k];
        if (w === 0) continue;
        const m = off + idxR[i * K + k] * 12;
        ox += w * (M[m] * x + M[m + 1] * y + M[m + 2] * z + M[m + 3]);
        oy += w * (M[m + 4] * x + M[m + 5] * y + M[m + 6] * z + M[m + 7]);
        oz += w * (M[m + 8] * x + M[m + 9] * y + M[m + 10] * z + M[m + 11]);
      }
      pos[3 * i] = ox; pos[3 * i + 1] = oy; pos[3 * i + 2] = oz;
    }
    geom.attributes.position.needsUpdate = true;
    const R = data.joints;                                   // posed joint j = M_j * rest joint j
    for (let j = 0; j < J; j++) {
      const m = off + j * 12, x = R[3 * j], y = R[3 * j + 1], z = R[3 * j + 2];
      jointPos[3 * j] = M[m] * x + M[m + 1] * y + M[m + 2] * z + M[m + 3];
      jointPos[3 * j + 1] = M[m + 4] * x + M[m + 5] * y + M[m + 6] * z + M[m + 7];
      jointPos[3 * j + 2] = M[m + 8] * x + M[m + 9] * y + M[m + 10] * z + M[m + 11];
    }
    bonePairs.forEach(([a, b], n) => {
      bonePos.set(jointPos.subarray(3 * a, 3 * a + 3), 6 * n);
      bonePos.set(jointPos.subarray(3 * b, 3 * b + 3), 6 * n + 3);
    });
    boneGeom.attributes.position.needsUpdate = true;
    jointGeom.attributes.position.needsUpdate = true;
    jointGeom.computeBoundingSphere();
  }

  function colour() {
    const [idx, w] = data.weights[weightSet];
    const c = new THREE.Color(), acc = new THREE.Color();
    for (let i = 0; i < N; i++) {
      if (selected >= 0) {
        let ws = 0;
        for (let k = 0; k < K; k++) if (idx[i * K + k] === selected) ws += w[i * K + k];
        heat(ws, c);
      } else {
        acc.setRGB(0, 0, 0);
        for (let k = 0; k < K; k++) {
          const wk = w[i * K + k];
          if (wk > 0) acc.add(c.copy(palette[idx[i * K + k]]).multiplyScalar(wk));
        }
        c.copy(acc);
      }
      col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b;
    }
    geom.attributes.color.needsUpdate = true;
    for (let j = 0; j < J; j++) {
      const jc = j === selected ? new THREE.Color(0x000000) : (selected >= 0 ? GREY : palette[j]);
      jointCol[3 * j] = jc.r; jointCol[3 * j + 1] = jc.g; jointCol[3 * j + 2] = jc.b;
    }
    jointGeom.attributes.color.needsUpdate = true;
    el.querySelector('.rig-joint').textContent =
      selected >= 0 ? `Joint ${selected}: weight heatmap` : 'Colour = blend of joint colours by skinning weight';
  }

  // ---- camera framing from the rest pose
  pose(0);
  geom.computeBoundingBox();
  const box = geom.boundingBox, centre = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const r = size.length() / 2;
  controls.target.copy(centre);
  // Side view of T-Rex, head to the right (its body runs along y). Per-subject framing can move to the JSON.
  camera.position.copy(centre).add(new THREE.Vector3(-r * 2.2, r * 1.25, r * 0.6));
  controls.update();

  // ---- UI
  const slider = el.querySelector('.rig-frame'), label = el.querySelector('.rig-frame-label');
  const playBtn = el.querySelector('.rig-play');
  slider.max = T - 1;
  function setFrame(f) {
    frame = f; slider.value = f; pose(f);
    label.textContent = f === 0 ? 'rest pose' : `frame ${data.meta.frames[f]}`;
  }
  slider.addEventListener('input', () => { playing = false; playBtn.textContent = 'Play'; setFrame(+slider.value); });
  playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; });
  el.querySelectorAll('.rig-weights button').forEach((b) => b.addEventListener('click', () => {
    el.querySelectorAll('.rig-weights button').forEach((o) => o.classList.remove('is-dark', 'is-selected'));
    b.classList.add('is-dark', 'is-selected');
    weightSet = b.dataset.value; colour();
  }));
  el.querySelector('.rig-skeleton').addEventListener('change', (e) => {
    showSkeleton = e.target.checked; bones.visible = jointDots.visible = showSkeleton;
  });

  // Click a joint to see its weights; click empty space to go back. Drags (orbiting) are ignored.
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = r * 0.03;
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1,
                                        -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
    const hit = showSkeleton ? ray.intersectObject(jointDots)[0] : null;
    selected = hit ? hit.index : -1;
    colour();
  });

  function resize() {
    const w = el.querySelector('.rig-canvas').clientWidth, h = el.querySelector('.rig-canvas').clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize(); colour(); setFrame(0);

  // Play forward and back (rest pose -> end of the clip -> rest pose), like the result clips.
  let last = 0, dir = 1;
  renderer.setAnimationLoop((t) => {
    if (playing && t - last > 1000 / fps) {
      last = t;
      if (frame + dir < 0 || frame + dir >= T) dir = -dir;
      setFrame(frame + dir);
    }
    controls.update();
    renderer.render(scene, camera);
  });
}
