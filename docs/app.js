// app.js — main-thread DOM/UI layer.
//
// All Pyodide work happens in docs/worker.js. The main thread only:
//   - runs the Three.js viewers (PathViewer + PlannerViewer)
//   - drives the loader / live-output / metrics UI
//   - relays user actions to the worker over postMessage
//
// This is what keeps the page responsive: Python execution sits on a separate
// OS thread, so the spinner spins, dropdowns work, and stdout from the planner
// streams in live (each Python `print` posts a `stdout` message, the main
// thread's event loop picks it up between repaints).

import { PathViewer, PlannerViewer } from "./viewer.js";

const WORKER_URL    = "worker.js";
const NUM_WAYPOINTS = 10;
const PATH_START    = [0, 0, 0];          // start fixed at origin for both shapes
const S_END         = [-2.3, 0, 0.8];     // S-shape end (cubic Bezier endpoint)

// ────────────────────────────────────────────────────────────────────────────
//  DOM handles
// ────────────────────────────────────────────────────────────────────────────
const loaderEl       = document.getElementById("loader");
const loaderTitle    = document.getElementById("loader-title");
const loaderPct      = document.getElementById("loader-pct");
const loaderFill     = document.getElementById("loader-fill");
const loaderSteps    = document.getElementById("loader-steps");

const runBtn         = document.getElementById("run-btn");
const plannerSel     = document.getElementById("planner-select");
const shapeLibSel    = document.getElementById("shape-lib-select");
const interpInput    = document.getElementById("interp-steps");
const gammaSlider    = document.getElementById("gamma-slider");
const gammaReadout   = document.getElementById("gamma-value");

const resultsEl      = document.getElementById("results");
const playbackEl     = document.getElementById("playback");
const playBtn        = document.getElementById("play-btn");
const frameSlider    = document.getElementById("frame-slider");
const frameReadout   = document.getElementById("frame-readout");

const pathViewerEl    = document.getElementById("path-viewer");
const plannerViewerEl = document.getElementById("planner-viewer");

const pathShapeSel    = document.getElementById("path-shape");
const pathSlidersS    = document.getElementById("path-sliders-s");
const pathSlidersC    = document.getElementById("path-sliders-c");
const pathRestoreBtn  = document.getElementById("path-restore");

const runOutputEl     = document.getElementById("run-output");
const outputIndicator = document.getElementById("output-indicator");
const outputLabel     = document.getElementById("output-label");

// ────────────────────────────────────────────────────────────────────────────
//  Loader / progress UI
// ────────────────────────────────────────────────────────────────────────────
const STEP_WEIGHTS = {
  pyodide:  18,
  packages: 60,
  bundle:   12,
  ready:    10,
};

let pctBase = 0;
let activeStep = null;

function setLoaderTitle(text) { if (loaderTitle) loaderTitle.textContent = text; }
function setLoaderPct(pct) {
  pct = Math.max(0, Math.min(100, pct));
  if (loaderFill) loaderFill.style.width = `${pct}%`;
  if (loaderPct)  loaderPct.textContent  = `${Math.round(pct)}%`;
}
function startStep(stepName, title) {
  activeStep = stepName;
  for (const li of loaderSteps.children) {
    li.classList.remove("active");
    if (li.dataset.step === stepName) li.classList.add("active");
  }
  setLoaderTitle(title);
  setLoaderPct(pctBase);
}
function progressStep(fraction) {
  if (!activeStep) return;
  const w = STEP_WEIGHTS[activeStep] || 0;
  setLoaderPct(pctBase + w * Math.max(0, Math.min(1, fraction)));
}
function completeStep(stepName) {
  pctBase += STEP_WEIGHTS[stepName] || 0;
  for (const li of loaderSteps.children) {
    if (li.dataset.step === stepName) {
      li.classList.remove("active");
      li.classList.add("done");
    }
  }
  setLoaderPct(pctBase);
}
function loaderReady() {
  setLoaderPct(100);
  setLoaderTitle("Ready");
  loaderEl.classList.add("loader--done");
  setTimeout(() => { loaderEl.hidden = true; }, 600);
}
function loaderError(msg) {
  loaderEl.classList.add("loader--error");
  setLoaderTitle(`Error: ${msg}`);
}

// ────────────────────────────────────────────────────────────────────────────
//  Live output + persistent runtime log
// ────────────────────────────────────────────────────────────────────────────
const MAX_OUTPUT_LINES = 200;
let outputLines = [];

function setOutputState(state, label) {
  outputIndicator.classList.remove("idle", "busy", "ok", "err");
  outputIndicator.classList.add(state);
  outputLabel.textContent = label;
}
function clearOutput() {
  outputLines = [];
  runOutputEl.textContent = "";
}
function appendOutput(line) {
  outputLines.push(line);
  if (outputLines.length > MAX_OUTPUT_LINES) outputLines.shift();
  runOutputEl.textContent = outputLines.join("\n");
  runOutputEl.scrollTop = runOutputEl.scrollHeight;
}
function pyOut(text) {
  if (!text) return;
  appendOutput(text);
}
function appLog(...parts) {
  // App-level status messages — kept for the dev console; the visible page no
  // longer carries a "Full runtime log" section.
  const line = "[demo] " + parts
    .map((p) => (typeof p === "string" ? p : JSON.stringify(p)))
    .join(" ");
  console.log(line);
}

// ────────────────────────────────────────────────────────────────────────────
//  Path generation (mirrors src/PathGenerators/PathGenerator.py)
//
//  Computed on the main thread for instant slider feedback. The waypoints are
//  shipped to the worker only when the user clicks Run.
// ────────────────────────────────────────────────────────────────────────────
function v_add(a, b)   { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function v_sub(a, b)   { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function v_scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
function v_norm(a)     { return Math.hypot(a[0], a[1], a[2]); }
function v_cross(a, b) {
  return [a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]];
}
function v_rot_axis(v, axis, angleRad) {
  // Rodrigues' formula. axis must be a unit vector.
  const c  = Math.cos(angleRad), s = Math.sin(angleRad), oc = 1 - c;
  const dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2];
  const cr  = v_cross(axis, v);
  return [
    v[0]*c + cr[0]*s + axis[0]*dot*oc,
    v[1]*c + cr[1]*s + axis[1]*dot*oc,
    v[2]*c + cr[2]*s + axis[2]*dot*oc,
  ];
}

/** Cubic Bezier through 4 control points, sampled at n equally-spaced t. */
function cubicBezier(p0, p1, p2, p3, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    const omt = 1 - t;
    out.push([
      omt**3 * p0[0] + 3*omt**2*t * p1[0] + 3*omt*t**2 * p2[0] + t**3 * p3[0],
      omt**3 * p0[1] + 3*omt**2*t * p1[1] + 3*omt*t**2 * p2[1] + t**3 * p3[1],
      omt**3 * p0[2] + 3*omt**2*t * p1[2] + 3*omt*t**2 * p2[2] + t**3 * p3[2],
    ]);
  }
  return out;
}

/** Half-circle arc from start to end with bending plane rotated by radial_angle
 *  about the chord. Mirrors PathGenerator.generate_c_shape_path including the
 *  `[len // 4:]` truncation. */
function semiCircleArc(start, end, radialAngleDeg, n) {
  const chord = v_sub(end, start);
  const L = v_norm(chord);
  if (L < 1e-6) return Array.from({ length: n }, () => start.slice());

  const center = v_scale(v_add(start, end), 0.5);
  const radius = L / 2;
  const v1     = v_scale(v_sub(start, center), 1 / radius);

  let vPerp;
  if (Math.abs(chord[2]) < 0.9 * L) vPerp = v_cross(chord, [0, 0, 1]);
  else                              vPerp = v_cross(chord, [1, 0, 0]);
  vPerp = v_scale(vPerp, 1 / v_norm(vPerp));

  const v2 = v_rot_axis(vPerp, v_scale(chord, 1 / L), radialAngleDeg * Math.PI / 180);

  const out = [];
  for (let i = 0; i < n; i++) {
    const t = (n === 1 ? 0 : (i / (n - 1))) * Math.PI;
    const cosT = Math.cos(t), sinT = Math.sin(t);
    out.push([
      center[0] + radius * cosT * v1[0] + radius * sinT * v2[0],
      center[1] + radius * cosT * v1[1] + radius * sinT * v2[1],
      center[2] + radius * cosT * v1[2] + radius * sinT * v2[2],
    ]);
  }
  return out.slice(Math.floor(n / 4));
}

/** Read the current S-shape sliders and produce its waypoint array. */
function curveFromSSliders() {
  const c1 = [readNum("s-c1x"), 0, readNum("s-c1z")];
  const c2 = [readNum("s-c2x"), 0, readNum("s-c2z")];
  return cubicBezier(PATH_START, c1, c2, S_END, NUM_WAYPOINTS);
}

/** Read the current C-shape sliders and produce its waypoint array. */
function curveFromCSliders() {
  const end   = [readNum("c-ex"), 0, readNum("c-ez")];
  const angle = readNum("c-angle");
  return semiCircleArc(PATH_START, end, angle, NUM_WAYPOINTS);
}

function readNum(id) { return parseFloat(document.getElementById(id).value); }

function computeCurrentCurve() {
  return pathShapeSel.value === "c" ? curveFromCSliders() : curveFromSSliders();
}

/** Apply a curve to both viewers and remember it as the current target.
 *  If a planned motion is already loaded, drop it (the path it was planned
 *  against has just changed). */
function setTargetPath(curve, { resetPlan = true } = {}) {
  currentCurve = curve;
  pathViewer.showTargetPath(curve);

  if (resetPlan && plannerViewer.frames) {
    plannerViewer.pause();
    plannerViewer.frames = null;
    playBtn.textContent = "▶";
    playbackEl.hidden   = true;
    resultsEl.hidden    = true;
    setOutputState("idle", "Path edited — re-run to see the planned motion.");
  }

  plannerViewer.showTargetPath(curve);
  if (defaultRobotState) plannerViewer.showDefaultRobot(defaultRobotState);
}

let currentCurve       = null;     // the waypoints array we feed to the planner
let defaultRobotState  = null;     // populated when the worker reports ready

// ────────────────────────────────────────────────────────────────────────────
//  Three.js viewers (live as soon as the page loads)
// ────────────────────────────────────────────────────────────────────────────
const pathViewer    = new PathViewer(pathViewerEl);
const plannerViewer = new PlannerViewer(plannerViewerEl);
plannerViewer.frameChangeListener = (i) => syncFrameUi(i);

function syncFrameUi(i) {
  if (!plannerViewer.frames) return;
  frameSlider.value = String(i);
  frameReadout.textContent = `${i + 1} / ${plannerViewer.frames.length}`;
}

// ────────────────────────────────────────────────────────────────────────────
//  UI helpers
// ────────────────────────────────────────────────────────────────────────────
function setControlsEnabled(enabled) {
  runBtn.disabled         = !enabled;
  plannerSel.disabled     = !enabled;
  shapeLibSel.disabled    = !enabled;
  interpInput.disabled    = !enabled;
  pathShapeSel.disabled   = !enabled;
  pathRestoreBtn.disabled = !enabled;
  for (const id of PATH_SLIDER_IDS) {
    document.getElementById(id).disabled = !enabled;
  }
  syncGammaSliderState(enabled);
}
function syncGammaSliderState(enabled) {
  gammaSlider.disabled = !enabled || plannerSel.value !== "Threshold Cluster";
}
function setMetric(name, value) {
  const el = resultsEl.querySelector(`[data-metric="${name}"]`);
  if (el) el.textContent = value;
}
function formatDev(dev) {
  if (!dev) return "—";
  return `${dev.mean.toFixed(3)}% / ${dev.max.toFixed(3)}%`;
}

let _toastTimer = null;
function toast(message, durationMs = 3200) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
  el.classList.add("toast--show");
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    el.classList.remove("toast--show");
    setTimeout(() => { el.hidden = true; }, 220);
  }, durationMs);
}

// ────────────────────────────────────────────────────────────────────────────
//  Worker setup + request/response plumbing
// ────────────────────────────────────────────────────────────────────────────
const worker = new Worker(WORKER_URL);

let nextCallId = 0;
const pendingCalls = new Map();

function workerCall(type, params = {}) {
  const id = ++nextCallId;
  return new Promise((resolve, reject) => {
    pendingCalls.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...params });
  });
}

worker.onerror = (e) => {
  console.error("[worker error]", e);
  appLog("worker error:", e.message || String(e));
  loaderError(e.message || "worker crashed");
  setOutputState("err", "Worker crashed — see runtime log.");
};

worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "stage":
      startStep(msg.step, msg.title);
      break;
    case "stageProgress":
      if (msg.step === activeStep) progressStep(msg.fraction);
      break;
    case "stageDone":
      completeStep(msg.step);
      break;
    case "ready":
      onWorkerReady(msg);
      break;
    case "stdout":
      pyOut(msg.text);
      break;
    case "log":
      appLog(msg.text);
      break;
    case "response": {
      const slot = pendingCalls.get(msg.id);
      if (slot) { pendingCalls.delete(msg.id); slot.resolve(msg.payload); }
      break;
    }
    case "error": {
      if (msg.id != null) {
        const slot = pendingCalls.get(msg.id);
        if (slot) { pendingCalls.delete(msg.id); slot.reject(new Error(msg.message)); }
      } else {
        loaderError(msg.message);
        setOutputState("err", "Setup failed.");
      }
      break;
    }
    default:
      console.warn("[unknown worker message]", msg);
  }
};

function onWorkerReady({ defaultRobot }) {
  // Worker only sends us the robot's default pose; the curve is computed
  // entirely on the main thread from the slider values, so editing is instant.
  defaultRobotState = defaultRobot;
  refreshAllReadouts();
  setTargetPath(computeCurrentCurve(), { resetPlan: false });

  loaderReady();
  setOutputState("idle", "Idle — pick a planner and hit Run.");
  setControlsEnabled(true);

  runBtn.addEventListener("click", onRunClicked);
  plannerSel.addEventListener("change", () => syncGammaSliderState(true));
  gammaSlider.addEventListener("input", () => {
    gammaReadout.textContent = parseFloat(gammaSlider.value).toFixed(2);
  });
  playBtn.addEventListener("click", () => {
    if (!plannerViewer.frames) {
      toast("Run the planner first — there's no motion plan to play yet.");
      return;
    }
    plannerViewer.togglePlay();
    playBtn.textContent = plannerViewer.playing ? "⏸" : "▶";
  });
  frameSlider.addEventListener("input", () => {
    if (!plannerViewer.frames) {
      toast("Run the planner first — there's no motion plan to scrub yet.");
      return;
    }
    plannerViewer.pause();
    playBtn.textContent = "▶";
    plannerViewer.setFrame(parseInt(frameSlider.value, 10));
  });

  // Path-editing controls.
  pathShapeSel.addEventListener("change", onPathShapeChanged);
  pathRestoreBtn.addEventListener("click", onPathRestoreClicked);
  for (const id of PATH_SLIDER_IDS) {
    document.getElementById(id).addEventListener("input", onPathSliderInput);
  }
}

const PATH_SLIDER_IDS = ["s-c1x", "s-c1z", "s-c2x", "s-c2z", "c-ex", "c-ez", "c-angle"];

// Defaults — must match the `value=` attributes on the slider inputs in index.html.
const PATH_DEFAULTS = {
  shape:     "s",
  "s-c1x":   -0.575, "s-c1z":  0.9,
  "s-c2x":   -1.725, "s-c2z": -0.1,
  "c-ex":     1,     "c-ez":   1.5,
  "c-angle":  45,
};

function onPathShapeChanged() {
  const isC = pathShapeSel.value === "c";
  pathSlidersS.hidden = isC;
  pathSlidersC.hidden = !isC;
  setTargetPath(computeCurrentCurve());
}

function onPathRestoreClicked() {
  pathShapeSel.value = PATH_DEFAULTS.shape;
  for (const id of PATH_SLIDER_IDS) {
    document.getElementById(id).value = String(PATH_DEFAULTS[id]);
  }
  refreshAllReadouts();
  // onPathShapeChanged also calls setTargetPath, so this single call covers
  // toggling slider visibility AND pushing the new curve into both viewers.
  onPathShapeChanged();
}

function onPathSliderInput(e) {
  updateReadout(e.target.id);
  setTargetPath(computeCurrentCurve());
}

function updateReadout(sliderId) {
  const readout = document.querySelector(`[data-readout="${sliderId}"]`);
  if (!readout) return;
  const v = parseFloat(document.getElementById(sliderId).value);
  readout.textContent = sliderId === "c-angle" ? v.toFixed(0) : v.toFixed(2);
}

function refreshAllReadouts() {
  for (const id of PATH_SLIDER_IDS) updateReadout(id);
}

// Kick off the worker bootstrap.
setOutputState("busy", "Booting Pyodide…");
worker.postMessage({ type: "init" });

// ────────────────────────────────────────────────────────────────────────────
//  Run pipeline
// ────────────────────────────────────────────────────────────────────────────
async function onRunClicked() {
  setControlsEnabled(false);
  setOutputState("busy", "Planner running…");
  clearOutput();

  const plannerName = plannerSel.value;
  const interpSteps = Math.max(1, parseInt(interpInput.value, 10) || 20);
  const gamma       = parseFloat(gammaSlider.value);
  const shapeLibUrl = shapeLibSel.value;

  appLog(`Running planner=${plannerName}, interp_steps=${interpSteps}` +
         (plannerName === "Threshold Cluster" ? `, gamma=${gamma}` : "") +
         (plannerName !== "Direct Optimization" ? `, shape_lib=${shapeLibUrl}` : ""));

  const t0 = performance.now();
  try {
    const result = await workerCall("run", {
      plannerName,
      shapeLibUrl,
      interpSteps,
      gamma,
      waypointsJson: JSON.stringify(currentCurve),
    });
    const wallSecs = (performance.now() - t0) / 1000;
    appLog(`run complete in ${wallSecs.toFixed(2)} s — ${result.frames} frames`);

    resultsEl.hidden = false;
    setMetric("tip",        formatDev(result.tip_dev));
    setMetric("shape",      formatDev(result.shape_dev));
    setMetric("setup-time", `${result.setup_secs.toFixed(2)} s`);
    setMetric("plan-time",  `${result.plan_secs.toFixed(2)} s`);
    setMetric("frames",     `${result.frames}`);

    plannerViewer.loadResult(result);
    playbackEl.hidden        = false;
    frameSlider.disabled     = false;
    frameSlider.min          = "0";
    frameSlider.max          = String(result.frames - 1);
    frameSlider.value        = "0";
    frameReadout.textContent = `1 / ${result.frames}`;

    // Start playback automatically so the user immediately sees what was just
    // computed. They can pause / scrub with the controls.
    plannerViewer.play();
    playBtn.textContent = "⏸";

    setOutputState("ok", `Run finished in ${wallSecs.toFixed(2)} s.`);
  } catch (err) {
    console.error(err);
    appLog("ERROR:", err.message || String(err));
    setOutputState("err", "Run failed — see runtime log.");
  } finally {
    setControlsEnabled(true);
  }
}
