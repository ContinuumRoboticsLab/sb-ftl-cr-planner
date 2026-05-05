// worker.js — classic Web Worker that owns Pyodide, the planner source bundle,
// and any pre-generated shape libraries.
//
// Runs Pyodide on a separate thread so the main thread stays responsive
// (spinners spin, dropdowns work, stdout from the planner streams live to the
// page via postMessage instead of dumping all at once at the end).
//
// Message protocol:
//   main -> worker: { type: 'init' }
//                   { type: 'run', id, plannerName, shapeLibUrl, interpSteps,
//                                  gamma, waypointsJson }
//
//   worker -> main: { type: 'stage', step, title }
//                   { type: 'stageProgress', step, fraction }
//                   { type: 'stageDone', step }
//                   { type: 'ready', defaultRobot }
//                   { type: 'stdout', text }       // single line of Python stdout
//                   { type: 'log',    text }       // worker-level status line
//                   { type: 'response', id, payload }
//                   { type: 'error',    id?, message }

importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js");

const PYODIDE_BASE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
const SRC_BUNDLE_URL   = "assets/src_bundle.zip";

let pyodide = null;
const cachedShapeLibPaths = new Set();

// ---------------------------------------------------------------------------
// Python helpers — same code that used to live in app.js. Defined once after
// the bundle is unpacked.
// ---------------------------------------------------------------------------
const PYTHON_HELPERS = `
import json
import time
import warnings

# Pyodide-specific noise: scikit-learn's transitive dep threadpoolctl calls a
# deprecated Pyodide internal (JsProxy.as_object_map) on import. We silence the
# warning so it doesn't pollute the planner's live output box. Native runs are
# unaffected (the warning only fires under Emscripten Python).
warnings.filterwarnings("ignore", message=".*as_object_map.*")

import numpy as np

import sys
if "/srcroot" not in sys.path:
    sys.path.insert(0, "/srcroot")

from src.MasterClass import GeneralPathFollower
from src.RobotModels.ConstantCurvatureModel import ConstantCurvature
from src.MotionPlanners.OptimizationBased.DirectOptimization import DirectOptimization

# Robot config matches run_example.py / tools/generate_shape_libs.py.
ROBOT_KWARGS = dict(
    num_segments=3,
    segment_lengths=[1, 1, 1],
    tendon_offset=[0.2, 0.2, 0.2],
    points_resolution=0.05,
)


def _serialize_history(history):
    out = []
    for h in history:
        out.append({
            "shape_points":     h["shape_points"].tolist(),
            "endpoints":        h["endpoints"].tolist(),
            "base_position":    h["base_position"].tolist(),
            "base_orientation": h["base_orientation"].tolist(),
            "target_tip":       h["target_tip"].tolist(),
        })
    return out


def run_demo(planner_name: str,
             shape_lib_path: str,
             waypoints_json: str,
             interp_steps: int = 20,
             gamma: float = 2.5) -> dict:
    robot = ConstantCurvature(**ROBOT_KWARGS)
    waypoints = np.array(json.loads(waypoints_json))
    num_waypoints = len(waypoints)

    general_follower = GeneralPathFollower(robot)
    samplers = general_follower.get_sampling_methods_by_name([planner_name])
    if not samplers:
        raise ValueError(f"Unknown planner: {planner_name!r}")
    sampler = samplers[0]

    sampler_kwargs = dict(
        base_stability_weight=0.3,
        base_stability_weight_rot=0.3,
        verbose=False,
    )
    if planner_name == "Direct Optimization":
        sampler_kwargs["num_samples"] = 1
    else:
        sampler_kwargs["num_samples"] = 1
        sampler_kwargs["custom_shape_lib_path"] = shape_lib_path

    if planner_name == "Threshold Cluster":
        sampler_kwargs["similarity_threshold"] = float(gamma)

    setup_t0 = time.perf_counter()
    follower = sampler(**sampler_kwargs)
    setup_secs = time.perf_counter() - setup_t0

    plan_t0 = time.perf_counter()
    history = follower.follow_path(waypoints, verbose=True)

    if isinstance(follower, DirectOptimization):
        interp_method = general_follower.interpolate_optimization
    else:
        interp_method = general_follower.interpolate_mp

    interp_history, interp_waypoints = interp_method(
        history,
        steps_per_waypoint=interp_steps,
        enable_optimization=True,
        verbose=True,
    )
    plan_secs = time.perf_counter() - plan_t0

    skip_first_n = 2 * interp_steps
    tip_dev = general_follower.compute_tip_deviation(
        interp_history[skip_first_n:], interp_waypoints[skip_first_n:]
    )
    shape_dev = general_follower.compute_shape_deviation_closest(
        interp_history[skip_first_n:],
        interp_waypoints[skip_first_n:],
        plot_deviation=False,
        num_waypoints=num_waypoints,
    )

    return {
        "planner":    planner_name,
        "frames":     len(interp_history),
        "setup_secs": setup_secs,
        "plan_secs":  plan_secs,
        "tip_dev":    {k: float(v) for k, v in tip_dev.items()},
        "shape_dev":  {k: float(v) for k, v in shape_dev.items()},
        "waypoints":  waypoints.tolist(),
        "history":    _serialize_history(interp_history),
    }


def default_robot_state():
    robot = ConstantCurvature(**ROBOT_KWARGS)
    clark = np.zeros(robot.num_segments * 2)
    endpoints, shape_points, _ = robot.forward_kinematics(clark)
    base_position    = shape_points[0]
    base_orientation = np.eye(3)
    return {
        "shape_points":     shape_points.tolist(),
        "endpoints":        endpoints.tolist(),
        "base_position":    base_position.tolist(),
        "base_orientation": base_orientation.tolist(),
    }
`;


// ---------------------------------------------------------------------------
// Wire stdout/stderr through the message channel. The Pyodide stdout callback
// fires synchronously from inside the worker thread; postMessage queues a job
// on the main thread's event loop, so output appears live in the UI even while
// Python is mid-run on the worker.
// ---------------------------------------------------------------------------
function pyOut(text) {
  text = String(text).replace(/\n$/, "");
  if (text.length === 0) return;
  self.postMessage({ type: "stdout", text });
}

function notify(type, extra = {}) {
  self.postMessage({ type, ...extra });
}

function workerLog(text) {
  notify("log", { text });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
async function init() {
  // Stage 1: Pyodide runtime
  notify("stage", { step: "pyodide", title: "Initializing Python runtime…" });
  pyodide = await loadPyodide({
    indexURL: PYODIDE_BASE_URL,
    stdout: pyOut,
    stderr: pyOut,
  });
  workerLog("Pyodide runtime loaded");
  notify("stageDone", { step: "pyodide" });

  // Stage 2: scientific packages
  notify("stage", { step: "packages", title: "Loading numpy / scipy / scikit-learn (heavy)…" });
  let pkgsLoaded = 0;
  await pyodide.loadPackage(["numpy", "scipy", "scikit-learn"], {
    messageCallback: (msg) => {
      workerLog("[pyodide pkg] " + msg);
      if (/^Loaded /.test(msg)) {
        pkgsLoaded++;
        notify("stageProgress", { step: "packages", fraction: Math.min(1, pkgsLoaded / 6) });
      }
    },
    errorCallback: (msg) => workerLog("[pyodide pkg err] " + msg),
  });
  notify("stageDone", { step: "packages" });

  // Stage 3: source bundle
  notify("stage", { step: "bundle", title: "Fetching planner source bundle…" });
  workerLog(`Fetching ${SRC_BUNDLE_URL}`);
  const bundleResp = await fetch(SRC_BUNDLE_URL);
  if (!bundleResp.ok) throw new Error(`bundle fetch failed: HTTP ${bundleResp.status}`);
  const bundleBytes = await bundleResp.arrayBuffer();
  workerLog(`Unpacking ${(bundleBytes.byteLength / 1024).toFixed(1)} KB into Pyodide FS at /srcroot`);
  pyodide.FS.mkdir("/srcroot");
  pyodide.unpackArchive(bundleBytes, "zip", { extractDir: "/srcroot" });
  notify("stageDone", { step: "bundle" });

  // Stage 4: import planner + grab the robot's default starting pose
  // (the target curve is now computed entirely on the main thread).
  notify("stage", { step: "ready", title: "Importing planner modules…" });
  pyodide.runPython(PYTHON_HELPERS);

  const robotProxy = pyodide.globals.get("default_robot_state")();
  const defaultRobot = robotProxy.toJs({ dict_converter: Object.fromEntries });
  robotProxy.destroy();

  notify("stageDone", { step: "ready" });
  workerLog("planner ready");
  notify("ready", { defaultRobot });
}

// ---------------------------------------------------------------------------
// Run pipeline
// ---------------------------------------------------------------------------
async function ensureShapeLib(url) {
  const fsPath = "/" + url.split("/").pop();
  if (cachedShapeLibPaths.has(fsPath)) return fsPath;

  workerLog(`Fetching ${url}`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`shape lib fetch failed: HTTP ${resp.status}`);
  const buf = await resp.arrayBuffer();
  pyodide.FS.writeFile(fsPath, new Uint8Array(buf));
  cachedShapeLibPaths.add(fsPath);
  workerLog(`shape lib written to ${fsPath} (${(buf.byteLength / 1_048_576).toFixed(1)} MB)`);
  return fsPath;
}

async function runPlanner(params) {
  const { plannerName, shapeLibUrl, interpSteps, gamma, waypointsJson } = params;

  let fsPath = "";
  if (plannerName !== "Direct Optimization") {
    fsPath = await ensureShapeLib(shapeLibUrl);
  }

  const runDemo = pyodide.globals.get("run_demo");
  const pyResult = runDemo.callKwargs({
    planner_name:   plannerName,
    shape_lib_path: fsPath,
    waypoints_json: waypointsJson,
    interp_steps:   interpSteps,
    gamma:          gamma,
  });
  const result = pyResult.toJs({ dict_converter: Object.fromEntries });
  pyResult.destroy();
  runDemo.destroy();
  return result;
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------
self.onmessage = async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "init") {
      await init();
      return;
    }
    if (type === "run") {
      const result = await runPlanner(e.data);
      self.postMessage({ type: "response", id, payload: result });
      return;
    }
    throw new Error(`unknown message type: ${type}`);
  } catch (err) {
    console.error("[worker]", err);
    self.postMessage({
      type:    "error",
      id,
      message: err && err.message ? err.message : String(err),
    });
  }
};
