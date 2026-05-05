// viewer.js — Three.js scene, target-path viewer, and planner-result playback.
//
// Mirrors the matplotlib visualization in src/Visualizations/visualizer.py:
//   - blue robot backbone (thick line)
//   - red dots at segment endpoints
//   - green dot at the base, plus an SE(3) frame (R/G/B axes) at the base pose
//   - gray waypoints, orange = active (already passed), purple = current target tip
//   - orange tube along the target path
// Camera is OrbitControls (drag to rotate, scroll to zoom, right-drag to pan).

import * as THREE from "three";
import { OrbitControls }  from "three/addons/controls/OrbitControls.js";


// Bright, saturated palette so the viewer matches the legend swatches at a glance.
// Values are aligned with the legend SVG colors in docs/index.html and the
// .swatch--* CSS rules in docs/styles.css.
const COLORS = {
  background:        0xfafbfc,
  grid_major:        0x9aa0aa,
  grid_minor:        0xd2d6dc,
  robot_line:        0x2563eb, // bright blue
  endpoints:         0xdc2626, // bright red
  base_point:        0x16a34a, // bright green
  waypoints_idle:    0x9ca3af, // gray
  waypoints_active:  0xea580c, // bright orange
  target_tip:        0x9333ea, // bright purple
  target_path:       0xea580c, // orange tube along the target curve
  axis_x:            0xdc2626, // red
  axis_y:            0x16a34a, // green
  axis_z:            0x2563eb, // blue
};

const ROBOT_TUBE_RADIUS     = 0.04;  // robot backbone tube radius (world units)
const TARGET_TUBE_RADIUS    = 0.028;
const SE3_AXIS_LENGTH       = 0.45;  // base-frame arrow length
const SE3_AXIS_SHAFT_RADIUS = 0.012;
const PLAYBACK_FPS          = 30;


/**
 * Shared Three.js scene scaffolding: renderer, camera, OrbitControls, lighting,
 * floor grid, and an animation loop. Both PathViewer and PlannerViewer extend this.
 */
class BaseViewer {
  constructor(container, { showFloor = true } = {}) {
    this.container = container;
    this._initScene();
    if (showFloor) this._initFloor();
    this._initResize();

    this._render = this._render.bind(this);
    requestAnimationFrame(this._render);
  }

  _initScene() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight || 360;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.background);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 200);
    this.camera.position.set(5, 3.5, 5);
    this.camera.up.set(0, 0, 1);   // Z-up, matching the FTL coordinate frame

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 1.5);
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.55);
    dir.position.set(5, 5, 8);
    this.scene.add(dir);
  }

  _initFloor() {
    const grid = new THREE.GridHelper(8, 16, COLORS.grid_major, COLORS.grid_minor);
    grid.rotation.x = Math.PI / 2;     // -> XY plane
    grid.position.z = -3;              // robot length = 3, base sits at z=-3 by default
    this.scene.add(grid);
  }

  _initResize() {
    const ro = new ResizeObserver(() => {
      const w = this.container.clientWidth;
      const h = this.container.clientHeight || 360;
      if (w === 0 || h === 0) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h, false);
      this._notifyResolutionChange(w, h);
    });
    ro.observe(this.container);
  }

  _notifyResolutionChange(_w, _h) { /* subclasses override */ }

  _render() {
    requestAnimationFrame(this._render);
    this._tick();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _tick() { /* subclasses override */ }

  _frameOnPoints(points) {
    if (!points?.length) return;
    let cx = 0, cy = 0, cz = 0;
    for (const [x, y, z] of points) { cx += x; cy += y; cz += z; }
    cx /= points.length; cy /= points.length; cz /= points.length;
    this.controls.target.set(cx, cy, cz);
    this.controls.update();
  }
}


/**
 * Shared helpers for drawing the orange target curve + waypoint markers.
 * Both viewers use these.
 */
function buildTargetCurve(group, waypoints) {
  while (group.children.length) {
    const c = group.children.pop();
    c.geometry?.dispose?.();
    c.material?.dispose?.();
  }
  if (!waypoints || waypoints.length < 2) return;

  const curvePts = waypoints.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const curve    = new THREE.CatmullRomCurve3(curvePts, false, "centripetal");
  const tube     = new THREE.TubeGeometry(curve, Math.max(64, waypoints.length * 8),
                                          TARGET_TUBE_RADIUS, 12, false);
  const mesh     = new THREE.Mesh(
    tube,
    new THREE.MeshStandardMaterial({ color: COLORS.target_path, roughness: 0.55 }),
  );
  group.add(mesh);
}

function buildWaypointMarkers(scene, existing, waypoints) {
  for (const m of existing) {
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  const out = [];
  if (!waypoints) return out;
  const geom = new THREE.SphereGeometry(0.05, 14, 10);
  for (const [x, y, z] of waypoints) {
    const mat = new THREE.MeshStandardMaterial({ color: COLORS.waypoints_idle });
    const m = new THREE.Mesh(geom, mat);
    m.position.set(x, y, z);
    scene.add(m);
    out.push(m);
  }
  return out;
}


// ════════════════════════════════════════════════════════════════════════════
//  Path viewer (top-of-page, just the target curve)
// ════════════════════════════════════════════════════════════════════════════
export class PathViewer extends BaseViewer {
  constructor(container) {
    super(container);
    this.targetPathGroup = new THREE.Group();
    this.scene.add(this.targetPathGroup);
    this.waypointMeshes = [];
  }

  showTargetPath(waypoints) {
    buildTargetCurve(this.targetPathGroup, waypoints);
    this.waypointMeshes = buildWaypointMarkers(this.scene, this.waypointMeshes, waypoints);
    this._frameOnPoints(waypoints);
  }
}


// ════════════════════════════════════════════════════════════════════════════
//  Planner viewer (bottom-of-page, full motion plan playback)
// ════════════════════════════════════════════════════════════════════════════
export class PlannerViewer extends BaseViewer {
  constructor(container) {
    super(container);
    this.frames    = null;
    this.waypoints = null;
    this.frameIdx  = 0;
    this.playing   = false;
    this._lastFrameTimeMs = 0;
    this.frameChangeListener = null;

    this.targetPathGroup = new THREE.Group();
    this.scene.add(this.targetPathGroup);
    this.waypointMeshes = [];

    this.robotGroup = new THREE.Group();
    this.scene.add(this.robotGroup);

    this._initRobotTube();
    this._initBaseFrame();
    this._initTargetTipMarker();

    this.endpointMarkers = [];
  }

  // ----- scene-object setup ------------------------------------------------

  _initRobotTube() {
    // The robot backbone is rendered as a TubeGeometry along the spline
    // through shape_points. The geometry is rebuilt per frame in setFrame();
    // the material is shared across frames.
    // We push the emissive component up so the rendered color reads as a
    // saturated solid blue regardless of lighting — this is what makes it
    // match the SVG legend swatches.
    this.robotMaterial = brightSolidMaterial(COLORS.robot_line);
    this.robotMesh = null;   // populated on first setFrame
  }

  _initBaseFrame() {
    // Three colored arrows (R=X, G=Y, B=Z) anchored at the base position.
    // Wrapped in an Object3D so a single matrix update positions+rotates all of them.
    this.baseFrame = new THREE.Object3D();
    this.scene.add(this.baseFrame);

    const xArrow = makeArrow(COLORS.axis_x, SE3_AXIS_LENGTH, SE3_AXIS_SHAFT_RADIUS);
    xArrow.rotation.z = -Math.PI / 2; // arrow points +Y by default; rotate to +X
    const yArrow = makeArrow(COLORS.axis_y, SE3_AXIS_LENGTH, SE3_AXIS_SHAFT_RADIUS);
    // already +Y
    const zArrow = makeArrow(COLORS.axis_z, SE3_AXIS_LENGTH, SE3_AXIS_SHAFT_RADIUS);
    zArrow.rotation.x =  Math.PI / 2; // rotate to +Z

    this.baseFrame.add(xArrow, yArrow, zArrow);

    // Green base-position marker on top of the SE(3) origin.
    this.baseMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 16, 12),
      brightSolidMaterial(COLORS.base_point),
    );
    this.baseFrame.add(this.baseMarker);

    this.baseFrame.visible = false;
  }

  _initTargetTipMarker() {
    this.targetTipMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 16, 12),
      brightSolidMaterial(COLORS.target_tip),
    );
    this.scene.add(this.targetTipMarker);
    this.targetTipMarker.visible = false;
  }

  // ----- public api --------------------------------------------------------

  showTargetPath(waypoints) {
    this.waypoints = waypoints;
    buildTargetCurve(this.targetPathGroup, waypoints);
    this.waypointMeshes = buildWaypointMarkers(this.scene, this.waypointMeshes, waypoints);
    this._frameOnPoints(waypoints);
  }

  /**
   * Render the robot at its default starting pose (zero Clarke coords) with no
   * playback. Called once after Pyodide is ready, so the bottom viewer doesn't
   * sit empty before the user kicks off a run. Does NOT enable playback.
   */
  showDefaultRobot(state) {
    this._rebuildEndpointMarkers(state.endpoints.length);
    this._setRobotMeshFromShape(state.shape_points);

    // Endpoint markers.
    for (let k = 0; k < this.endpointMarkers.length; k++) {
      const p = state.endpoints[k];
      this.endpointMarkers[k].position.set(p[0], p[1], p[2]);
    }

    // Base SE(3).
    this._setBaseFrame(state.base_position, state.base_orientation);
    this.baseFrame.visible       = true;
    this.targetTipMarker.visible = false;
  }

  loadResult(result) {
    this.waypoints = result.waypoints;
    this.frames    = result.history;
    this.frameIdx  = 0;
    this.playing   = false;
    this._lastFrameTimeMs = 0;

    buildTargetCurve(this.targetPathGroup, this.waypoints);
    this.waypointMeshes = buildWaypointMarkers(this.scene, this.waypointMeshes, this.waypoints);
    this._rebuildEndpointMarkers(result.history[0].endpoints.length);

    this.targetTipMarker.visible = true;
    this.baseFrame.visible       = true;

    this.setFrame(0);
    this._frameOnPoints(this.waypoints);
  }

  setFrame(i) {
    if (!this.frames) return;
    this.frameIdx = Math.max(0, Math.min(this.frames.length - 1, i | 0));
    const f = this.frames[this.frameIdx];

    this._setRobotMeshFromShape(f.shape_points);

    // Endpoints (red dots).
    for (let k = 0; k < this.endpointMarkers.length; k++) {
      const m = this.endpointMarkers[k];
      const p = f.endpoints[k];
      m.position.set(p[0], p[1], p[2]);
    }

    this._setBaseFrame(f.base_position, f.base_orientation);

    // Target tip (purple).
    const t = f.target_tip;
    this.targetTipMarker.position.set(t[0], t[1], t[2]);

    // Waypoint coloring: gray until reached, orange after.
    if (this.waypoints && this.waypointMeshes.length === this.waypoints.length) {
      const reached = this._waypointsReached(t);
      for (let k = 0; k < this.waypointMeshes.length; k++) {
        this.waypointMeshes[k].material.color.setHex(
          k < reached ? COLORS.waypoints_active : COLORS.waypoints_idle,
        );
      }
    }

    if (this.frameChangeListener) this.frameChangeListener(this.frameIdx);
  }

  play()  { if (this.frames) { this.playing = true; this._lastFrameTimeMs = performance.now(); } }
  pause() { this.playing = false; }
  togglePlay() {
    if (!this.frames) return;
    if (this.frameIdx >= this.frames.length - 1) this.frameIdx = 0;
    if (this.playing) this.pause(); else this.play();
  }

  // ----- subclass overrides -----------------------------------------------

  _tick() {
    if (this.playing && this.frames) {
      const now = performance.now();
      const dt  = now - this._lastFrameTimeMs;
      if (dt >= 1000 / PLAYBACK_FPS) {
        this._lastFrameTimeMs = now;
        this.setFrame((this.frameIdx + 1) % this.frames.length);
      }
    }
  }

  // ----- private helpers --------------------------------------------------

  /** Replace the robot tube geometry with a fresh one built from shape_points. */
  _setRobotMeshFromShape(shapePoints) {
    const curvePts = new Array(shapePoints.length);
    for (let k = 0; k < shapePoints.length; k++) {
      curvePts[k] = new THREE.Vector3(shapePoints[k][0], shapePoints[k][1], shapePoints[k][2]);
    }
    const curve = new THREE.CatmullRomCurve3(curvePts, false, "centripetal");
    const tube  = new THREE.TubeGeometry(
      curve,
      Math.max(64, shapePoints.length * 2),
      ROBOT_TUBE_RADIUS,
      12,
      false,
    );
    if (this.robotMesh) {
      this.robotGroup.remove(this.robotMesh);
      this.robotMesh.geometry.dispose();
    }
    this.robotMesh = new THREE.Mesh(tube, this.robotMaterial);
    this.robotGroup.add(this.robotMesh);
  }

  /** Position + orient the base SE(3) frame from a 3-vector and 3x3 rotation. */
  _setBaseFrame(position, orientation3x3) {
    const b = position;
    const R = orientation3x3;
    this.baseFrame.matrix.set(
      R[0][0], R[0][1], R[0][2], b[0],
      R[1][0], R[1][1], R[1][2], b[1],
      R[2][0], R[2][1], R[2][2], b[2],
      0,       0,       0,       1,
    );
    this.baseFrame.matrixAutoUpdate = false;
    this.baseFrame.matrixWorldNeedsUpdate = true;
  }

  _rebuildEndpointMarkers(n) {
    for (const m of this.endpointMarkers) {
      this.robotGroup.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this.endpointMarkers = [];
    const baseGeom = new THREE.SphereGeometry(0.06, 14, 10);
    for (let i = 0; i < n; i++) {
      const m = new THREE.Mesh(baseGeom.clone(), brightSolidMaterial(COLORS.endpoints));
      this.robotGroup.add(m);
      this.endpointMarkers.push(m);
    }
  }

  _waypointsReached(currentTarget) {
    if (!this.waypoints) return 0;
    let bestIdx = 0;
    let bestD2  = Infinity;
    for (let i = 0; i < this.waypoints.length; i++) {
      const dx = this.waypoints[i][0] - currentTarget[0];
      const dy = this.waypoints[i][1] - currentTarget[1];
      const dz = this.waypoints[i][2] - currentTarget[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
    }
    return bestIdx;
  }
}


/**
 * Material that renders a saturated solid color regardless of lighting, while
 * still picking up some shading for 3D depth cues. We use a strong emissive
 * contribution so the rendered hue matches the same hex code in the SVG legend.
 */
function brightSolidMaterial(colorHex) {
  return new THREE.MeshStandardMaterial({
    color:             colorHex,
    emissive:          colorHex,
    emissiveIntensity: 0.55,
    roughness:         0.55,
    metalness:         0.0,
  });
}


// Helper: a single colored arrow as a thin cylinder + cone, pointing along +Y by default.
function makeArrow(colorHex, length, shaftRadius) {
  const headLength  = length * 0.22;
  const shaftLength = length - headLength;
  const headRadius  = shaftRadius * 2.2;

  const mat = new THREE.MeshStandardMaterial({
    color: colorHex, roughness: 0.4, metalness: 0.05,
  });

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 14),
    mat,
  );
  shaft.position.y = shaftLength / 2;

  const head = new THREE.Mesh(
    new THREE.ConeGeometry(headRadius, headLength, 18),
    mat,
  );
  head.position.y = shaftLength + headLength / 2;

  const g = new THREE.Group();
  g.add(shaft, head);
  return g;
}
