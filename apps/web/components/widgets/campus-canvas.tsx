"use client";

import {
  Canvas,
  useFrame,
  useLoader,
  useThree,
  type ThreeEvent,
} from "@react-three/fiber";
import {
  Component,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  spreadFlags,
  type BeadKind,
  type RoadTraffic,
  type StageId,
} from "@/lib/campus-status";
import type { Campus3D, ComponentModels } from "@/lib/overview-scene";
import type { Tone } from "@/lib/tone";

/** Resolved scene colours (hex), read from the `--aw-scene-*` and tone tokens. */
export interface ScenePalette {
  theme: "light" | "dark";
  structure: string;
  shade: string;
  recess: string;
  glass: string;
  road: string;
  ground: string;
  bead: string;
  brand: string;
  tones: Record<Tone, string>;
}

export interface CampusCanvasProps {
  manifest: Campus3D;
  stageTones: Record<StageId, Tone | null>;
  traffic: Record<string, RoadTraffic>;
  /** Hovered or selected stage, drawn raised and brighter. */
  active: StageId | null;
  /** The stage the camera zooms to and opens as a component model, if any. */
  focus: StageId | null;
  /**
   * The canvas span (NDC, -1 left to 1 right) left free for the focused model,
   * clear of the page's own panels; the model is centred and fitted in it.
   */
  focusRegion: { x0: number; x1: number };
  components: ComponentModels;
  animate: boolean;
  palette: ScenePalette;
  /** Called every frame with each stage anchor's position in canvas pixels. */
  onProject: (stage: StageId, x: number, y: number) => void;
  onHover: (stage: StageId | null) => void;
  onSelect: (stage: StageId) => void;
  /** A click that hit nothing in the scene. */
  onMiss: () => void;
  onReady: () => void;
  onError: () => void;
}

const BEAD_RADIUS_M = 0.11;
const BEAD_SPACING_M = 0.42;
const BEAD_HEIGHT_M = 0.16;
const TRAFFIC_SPEED_M_PER_S = 0.55;
const ACTIVE_LIFT_M = 0.12;
/** Focus zoom: the model fills this share of the free region, within these limits. */
const FOCUS_FILL = 0.72;
const FOCUS_ZOOM_LIMITS = { min: 1, max: 6 };
/** Below centre, clear of the page title that floats over the top of the scene. */
const FOCUS_Y_NDC = -0.18;
/** Height of the focus point above the ground, metres (about mid-model). */
const FOCUS_HEIGHT_M = 0.8;
/**
 * Screen size of a square footprint of side `s` under the 45° heading and 30°
 * elevation: about 1.41 s wide and 0.71 s + model height tall.
 */
const ISO_WIDTH_PER_SIDE = 1.42;
const ISO_HEIGHT_PER_SIDE = 1.05;
/** Camera easing rate (1/s): about 90% of the way in half a second. */
const CAMERA_EASE_PER_S = 5;
const REVEAL_S = 0.55;
/** Rate (1/s) at which the other buildings fold into the ground and rise back. */
const FOLD_PER_S = 7;
const PACKET_RADIUS_M = 0.06;
const PACKET_SPACING_M = 0.32;
const PACKET_SPEED_M_PER_S = 0.7;
const HALO_MARGIN_M = 0.35;
/** Share of the halo texture covered by its rounded rectangle (176 of 256 px). */
const HALO_FILL = 176 / 256;
const GROUND_DISC_M: [number, number] = [21, 15];
const SHADOW_OPACITY = 0.22;
const MAX_FRAME_DELTA_S = 0.05;

const STRUCTURE_MATERIALS: Record<string, keyof ScenePalette> = {
  aw_structure: "structure",
  aw_structure_shade: "shade",
  aw_recess: "recess",
  aw_glass: "glass",
  aw_road: "road",
};

function beadColor(kind: BeadKind, palette: ScenePalette) {
  switch (kind) {
    case "excess":
      return palette.tones.caution;
    case "failed":
      return palette.tones.danger;
    case "passed":
      return palette.tones.success;
    case "normal":
      return palette.bead;
  }
}

/** Stage glow colour: its tone, brand green for the informational gateway, grey without data. */
function glowColor(tone: Tone | null, palette: ScenePalette) {
  if (tone === null) return palette.recess;
  return tone === "info" ? palette.brand : palette.tones[tone];
}

// ------------------------------------------------------------------ camera

function CameraRig({
  manifest,
  focus,
  focusBox,
  focusRegion,
  animate,
}: {
  manifest: Campus3D;
  focus: StageId | null;
  /** The focused building's bounds, which the zoom fits. */
  focusBox: THREE.Box3 | undefined;
  focusRegion: { x0: number; x1: number };
  animate: boolean;
}) {
  // Three objects are read through `get()` inside effects and frames: they are
  // mutable engine state, not React values, and must not change during render.
  const get = useThree((state) => state.get);
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const goal = useRef({ position: new THREE.Vector3(), zoom: 1 });

  // Match the Blender camera, then "contain" it: the designed frustum always
  // fits, and a frame wider than the render gains room on the left, where the
  // page floats its callouts. The fallback render is right-aligned the same way.
  useEffect(() => {
    const camera = get().camera as THREE.OrthographicCamera;
    const { orthoWidth, shift, aspect } = manifest.camera;
    const designHeight = orthoWidth / aspect;
    const canvasAspect = width / height;
    const [frustumWidth, frustumHeight] =
      canvasAspect >= aspect
        ? [designHeight * canvasAspect, designHeight]
        : [orthoWidth, orthoWidth / canvasAspect];
    const centerX = shift[0] * orthoWidth;
    const centerY = shift[1] * orthoWidth;
    camera.right = centerX + orthoWidth / 2;
    camera.left = camera.right - frustumWidth;
    camera.top = centerY + frustumHeight / 2;
    camera.bottom = centerY - frustumHeight / 2;
    camera.near = 0.1;
    camera.far = 200;
    camera.updateProjectionMatrix();
    get().invalidate();
  }, [get, manifest, width, height]);

  // The view never rotates: the camera keeps its orientation and only moves
  // in its own plane and zooms, toward the focused stage and back.
  useEffect(() => {
    const camera = get().camera as THREE.OrthographicCamera;
    camera.position.set(...manifest.camera.position);
    camera.up.set(0, 1, 0);
    camera.lookAt(...manifest.camera.target);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    goal.current.position.copy(camera.position);
  }, [get, manifest]);

  useEffect(() => {
    const camera = get().camera as THREE.OrthographicCamera;
    const base = new THREE.Vector3(...manifest.camera.position);
    if (focus === null || !focusBox) {
      goal.current = { position: base, zoom: 1 };
    } else {
      const center = focusBox.getCenter(new THREE.Vector3());
      const size = focusBox.getSize(new THREE.Vector3());
      const side = Math.max(size.x, size.z);
      const point = new THREE.Vector3(center.x, FOCUS_HEIGHT_M, center.z);
      const frustumWidth = camera.right - camera.left;
      const frustumHeight = camera.top - camera.bottom;
      const regionShare = (focusRegion.x1 - focusRegion.x0) / 2;
      const zoom = THREE.MathUtils.clamp(
        Math.min(
          (FOCUS_FILL * frustumWidth * regionShare) /
            (ISO_WIDTH_PER_SIDE * side),
          (FOCUS_FILL * frustumHeight) / (ISO_HEIGHT_PER_SIDE * side),
        ),
        FOCUS_ZOOM_LIMITS.min,
        FOCUS_ZOOM_LIMITS.max,
      );
      // Ortho zoom scales around the frustum centre, so solve for the camera
      // offset that puts `point` at the region's centre on screen.
      const focusX = (focusRegion.x0 + focusRegion.x1) / 2;
      const halfWidth = frustumWidth / (2 * zoom);
      const halfHeight = frustumHeight / (2 * zoom);
      const centerX = (camera.left + camera.right) / 2;
      const centerY = (camera.top + camera.bottom) / 2;
      const right = new THREE.Vector3().setFromMatrixColumn(
        camera.matrixWorld,
        0,
      );
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
      const position = base
        .clone()
        .add(point.sub(new THREE.Vector3(...manifest.camera.target)))
        .addScaledVector(right, -focusX * halfWidth - centerX)
        .addScaledVector(up, -FOCUS_Y_NDC * halfHeight - centerY);
      goal.current = { position, zoom };
    }
    if (!animate) {
      camera.position.copy(goal.current.position);
      camera.zoom = goal.current.zoom;
      camera.updateProjectionMatrix();
    }
    get().invalidate();
  }, [get, manifest, focus, focusBox, focusRegion, animate, width, height]);

  useFrame((_, delta) => {
    const camera = get().camera as THREE.OrthographicCamera;
    const { position, zoom } = goal.current;
    const distance =
      camera.position.distanceTo(position) + Math.abs(camera.zoom - zoom);
    if (distance < 1e-3) return;
    const k =
      1 - Math.exp(-Math.min(delta, MAX_FRAME_DELTA_S) * CAMERA_EASE_PER_S);
    camera.position.lerp(position, k);
    camera.zoom += (zoom - camera.zoom) * k;
    camera.updateProjectionMatrix();
    get().invalidate();
  });
  return null;
}

// ------------------------------------------------------------------ lighting

function Lighting() {
  const get = useThree((state) => state.get);

  useEffect(() => {
    const { gl, scene } = get();
    const pmrem = new THREE.PMREMGenerator(gl);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = environment;
    scene.environmentIntensity = 0.55;
    return () => {
      scene.environment = null;
      environment.dispose();
      pmrem.dispose();
    };
  }, [get]);

  const sun = useMemo(() => {
    const light = new THREE.DirectionalLight("#ffffff", 2.2);
    light.position.set(-6, 16, 9);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    light.shadow.bias = -0.0004;
    light.shadow.normalBias = 0.02;
    Object.assign(light.shadow.camera, {
      left: -16,
      right: 16,
      top: 16,
      bottom: -16,
      near: 1,
      far: 50,
    });
    light.target.position.set(1.5, 0, -1);
    return light;
  }, []);

  return (
    <>
      <primitive object={sun} />
      <primitive object={sun.target} />
      <hemisphereLight args={["#ffffff", "#c9d2cc", 0.6]} />
    </>
  );
}

/** Soft alpha masks: a radial fade for the ground, a glowing outline for stage halos. */
function softTexture(kind: "radial" | "halo") {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext("2d")!;
  if (kind === "radial") {
    const gradient = ctx.createRadialGradient(128, 128, 20, 128, 128, 128);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.6, "rgba(255,255,255,0.6)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);
  } else {
    // A glowing outline, so the halo rings the podium instead of tinting the ground.
    ctx.shadowColor = "white";
    ctx.shadowBlur = 14;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.roundRect(40, 40, 176, 176, 24);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  return texture;
}

/**
 * A shadow catcher (invisible except where shadows fall) and a faint disc that
 * fades out well inside the frame, so the campus sits on the page with no edge.
 */
function Ground({
  color,
  center,
}: {
  color: string;
  center: [number, number, number];
}) {
  const alphaMap = useMemo(() => softTexture("radial"), []);
  useEffect(() => () => alphaMap.dispose(), [alphaMap]);
  return (
    <>
      <mesh rotation-x={-Math.PI / 2} position={[center[0], -0.012, center[2]]}>
        <planeGeometry args={GROUND_DISC_M} />
        <meshBasicMaterial
          color={color}
          alphaMap={alphaMap}
          transparent
          opacity={0.7}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <mesh
        rotation-x={-Math.PI / 2}
        position={[center[0], -0.005, center[2]]}
        receiveShadow
      >
        <planeGeometry args={[40, 40]} />
        <shadowMaterial
          transparent
          opacity={SHADOW_OPACITY}
          depthWrite={false}
        />
      </mesh>
    </>
  );
}

// ------------------------------------------------------------------ campus

type StageNodes = Partial<
  Record<
    StageId,
    {
      group: THREE.Object3D;
      glow: THREE.MeshStandardMaterial[];
      box: THREE.Box3;
    }
  >
>;

function useCampusModel(url: string, palette: ScenePalette) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf]);

  const stages = useMemo(() => {
    const found: StageNodes = {};
    scene.updateMatrixWorld(true);
    scene.traverse((node) => {
      const id = node.userData.aw_stage as StageId | undefined;
      if (!id) return;
      const glow: THREE.MeshStandardMaterial[] = [];
      node.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const material = child.material as THREE.MeshStandardMaterial;
        if (material.name.startsWith("aw_glow_") && !glow.includes(material))
          glow.push(material);
      });
      found[id] = {
        group: node,
        glow,
        box: new THREE.Box3().setFromObject(node),
      };
    });
    return found;
  }, [scene]);

  // Theme colours: structure materials take the scene tokens.
  useEffect(() => {
    scene.traverse((node) => {
      if (!(node instanceof THREE.Mesh)) return;
      node.castShadow = true;
      node.receiveShadow = true;
      const material = node.material as THREE.MeshStandardMaterial;
      const key = STRUCTURE_MATERIALS[material.name];
      if (key) material.color.set(palette[key] as string);
      if (material.name === "aw_glass") {
        material.roughness = 0.15;
        material.metalness = 0.3;
      }
    });
  }, [scene, palette]);

  return { scene, stages };
}

function Halo({
  box,
  color,
  active,
  animate,
}: {
  box: THREE.Box3;
  color: string;
  active: boolean;
  animate: boolean;
}) {
  const texture = useMemo(() => softTexture("halo"), []);
  const material = useRef<THREE.MeshBasicMaterial>(null);
  useEffect(() => () => texture.dispose(), [texture]);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const base = active ? 0.9 : 0.55;

  useFrame(({ clock }) => {
    if (!material.current) return;
    material.current.opacity = animate
      ? base + 0.08 * Math.sin(clock.elapsedTime * 2.2)
      : base;
  });

  return (
    <mesh
      rotation-x={-Math.PI / 2}
      position={[center.x, 0.012, center.z]}
      renderOrder={1}
    >
      {/* The texture's rectangle fills HALO_FILL of the plane; scale up so it matches the footprint. */}
      <planeGeometry
        args={[
          (size.x + HALO_MARGIN_M) / HALO_FILL,
          (size.z + HALO_MARGIN_M) / HALO_FILL,
        ]}
      />
      <meshBasicMaterial
        ref={material}
        color={color}
        alphaMap={texture}
        transparent
        depthWrite={false}
        toneMapped={false}
        opacity={base}
      />
    </mesh>
  );
}

function Campus({
  manifest,
  palette,
  stageTones,
  active,
  animate,
  onHover,
  onSelect,
  onReady,
  hidden,
  focus,
  onStages,
}: Pick<
  CampusCanvasProps,
  | "manifest"
  | "palette"
  | "stageTones"
  | "active"
  | "animate"
  | "onHover"
  | "onSelect"
  | "onReady"
> & {
  /** The building currently replaced by its component model. */
  hidden: StageId | null;
  /** While a stage is open, the rest of the campus folds away. */
  focus: StageId | null;
  onStages: (boxes: Partial<Record<StageId, THREE.Box3>>) => void;
}) {
  const { scene, stages } = useCampusModel(manifest.model, palette);
  const invalidate = useThree((state) => state.invalidate);
  // Roads are the scene's direct meshes (stages are groups); they hide while a stage is open.
  const showRoads = useRef<((visible: boolean) => void) | null>(null);
  useEffect(() => {
    const roads = scene.children.filter((child) => !child.userData.aw_stage);
    showRoads.current = (visible) => {
      for (const road of roads) road.visible = visible;
    };
  }, [scene]);

  useEffect(() => {
    onStages(
      Object.fromEntries(
        Object.entries(stages).map(([id, nodes]) => [id, nodes!.box]),
      ),
    );
  }, [stages, onStages]);

  useEffect(() => {
    for (const [id, nodes] of Object.entries(stages) as [
      StageId,
      NonNullable<StageNodes[StageId]>,
    ][])
      if (id === hidden) nodes.group.visible = false;
    showRoads.current?.(focus === null);
    invalidate();
  }, [stages, hidden, focus, invalidate]);

  useEffect(() => {
    for (const [id, nodes] of Object.entries(stages) as [
      StageId,
      NonNullable<StageNodes[StageId]>,
    ][]) {
      const color = glowColor(stageTones[id], palette);
      for (const material of nodes.glow) {
        material.color.set(color);
        material.emissive.set(color);
        material.toneMapped = false;
      }
    }
    invalidate();
  }, [stages, stageTones, palette, invalidate]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => onReady());
    return () => cancelAnimationFrame(frame);
  }, [onReady]);

  useFrame(({ clock }, delta) => {
    const step = Math.min(delta, MAX_FRAME_DELTA_S) * 10;
    for (const [id, nodes] of Object.entries(stages) as [
      StageId,
      NonNullable<StageNodes[StageId]>,
    ][]) {
      const lifted = id === active ? ACTIVE_LIFT_M : 0;
      nodes.group.position.y +=
        (lifted - nodes.group.position.y) * Math.min(1, step);
      if (id !== hidden) {
        // Fold the other buildings into the ground while a stage is open.
        const height = focus !== null && id !== focus ? 0.001 : 1;
        const k = animate
          ? 1 - Math.exp(-Math.min(delta, MAX_FRAME_DELTA_S) * FOLD_PER_S)
          : 1;
        nodes.group.scale.y += (height - nodes.group.scale.y) * k;
        nodes.group.visible = nodes.group.scale.y > 0.01;
      }
      const pulse =
        animate && stageTones[id] === "danger"
          ? 0.5 * Math.sin(clock.elapsedTime * 3)
          : 0;
      for (const material of nodes.glow)
        material.emissiveIntensity = (id === active ? 2.6 : 1.5) + pulse;
    }
  });

  const handlers = (id: StageId) => ({
    onPointerOver: (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      document.body.style.cursor = "pointer";
      onHover(id);
    },
    onPointerOut: () => {
      document.body.style.cursor = "";
      onHover(null);
    },
    onClick: (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();
      onSelect(id);
    },
  });

  return (
    <>
      <primitive object={scene} />
      {(
        Object.entries(stages) as [StageId, NonNullable<StageNodes[StageId]>][]
      ).map(([id, nodes]) => (
        <group key={id}>
          {/* Invisible hit volume over the stage, so hover works across gaps between parts. */}
          <mesh
            position={nodes.box.getCenter(new THREE.Vector3())}
            {...handlers(id)}
          >
            <boxGeometry
              args={nodes.box.getSize(new THREE.Vector3()).toArray()}
            />
            <meshBasicMaterial visible={false} />
          </mesh>
          {(focus === null || focus === id) && (
            <Halo
              box={nodes.box}
              color={glowColor(stageTones[id], palette)}
              active={id === active}
              animate={animate}
            />
          )}
        </group>
      ))}
    </>
  );
}

// ------------------------------------------------------------------ traffic

interface RoadPath {
  points: THREE.Vector3[];
  /** Cumulative length at each point, metres. */
  lengths: number[];
  total: number;
}

function roadPath(points: [number, number, number][]): RoadPath {
  const vectors = points.map(
    (p) => new THREE.Vector3(p[0], BEAD_HEIGHT_M, p[2]),
  );
  const lengths = [0];
  for (let i = 1; i < vectors.length; i++)
    lengths.push(lengths[i - 1] + vectors[i].distanceTo(vectors[i - 1]));
  return { points: vectors, lengths, total: lengths[lengths.length - 1] };
}

function sampleRoad(path: RoadPath, distance: number, out: THREE.Vector3) {
  const d = ((distance % path.total) + path.total) % path.total;
  let i = 1;
  while (i < path.lengths.length - 1 && path.lengths[i] < d) i++;
  const span = path.lengths[i] - path.lengths[i - 1] || 1;
  return out.lerpVectors(
    path.points[i - 1],
    path.points[i],
    (d - path.lengths[i - 1]) / span,
  );
}

function Traffic({
  manifest,
  traffic,
  palette,
  animate,
}: Pick<CampusCanvasProps, "manifest" | "traffic" | "palette" | "animate">) {
  const roads = useMemo(
    () =>
      manifest.roads.flatMap((road) => {
        const spec = traffic[road.id];
        if (!spec) return [];
        const path = roadPath(road.points);
        const count = Math.max(1, Math.floor(path.total / BEAD_SPACING_M));
        const flags = spreadFlags(count, spec.flagShare);
        return [
          {
            id: road.id,
            path,
            count,
            kinds: flags.map((flag) => (flag ? spec.flag : spec.base)),
          },
        ];
      }),
    [manifest, traffic],
  );
  const geometry = useMemo(
    () => new THREE.SphereGeometry(BEAD_RADIUS_M, 18, 12),
    [],
  );
  const material = useMemo(
    () => new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0 }),
    [],
  );
  const meshes = useRef<(THREE.InstancedMesh | null)[]>([]);
  const offset = useRef(0);
  const scratch = useMemo(
    () => ({ position: new THREE.Vector3(), matrix: new THREE.Matrix4() }),
    [],
  );
  const invalidate = useThree((state) => state.invalidate);

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  // Colours follow the data; positions are set every frame below.
  useEffect(() => {
    const color = new THREE.Color();
    roads.forEach((road, r) => {
      const mesh = meshes.current[r];
      if (!mesh) return;
      road.kinds.forEach((kind, i) =>
        mesh.setColorAt(i, color.set(beadColor(kind, palette))),
      );
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    });
    invalidate();
  }, [roads, palette, invalidate]);

  useFrame((_, delta) => {
    if (animate)
      offset.current +=
        Math.min(delta, MAX_FRAME_DELTA_S) * TRAFFIC_SPEED_M_PER_S;
    roads.forEach((road, r) => {
      const mesh = meshes.current[r];
      if (!mesh) return;
      const spacing = road.path.total / road.count;
      for (let i = 0; i < road.count; i++) {
        sampleRoad(road.path, i * spacing + offset.current, scratch.position);
        mesh.setMatrixAt(i, scratch.matrix.makeTranslation(scratch.position));
      }
      mesh.instanceMatrix.needsUpdate = true;
    });
  });

  return (
    <>
      {roads.map((road, r) => (
        <instancedMesh
          key={road.id}
          ref={(mesh) => {
            meshes.current[r] = mesh;
          }}
          args={[geometry, material, road.count]}
          frustumCulled={false}
        />
      ))}
    </>
  );
}

// ------------------------------------------------------------------ components

/** Beads moving along fixed routes, coloured with the stage's status. */
function Packets({
  paths,
  color,
  animate,
}: {
  paths: [number, number, number][][];
  color: string;
  animate: boolean;
}) {
  const routes = useMemo(
    () =>
      paths.map((points) => {
        const vectors = points.map((p) => new THREE.Vector3(...p));
        const lengths = [0];
        for (let i = 1; i < vectors.length; i++)
          lengths.push(lengths[i - 1] + vectors[i].distanceTo(vectors[i - 1]));
        const total = lengths[lengths.length - 1];
        return {
          path: { points: vectors, lengths, total },
          count: Math.max(2, Math.floor(total / PACKET_SPACING_M)),
        };
      }),
    [paths],
  );
  const geometry = useMemo(
    () => new THREE.SphereGeometry(PACKET_RADIUS_M, 12, 8),
    [],
  );
  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false }),
    [],
  );
  const meshes = useRef<(THREE.InstancedMesh | null)[]>([]);
  const offset = useRef(0);
  const scratch = useMemo(
    () => ({ position: new THREE.Vector3(), matrix: new THREE.Matrix4() }),
    [],
  );

  useEffect(() => {
    material.color.set(color);
  }, [material, color]);
  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  useFrame((_, delta) => {
    if (animate)
      offset.current +=
        Math.min(delta, MAX_FRAME_DELTA_S) * PACKET_SPEED_M_PER_S;
    routes.forEach((route, r) => {
      const mesh = meshes.current[r];
      if (!mesh) return;
      const spacing = route.path.total / route.count;
      for (let i = 0; i < route.count; i++) {
        sampleRoad(route.path, i * spacing + offset.current, scratch.position);
        mesh.setMatrixAt(i, scratch.matrix.makeTranslation(scratch.position));
      }
      mesh.instanceMatrix.needsUpdate = true;
    });
  });

  return (
    <>
      {routes.map((route, r) => (
        <instancedMesh
          key={r}
          ref={(mesh) => {
            meshes.current[r] = mesh;
          }}
          args={[geometry, material, route.count]}
          frustumCulled={false}
        />
      ))}
    </>
  );
}

/** Per-frame updates for one model's moving parts, built when the model loads. */
type PartAnimators = {
  /** Advance every spinning part by `seconds`. */
  spin: (seconds: number) => void;
  /** Set every light's pulse for the clock time `t` (seconds). */
  blink: (t: number) => void;
};

/**
 * The close-up model of a stage (GPU, CPU board, database, router, store),
 * standing where its building was. Spinning fans, blinking lights and moving
 * packets are decorative; colour comes from the stage's measured status.
 */
function ComponentModel({
  stage,
  components,
  box,
  tone,
  palette,
  animate,
  onShown,
}: {
  stage: StageId;
  components: ComponentModels;
  box: THREE.Box3;
  tone: Tone | null;
  palette: ScenePalette;
  animate: boolean;
  onShown: (stage: StageId | null) => void;
}) {
  const entry = components[stage];
  const gltf = useLoader(GLTFLoader, entry.model);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf]);
  const group = useRef<THREE.Group>(null);
  const reveal = useRef(animate ? 0 : 1);
  const color = glowColor(tone, palette);

  // Animated parts are engine objects mutated every frame, so they live in a
  // ref filled by an effect, not in a memoised render value.
  const parts = useRef<PartAnimators | null>(null);

  useEffect(() => {
    const spins: {
      node: THREE.Object3D;
      speed: number;
      axis: "x" | "y" | "z";
    }[] = [];
    const blinks: { material: THREE.MeshStandardMaterial; phase: number }[] =
      [];
    const glows: THREE.MeshStandardMaterial[] = [];
    scene.traverse((node) => {
      const speed = Number(node.userData.aw_spin);
      if (Number.isFinite(speed) && speed !== 0) {
        const axis = node.userData.aw_spin_axis;
        spins.push({
          node,
          speed,
          axis: axis === "x" || axis === "z" ? axis : "y",
        });
      }
      if (!(node instanceof THREE.Mesh)) return;
      node.castShadow = true;
      node.receiveShadow = true;
      const material = node.material as THREE.MeshStandardMaterial;
      const key = STRUCTURE_MATERIALS[material.name];
      if (key) material.color.set(palette[key] as string);
      if (node.name.startsWith("blink_")) {
        // Each light pulses on its own phase, so it needs its own material.
        const own = material.clone();
        node.material = own;
        blinks.push({
          material: own,
          phase: Number(node.userData.aw_phase) || 0,
        });
        glows.push(own);
      } else if (material.name.startsWith("aw_glow_")) {
        glows.push(material);
      }
    });
    for (const material of glows) {
      material.color.set(color);
      material.emissive.set(color);
      material.toneMapped = false;
    }
    // Mutations happen inside these closures, created here with the parts.
    parts.current = {
      spin: (seconds) => {
        for (const part of spins)
          part.node.rotation[part.axis] += part.speed * seconds;
      },
      blink: (t) => {
        for (const part of blinks)
          part.material.emissiveIntensity =
            0.5 +
            1.8 * (0.5 + 0.5 * Math.sin(t * 3.2 + part.phase * Math.PI * 2));
      },
    };
    return () => {
      for (const blink of blinks) blink.material.dispose();
      parts.current = null;
    };
  }, [scene, palette, color]);

  // Fit the model's footprint to its building's, centred on the same spot.
  const placement = useMemo(() => {
    const own = new THREE.Box3().setFromObject(scene);
    const ownSize = own.getSize(new THREE.Vector3());
    const footprint = box.getSize(new THREE.Vector3());
    const scale =
      Math.min(footprint.x / ownSize.x, footprint.z / ownSize.z) * 1.05;
    const center = box.getCenter(new THREE.Vector3());
    return { scale, position: new THREE.Vector3(center.x, 0, center.z) };
  }, [scene, box]);

  useEffect(() => {
    onShown(stage);
    return () => onShown(null);
  }, [stage, onShown]);

  useFrame(({ clock }, delta) => {
    const step = Math.min(delta, MAX_FRAME_DELTA_S);
    if (group.current && reveal.current < 1) {
      reveal.current = Math.min(1, reveal.current + step / REVEAL_S);
      const eased = 1 - Math.pow(1 - reveal.current, 3);
      group.current.scale.set(
        placement.scale,
        placement.scale * eased,
        placement.scale,
      );
    }
    const current = parts.current;
    if (!animate || !current) return;
    current.spin(step);
    current.blink(clock.elapsedTime);
  });

  return (
    <group
      ref={group}
      position={placement.position}
      // The rise from the podium starts flat when animated; useFrame grows it.
      scale={[
        placement.scale,
        placement.scale * (animate ? 0.001 : 1),
        placement.scale,
      ]}
      // Clicks on the model keep it open instead of counting as a miss.
      onClick={(event) => event.stopPropagation()}
    >
      <primitive object={scene} />
      <Packets paths={entry.packetPaths} color={color} animate={animate} />
    </group>
  );
}

// ------------------------------------------------------------------ labels

/** Projects each stage anchor to canvas pixels so the page can place its labels. */
function LabelProjector({
  manifest,
  onProject,
}: Pick<CampusCanvasProps, "manifest" | "onProject">) {
  const anchors = useMemo(
    () =>
      Object.entries(manifest.stages).map(
        ([id, stage]) =>
          [id as StageId, new THREE.Vector3(...stage.anchor)] as const,
      ),
    [manifest],
  );
  const scratch = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    for (const [id, anchor] of anchors) {
      scratch.copy(anchor).project(camera);
      onProject(
        id,
        ((scratch.x + 1) / 2) * size.width,
        ((1 - scratch.y) / 2) * size.height,
      );
    }
  });
  return null;
}

class SceneErrorBoundary extends Component<
  { onError: () => void; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/** The interactive 3D campus. Loaded only in the browser (see CampusLiveScene). */
export default function CampusCanvas(props: CampusCanvasProps) {
  const {
    manifest,
    components,
    focus,
    focusRegion,
    animate,
    palette,
    onProject,
    onError,
  } = props;
  const [boxes, setBoxes] = useState<Partial<Record<StageId, THREE.Box3>>>({});
  const [shown, setShown] = useState<StageId | null>(null);

  // Fetch the close-up models once the campus is up, so a click opens at once.
  useEffect(() => {
    for (const entry of Object.values(components))
      useLoader.preload(GLTFLoader, entry.model);
  }, [components]);

  const focusBox = focus ? boxes[focus] : undefined;
  return (
    <Canvas
      orthographic
      shadows="percentage"
      dpr={[1, 2]}
      frameloop={animate ? "always" : "demand"}
      camera={{ manual: true, position: manifest.camera.position }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.NeutralToneMapping;
        gl.toneMappingExposure = 1.05;
      }}
      onPointerMissed={() => {
        props.onHover(null);
        props.onMiss();
      }}
      aria-hidden
    >
      <SceneErrorBoundary onError={onError}>
        <CameraRig
          manifest={manifest}
          focus={focus}
          focusBox={focusBox}
          focusRegion={focusRegion}
          animate={animate}
        />
        <Lighting />
        <Ground color={palette.ground} center={manifest.camera.target} />
        <Suspense fallback={null}>
          <Campus
            {...props}
            active={focus ?? props.active}
            hidden={shown}
            focus={focus}
            onStages={setBoxes}
          />
          {focus === null && <Traffic {...props} />}
        </Suspense>
        {/* Its own boundary: loading a model must not blank the campus. */}
        <Suspense fallback={null}>
          {focus && focusBox && (
            <ComponentModel
              key={focus}
              stage={focus}
              components={components}
              box={focusBox}
              tone={props.stageTones[focus]}
              palette={palette}
              animate={animate}
              onShown={setShown}
            />
          )}
        </Suspense>
        <LabelProjector manifest={manifest} onProject={onProject} />
      </SceneErrorBoundary>
    </Canvas>
  );
}
