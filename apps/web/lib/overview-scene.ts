import { z } from "zod";
import campus3dJson from "./overview-campus-3d.json";
import componentModelsJson from "./overview-component-models.json";
import anchorsJson from "./overview-scene-anchors.json";

/**
 * Label anchors for the Overview campus render. `scenes/overview_campus.py`
 * projects each building's anchor through the Blender camera and writes this
 * file, so overlays stay aligned with the image when the scene is re-rendered.
 */
export const SCENE_ANCHOR_IDS = [
  "gateway",
  "model",
  "retrieval",
  "cpuTool",
  "complete",
] as const;
export type SceneAnchorId = (typeof SCENE_ANCHOR_IDS)[number];

const fraction = z.number().finite().min(0).max(1);
const anchor = z.object({ x: fraction, y: fraction });

export const sceneManifestSchema = z.object({
  image: z.object({
    widthPx: z.number().int().positive(),
    heightPx: z.number().int().positive(),
  }),
  anchors: z.object({
    gateway: anchor,
    model: anchor,
    retrieval: anchor,
    cpuTool: anchor,
    complete: anchor,
  }),
});

export type SceneManifest = z.infer<typeof sceneManifestSchema>;

/** Parsed at import so a bad re-render fails the build, not the page. */
export const overviewScene: SceneManifest =
  sceneManifestSchema.parse(anchorsJson);

export const OVERVIEW_SCENE_IMAGES = {
  light: "/scenes/overview-campus-light.webp",
  dark: "/scenes/overview-campus-dark.webp",
} as const;

const vec3 = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
]);
const stage3d = z.object({ anchor: vec3, center: vec3 });

/**
 * The 3D campus: camera, stage anchors and road centre lines in glTF Y-up
 * metres, written with the GLB by `scenes/overview_campus.py`.
 */
export const campus3dSchema = z.object({
  model: z.string().regex(/^\/scenes\/[\w.-]+\.glb$/),
  camera: z.object({
    position: vec3,
    target: vec3,
    orthoWidth: z.number().positive(),
    shift: z.tuple([z.number().finite(), z.number().finite()]),
    aspect: z.number().positive(),
  }),
  stages: z.object({
    gateway: stage3d,
    model: stage3d,
    retrieval: stage3d,
    cpuTool: stage3d,
    complete: stage3d,
  }),
  roads: z
    .array(
      z.object({
        id: z.string().min(1),
        stage: z.enum(SCENE_ANCHOR_IDS),
        points: z.array(vec3).min(2),
      }),
    )
    .min(1),
});

export type Campus3D = z.infer<typeof campus3dSchema>;

export const campus3d: Campus3D = campus3dSchema.parse(campus3dJson);

const componentModel = z.object({
  model: z.string().regex(/^\/scenes\/component-[\w-]+\.glb$/),
  /** Packet routes in the model's own glTF coordinates (metres). */
  packetPaths: z.array(z.array(vec3).min(2)).max(8),
});

/**
 * Close-up models shown when a stage is clicked, written by
 * `scenes/component_models.py` next to their GLBs.
 */
export const componentModelsSchema = z.object({
  gateway: componentModel,
  model: componentModel,
  retrieval: componentModel,
  cpuTool: componentModel,
  complete: componentModel,
});

export type ComponentModels = z.infer<typeof componentModelsSchema>;

export const componentModels: ComponentModels =
  componentModelsSchema.parse(componentModelsJson);
