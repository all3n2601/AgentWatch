"""Build and render the Overview page's isometric infrastructure campus.

The campus is a spatial map of the agent pipeline (gateway, model runtime,
retrieval, CPU tool, completion). The geometry encodes no telemetry: the web
page colours each stage and its traffic from measured data at runtime, using
the per-stage glow materials (``aw_glow_<stage>``) and road paths exported
here. The static renders are the no-WebGL fallback and keep neutral traffic.

Run inside Blender 5.2+:

    blender --background --python apps/web/scenes/overview_campus.py

or, from Blender's Python console or the MCP add-on, import it with
``importlib.util.spec_from_file_location`` and call ``main()``. Outputs,
relative to
``apps/web``:

    public/scenes/overview-campus-light.webp   (fallback render)
    public/scenes/overview-campus-dark.webp    (fallback render)
    public/scenes/overview-campus.glb          (geometry for the 3D scene)
    lib/overview-scene-anchors.json   (fallback label anchors, 0-1 image fractions)
    lib/overview-campus-3d.json       (camera, stage anchors, road paths; glTF Y-up)
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

WEB_ROOT = Path(__file__).resolve().parent.parent

RENDER_WIDTH_PX = 2880
CAMERA_ORTHO_SCALE = 24.5  # starting scale; frame_campus() fits it to the buildings
# Room around the campus (share of its width), and headroom for the stage labels.
FRAME_MARGIN = 0.03
CAMERA_ELEVATION_DEG = 30.0
CAMERA_DISTANCE_M = 40.0
CAMPUS_CENTER = (1.0, 1.5, 0.0)
LABEL_HEADROOM_M = 1.3
COLLECTION_NAME = "AgentWatch Overview Campus"
BEAD_SPACING_M = 0.42
ROAD_WIDTH_M = 0.62
WEBP_QUALITY = 90
RENDER_SAMPLES = 128
STAGES = ("gateway", "model", "retrieval", "cpuTool", "complete")

GROUND_CENTER = (0.2, 0.6)
GROUND_FADE_RADIUS_X_M = 9.0
GROUND_FADE_RADIUS_Y_M = 7.5

GREEN = (0.076, 0.571, 0.027)  # brand #4ec72e, as linear RGB


@dataclass(frozen=True)
class Palette:
    name: str
    structure: tuple[float, float, float]
    structure_shade: tuple[float, float, float]
    recess: tuple[float, float, float]
    ground: tuple[float, float, float]
    road: tuple[float, float, float]
    bead: tuple[float, float, float]
    world: tuple[float, float, float]
    world_strength: float
    sun_strength: float
    glow_strength: float


LIGHT = Palette(
    name="light",
    structure=(0.86, 0.875, 0.87),
    structure_shade=(0.74, 0.76, 0.755),
    recess=(0.32, 0.35, 0.34),
    ground=(0.80, 0.815, 0.81),
    road=(0.66, 0.68, 0.675),
    bead=(0.95, 0.96, 0.955),
    world=(0.96, 0.97, 0.965),
    world_strength=0.45,
    sun_strength=1.8,
    glow_strength=1.4,
)

DARK = Palette(
    name="dark",
    structure=(0.11, 0.125, 0.135),
    structure_shade=(0.075, 0.085, 0.095),
    recess=(0.03, 0.035, 0.04),
    ground=(0.07, 0.08, 0.09),
    road=(0.12, 0.135, 0.145),
    bead=(0.55, 0.58, 0.6),
    world=(0.09, 0.11, 0.13),
    world_strength=0.35,
    sun_strength=1.1,
    glow_strength=2.2,
)


# ---------------------------------------------------------------- scene setup


def reset_collection() -> bpy.types.Collection:
    """Replace only this generator's collection; leave other scene data alone."""
    existing = bpy.data.collections.get(COLLECTION_NAME)
    if existing is not None:
        for obj in list(existing.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing)
    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    return collection


def hide_startup_objects(collection: bpy.types.Collection) -> None:
    """Keep Blender's startup cube and light out of the render without deleting them."""
    for obj in bpy.context.scene.objects:
        if obj.users_collection and collection not in obj.users_collection:
            obj.hide_render = True
            obj.hide_viewport = True


def material(
    name: str, color: tuple[float, float, float], roughness: float = 0.6, emission: float = 0.0
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
    bsdf.inputs["Emission Strength"].default_value = emission
    return mat


class Builder:
    def __init__(self, collection: bpy.types.Collection, palette: Palette) -> None:
        self.collection = collection
        self.palette = palette
        self.white = material("aw_structure", palette.structure, 0.55)
        self.shade = material("aw_structure_shade", palette.structure_shade, 0.6)
        self.recess = material("aw_recess", palette.recess, 0.35)
        self.ground = material("aw_ground", palette.ground, 0.75)
        self.road = material("aw_road", palette.road, 0.7)
        self.bead_material = material("aw_bead", palette.bead, 0.3)
        self.glass = material("aw_glass", palette.recess, 0.12)
        # One glow material per stage so the page can tint each stage's status.
        self.glows = {
            stage_id: material(f"aw_glow_{stage_id}", GREEN, 0.4, palette.glow_strength)
            for stage_id in STAGES
        }
        self.glow = self.glows["gateway"]
        self.stage_root: bpy.types.Object | None = None

        # Every bead shares one mesh; the data API avoids an operator (and a
        # depsgraph update) per object, which keeps the build to a few seconds.
        self.bead_mesh = bpy.data.meshes.new("aw_bead")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.11)
        bm.to_mesh(self.bead_mesh)
        bm.free()
        self.bead_mesh.shade_smooth()
        self.bead_mesh.materials.append(self.bead_material)

    def box(
        self,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        mat: bpy.types.Material,
        bevel_m: float = 0.04,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=size, verts=bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = center
        if bevel_m > 0:
            bevel = obj.modifiers.new("bevel", "BEVEL")
            bevel.width = bevel_m
            bevel.segments = 3
            bevel.limit_method = "ANGLE"
            bevel.harden_normals = True
        return self.place(obj)

    def cylinder(
        self,
        name: str,
        center: tuple[float, float, float],
        radius_m: float,
        depth_m: float,
        mat: bpy.types.Material,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm, cap_ends=True, segments=32, radius1=radius_m, radius2=radius_m, depth=depth_m
        )
        bm.to_mesh(mesh)
        bm.free()
        mesh.shade_smooth()
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = center
        return self.place(obj)

    def place(self, obj: bpy.types.Object) -> bpy.types.Object:
        """Link an object, parent it to the current stage, and mark it for export."""
        self.collection.objects.link(obj)
        obj["aw_export"] = True
        if self.stage_root is not None:
            obj.parent = self.stage_root
        return obj

    def begin_stage(self, stage_id: str) -> None:
        root = bpy.data.objects.new(f"stage_{stage_id}", None)
        root["aw_export"] = True
        root["aw_stage"] = stage_id
        self.collection.objects.link(root)
        self.stage_root = root
        self.glow = self.glows[stage_id]

    def end_stage(self) -> None:
        self.stage_root = None

    def bead(self, name: str, center: Vector) -> bpy.types.Object:
        obj = bpy.data.objects.new(name, self.bead_mesh)
        obj.location = center
        self.collection.objects.link(obj)
        return obj


def faded_ground(palette: Palette) -> bpy.types.Material:
    """Ground that fades radially to transparent so the campus floats on the page."""
    mat = bpy.data.materials.get("aw_ground_fade") or bpy.data.materials.new("aw_ground_fade")
    mat.use_nodes = True
    mat.surface_render_method = "BLENDED"  # smooth alpha; DITHERED speckles the fade
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*palette.ground, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    clear = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (
        1 / GROUND_FADE_RADIUS_X_M,
        1 / GROUND_FADE_RADIUS_Y_M,
        1,
    )
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "SPHERICAL"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.interpolation = "EASE"
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    links.new(clear.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------- structures
# World layout. The camera looks along (-1, 1, -1), so screen-right is +x+y
# and screen-up (further away) is -x+y.


def louvers(
    b: Builder,
    prefix: str,
    x0: float,
    x1: float,
    y: float,
    z0: float,
    z1: float,
    count: int,
    depth: float = 0.08,
) -> None:
    """Vertical fins on the front (-y) face of a block."""
    step = (x1 - x0) / count
    for i in range(count):
        x = x0 + step * (i + 0.5)
        b.box(
            f"{prefix}_fin_{i}",
            (x, y - depth / 2, (z0 + z1) / 2),
            (step * 0.42, depth, z1 - z0),
            b.white,
            0.01,
        )
    b.box(
        f"{prefix}_fin_back",
        ((x0 + x1) / 2, y + 0.01, (z0 + z1) / 2),
        (x1 - x0, 0.02, z1 - z0),
        b.recess,
        0,
    )


def side_louvers(
    b: Builder,
    prefix: str,
    x: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    count: int,
    depth: float = 0.08,
) -> None:
    """Vertical fins on the right (+x) face of a block."""
    step = (y1 - y0) / count
    for i in range(count):
        y = y0 + step * (i + 0.5)
        b.box(
            f"{prefix}_sfin_{i}",
            (x + depth / 2, y, (z0 + z1) / 2),
            (depth, step * 0.42, z1 - z0),
            b.white,
            0.01,
        )
    b.box(
        f"{prefix}_sfin_back",
        (x - 0.01, (y0 + y1) / 2, (z0 + z1) / 2),
        (0.02, y1 - y0, z1 - z0),
        b.recess,
        0,
    )


def podium(b: Builder, name: str, cx: float, cy: float, sx: float, sy: float) -> None:
    """Two-step plinth: a rounded base slab and an inset paving slab."""
    b.box(f"{name}_podium", (cx, cy, 0.09), (sx, sy, 0.18), b.white, 0.08)
    b.box(f"{name}_paving", (cx, cy, 0.2), (sx - 0.3, sy - 0.3, 0.05), b.shade, 0.03)


def glass_band(
    b: Builder, prefix: str, x0: float, x1: float, y: float, z0: float, z1: float, panes: int
) -> None:
    """Recessed glazing on a front (-y) face, split by thin mullions."""
    b.box(
        f"{prefix}_glass",
        ((x0 + x1) / 2, y + 0.005, (z0 + z1) / 2),
        (x1 - x0, 0.03, z1 - z0),
        b.glass,
        0,
    )
    step = (x1 - x0) / panes
    for i in range(panes + 1):
        b.box(
            f"{prefix}_mullion_{i}",
            (x0 + step * i, y - 0.02, (z0 + z1) / 2),
            (0.04, 0.05, z1 - z0),
            b.white,
            0,
        )
    b.box(
        f"{prefix}_sill",
        ((x0 + x1) / 2, y - 0.04, z0 - 0.03),
        (x1 - x0 + 0.1, 0.1, 0.05),
        b.white,
        0.01,
    )


def side_glass_band(
    b: Builder, prefix: str, x: float, y0: float, y1: float, z0: float, z1: float, panes: int
) -> None:
    """Recessed glazing on a right (+x) face, split by thin mullions."""
    b.box(
        f"{prefix}_sglass",
        (x - 0.005, (y0 + y1) / 2, (z0 + z1) / 2),
        (0.03, y1 - y0, z1 - z0),
        b.glass,
        0,
    )
    step = (y1 - y0) / panes
    for i in range(panes + 1):
        b.box(
            f"{prefix}_smullion_{i}",
            (x + 0.02, y0 + step * i, (z0 + z1) / 2),
            (0.05, 0.04, z1 - z0),
            b.white,
            0,
        )


def roof_unit(b: Builder, name: str, cx: float, cy: float, z: float, size: float = 0.8) -> None:
    """Rooftop air handler: a housing with a recessed fan."""
    b.box(f"{name}_housing", (cx, cy, z + 0.16), (size, size, 0.32), b.white, 0.04)
    b.cylinder(f"{name}_fan_well", (cx, cy, z + 0.33), size * 0.34, 0.04, b.recess)
    b.cylinder(f"{name}_fan_hub", (cx, cy, z + 0.36), size * 0.08, 0.05, b.shade)


def parapet(b: Builder, name: str, cx: float, cy: float, z: float, sx: float, sy: float) -> None:
    """Roof cap with a slight overhang, which reads as a crisp top edge in shading."""
    b.box(f"{name}_parapet", (cx, cy, z + 0.06), (sx + 0.12, sy + 0.12, 0.12), b.shade, 0.03)


def gateway(b: Builder, cx: float, cy: float) -> Vector:
    b.begin_stage("gateway")
    podium(b, "gateway", cx, cy, 3.4, 3.0)
    b.box("gateway_hall", (cx, cy + 0.2, 1.0), (2.6, 2.2, 1.6), b.white, 0.08)
    parapet(b, "gateway", cx, cy + 0.2, 1.8, 2.6, 2.2)
    roof_unit(b, "gateway_unit_a", cx + 0.45, cy + 0.65, 1.92)
    roof_unit(b, "gateway_unit_b", cx - 0.5, cy + 0.65, 1.92, 0.6)
    glass_band(b, "gateway_front", cx - 1.05, cx + 0.75, cy - 0.91, 0.75, 1.45, 5)
    # Glowing entrance on the right face, opening toward the road, under a canopy.
    b.box("gateway_door_frame", (cx + 1.32, cy - 0.1, 0.74), (0.08, 1.3, 1.1), b.recess, 0)
    b.box("gateway_door", (cx + 1.37, cy - 0.1, 0.7), (0.04, 1.1, 0.95), b.glow, 0)
    b.box("gateway_canopy", (cx + 1.6, cy - 0.1, 1.36), (0.6, 1.6, 0.08), b.white, 0.02)
    b.end_stage()
    return Vector((cx, cy, 2.4))


def model_runtime(b: Builder, cx: float, cy: float) -> Vector:
    b.begin_stage("model")
    podium(b, "model", cx, cy, 5.2, 4.2)
    # Stepped cluster of server halls.
    b.box("model_core", (cx - 0.4, cy + 0.5, 2.3), (2.6, 2.4, 4.2), b.white, 0.1)
    parapet(b, "model_core", cx - 0.4, cy + 0.5, 4.4, 2.6, 2.4)
    b.box("model_wing", (cx + 1.35, cy + 0.2, 1.4), (1.4, 2.6, 2.4), b.white, 0.08)
    parapet(b, "model_wing", cx + 1.35, cy + 0.2, 2.6, 1.4, 2.6)
    b.box("model_low", (cx - 1.2, cy - 1.1, 0.8), (2.0, 1.2, 1.2), b.white, 0.06)
    parapet(b, "model_low", cx - 1.2, cy - 1.1, 1.4, 2.0, 1.2)
    louvers(b, "model_core", cx - 1.55, cx + 0.75, cy - 0.7, 1.7, 4.1, 9)
    side_louvers(b, "model_wing", cx + 2.05, cy - 1.0, cy + 1.4, 0.6, 2.4, 7)
    # Glazed rack hall; one status light per rack bay uses the stage glow.
    glass_band(b, "model_hall", cx - 2.1, cx - 0.3, cy - 1.71, 0.45, 1.2, 5)
    for i in range(5):
        x = cx - 1.92 + i * 0.36
        b.box(f"model_rack_led_{i}", (x, cy - 1.75, 1.05), (0.2, 0.02, 0.04), b.glow, 0)
    for i in range(3):
        roof_unit(b, f"model_roof_{i}", cx - 1.15 + i * 0.75, cy + 0.9, 4.52, 0.62)
    # The model runtime's base glows: this is where inference work happens.
    b.box("model_glow_strip", (cx + 0.25, cy - 1.86, 0.3), (3.6, 0.06, 0.08), b.glow, 0)
    b.box("model_glow_side", (cx + 2.58, cy - 0.2, 0.3), (0.06, 3.4, 0.08), b.glow, 0)
    b.end_stage()
    return Vector((cx - 0.4, cy + 0.5, 5.2))


def retrieval(b: Builder, cx: float, cy: float) -> Vector:
    b.begin_stage("retrieval")
    podium(b, "retrieval", cx, cy, 3.6, 3.2)
    b.box("retrieval_block", (cx, cy, 1.5), (2.8, 2.4, 2.6), b.white, 0.08)
    parapet(b, "retrieval", cx, cy, 2.8, 2.8, 2.4)
    b.box("retrieval_upper", (cx - 0.4, cy + 0.3, 3.2), (1.6, 1.6, 0.7), b.white, 0.06)
    parapet(b, "retrieval_upper", cx - 0.4, cy + 0.3, 3.55, 1.6, 1.6)
    side_glass_band(b, "retrieval_upper", cx + 0.41, cy - 0.4, cy + 1.0, 3.0, 3.4, 3)
    louvers(b, "retrieval", cx - 1.2, cx + 1.2, cy - 1.2, 0.7, 2.5, 10)
    side_louvers(b, "retrieval", cx + 1.4, cy - 1.0, cy + 1.0, 0.9, 2.5, 6)
    roof_unit(b, "retrieval_roof", cx + 0.75, cy - 0.5, 2.92, 0.62)
    b.box("retrieval_glow_strip", (cx, cy - 1.31, 0.3), (2.6, 0.06, 0.08), b.glow, 0)
    b.end_stage()
    return Vector((cx - 0.4, cy + 0.3, 4.0))


def cpu_tool(b: Builder, cx: float, cy: float) -> Vector:
    b.begin_stage("cpuTool")
    podium(b, "cpu", cx, cy, 4.0, 3.2)
    b.box("cpu_block", (cx - 0.3, cy + 0.1, 1.2), (2.4, 2.4, 1.95), b.white, 0.08)
    parapet(b, "cpu_block", cx - 0.3, cy + 0.1, 2.18, 2.4, 2.4)
    b.box("cpu_annex", (cx + 1.25, cy - 0.1, 0.95), (1.0, 2.0, 1.45), b.white, 0.06)
    parapet(b, "cpu_annex", cx + 1.25, cy - 0.1, 1.68, 1.0, 2.0)
    louvers(b, "cpu", cx - 1.4, cx + 0.8, cy - 1.1, 0.55, 1.9, 9)
    side_louvers(b, "cpu_annex", cx + 1.75, cy - 1.0, cy + 0.8, 0.45, 1.45, 6)
    roof_unit(b, "cpu_roof_a", cx - 0.75, cy + 0.45, 2.3)
    roof_unit(b, "cpu_roof_b", cx + 0.2, cy + 0.45, 2.3, 0.6)
    b.box("cpu_glow_strip", (cx - 0.3, cy - 1.21, 0.3), (2.2, 0.06, 0.08), b.glow, 0)
    b.end_stage()
    return Vector((cx - 0.3, cy + 0.1, 2.9))


def completion(b: Builder, cx: float, cy: float) -> Vector:
    b.begin_stage("complete")
    podium(b, "complete", cx, cy, 2.8, 2.6)
    b.box("complete_hall", (cx, cy + 0.15, 1.1), (2.0, 1.9, 1.75), b.white, 0.08)
    parapet(b, "complete", cx, cy + 0.15, 1.98, 2.0, 1.9)
    roof_unit(b, "complete_roof", cx + 0.35, cy + 0.45, 2.1, 0.62)
    side_glass_band(b, "complete_side", cx + 1.01, cy - 0.5, cy + 0.9, 0.7, 1.5, 4)
    b.box("complete_door_frame", (cx - 0.2, cy - 0.82, 0.76), (1.1, 0.08, 1.1), b.recess, 0)
    b.box("complete_door", (cx - 0.2, cy - 0.87, 0.72), (0.92, 0.04, 0.95), b.glow, 0)
    b.box("complete_canopy", (cx - 0.2, cy - 1.1, 1.38), (1.5, 0.6, 0.08), b.white, 0.02)
    b.end_stage()
    return Vector((cx, cy + 0.15, 2.6))


# ---------------------------------------------------------------- roads


def catmull_rom(points: list[Vector], samples_per_segment: int = 24) -> list[Vector]:
    """Smooth polyline through every control point (endpoints are clamped)."""
    padded = [points[0], *points, points[-1]]
    out: list[Vector] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    out.append(points[-1])
    return out


def resample(path: list[Vector], spacing_m: float) -> list[Vector]:
    beads = [path[0]]
    carried = 0.0
    for a, b in itertools.pairwise(path):
        seg = (b - a).length
        pos = spacing_m - carried
        while pos <= seg:
            beads.append(a.lerp(b, pos / seg))
            pos += spacing_m
        carried = seg - (pos - spacing_m)
    return beads


def road(builder: Builder, name: str, controls: list[tuple[float, float]]) -> list[Vector]:
    """A soft-edged ribbon road; returns its centre line for the 3D scene's traffic."""
    pts = catmull_rom([Vector((x, y, 0.0)) for x, y in controls])
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = ROAD_WIDTH_M / 2
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for p, v in zip(spline.points, pts):
        p.co = (v.x, v.y, 0.0, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.scale = (1.0, 1.0, 0.12)  # flatten the tube into a soft-edged ribbon
    obj.location.z = 0.03
    obj.data.materials.append(builder.road)
    builder.place(obj)

    # Neutral beads for the static fallback only; the 3D scene draws its own traffic.
    for i, v in enumerate(resample(pts, BEAD_SPACING_M)[1:-1]):
        builder.bead(f"{name}_bead_{i}", v + Vector((0, 0, 0.13)))
    return pts


# Each road carries one stage's traffic, so the page can colour it by that stage's evidence.
ROADS: tuple[tuple[str, str, list[tuple[float, float]]], ...] = (
    ("ingress", "gateway", [(-4.6, -2.3), (-3.2, -2.8), (-1.6, -2.2), (0.6, -0.6), (1.2, 0.6)]),
    ("model", "model", [(-4.2, -0.2), (-2.8, -0.6), (-1.2, -0.4), (0.4, 0.2), (1.2, 0.6)]),
    (
        "retrieval",
        "retrieval",
        [
            (1.2, 0.6),
            (1.4, 2.6),
            (1.6, 4.4),
            (2.8, 4.4),
            (5.6, 4.2),
            (6.2, 2.8),
            (5.6, 1.4),
            (4.0, 1.0),
            (2.0, 1.0),
            (1.2, 0.6),
        ],
    ),
    ("tool", "cpuTool", [(1.2, 0.6), (1.0, -0.4), (1.3, -1.4), (1.4, -2.6), (3.0, -3.4)]),
    ("egress", "complete", [(3.0, -3.4), (5.6, -2.4), (6.6, -0.6), (8.2, -0.2)]),
)


# ---------------------------------------------------------------- camera, light


def stage(collection: bpy.types.Collection, palette: Palette) -> bpy.types.Object:
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("aw_camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = CAMERA_ORTHO_SCALE
    cam = bpy.data.objects.new("aw_camera", cam_data)
    collection.objects.link(cam)
    # Isometric heading, viewed from CAMERA_ELEVATION_DEG above the ground like
    # the approved mockup; the camera backs off along its view axis from the
    # campus centre, and frame_campus() then fits scale and shift.
    cam.rotation_euler = (math.radians(90 - CAMERA_ELEVATION_DEG), 0.0, math.radians(45))
    forward = cam.rotation_euler.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    cam.location = Vector(CAMPUS_CENTER) - forward * CAMERA_DISTANCE_M
    cam_data.clip_end = 2 * CAMERA_DISTANCE_M
    scene.camera = cam

    sun_data = bpy.data.lights.new("aw_sun", "SUN")
    sun_data.energy = palette.sun_strength
    sun_data.angle = math.radians(18)  # broad, soft shadows
    sun = bpy.data.objects.new("aw_sun", sun_data)
    sun.rotation_euler = (math.radians(42), math.radians(8), math.radians(-28))
    collection.objects.link(sun)

    world = scene.world or bpy.data.worlds.new("aw_world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*palette.world, 1.0)
    bg.inputs["Strength"].default_value = palette.world_strength

    render = scene.render
    render.engine = "BLENDER_EEVEE"
    render.resolution_x = RENDER_WIDTH_PX
    render.resolution_percentage = 100
    render.film_transparent = True
    render.image_settings.file_format = "WEBP"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.quality = WEBP_QUALITY
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.eevee.taa_render_samples = RENDER_SAMPLES
    scene.eevee.use_raytracing = True
    scene.eevee.use_shadows = True
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True
    return cam


# ---------------------------------------------------------------- build


@dataclass
class Campus:
    camera: bpy.types.Object
    collection: bpy.types.Collection
    anchors: dict[str, Vector]
    centers: dict[str, Vector]
    roads: list[tuple[str, str, list[Vector]]]


def build(palette: Palette) -> Campus:
    collection = reset_collection()
    b = Builder(collection, palette)

    ground = b.box(
        "ground", (GROUND_CENTER[0], GROUND_CENTER[1], -0.06), (34.0, 26.0, 0.12), b.ground, 0
    )
    ground.data.materials[0] = faded_ground(palette)
    del ground["aw_export"]  # the 3D scene draws its own fading ground

    sites = {
        "gateway": (gateway, (-6.4, -2.2)),
        "model": (model_runtime, (-1.6, 2.6)),
        "retrieval": (retrieval, (3.6, 6.2)),
        "cpuTool": (cpu_tool, (3.2, -1.2)),
        "complete": (completion, (8.6, 1.4)),
    }
    anchors = {key: fn(b, *xy) for key, (fn, xy) in sites.items()}
    centers = {key: Vector((xy[0], xy[1], 0.0)) for key, (_, xy) in sites.items()}
    roads = [
        (road_id, stage_id, road(b, f"road_{road_id}", pts)) for road_id, stage_id, pts in ROADS
    ]

    hide_startup_objects(collection)
    cam = stage(collection, palette)
    bpy.context.view_layer.update()
    frame_campus(cam, collection, anchors)
    return Campus(cam, collection, anchors, centers, roads)


def frame_campus(
    cam: bpy.types.Object, collection: bpy.types.Collection, anchors: dict[str, Vector]
) -> None:
    """Fit the orthographic frame tightly around the buildings and their labels.

    The page lays its own callouts beside the scene, so the render carries no
    empty margin; its aspect follows the campus instead of a fixed size.
    """
    to_camera = cam.matrix_world.inverted()
    points = [
        to_camera @ (obj.matrix_world @ Vector(corner))
        for obj in collection.objects
        if obj.get("aw_export") and obj.type in {"MESH", "CURVE"}
        for corner in obj.bound_box
    ]
    points += [
        to_camera @ (anchor + Vector((0, 0, LABEL_HEADROOM_M))) for anchor in anchors.values()
    ]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    width_m = (max(xs) - min(xs)) * (1 + 2 * FRAME_MARGIN)
    height_m = (max(ys) - min(ys)) + (max(xs) - min(xs)) * 2 * FRAME_MARGIN
    cam.data.ortho_scale = width_m
    # Blender's shift is a fraction of the larger frame dimension (the width here).
    cam.data.shift_x = ((max(xs) + min(xs)) / 2) / width_m
    cam.data.shift_y = ((max(ys) + min(ys)) / 2) / width_m
    render = bpy.context.scene.render
    render.resolution_x = RENDER_WIDTH_PX
    render.resolution_y = round(RENDER_WIDTH_PX * height_m / width_m)
    bpy.context.view_layer.update()


def project(cam: bpy.types.Object, point: Vector) -> Vector:
    from bpy_extras.object_utils import world_to_camera_view

    ndc = world_to_camera_view(bpy.context.scene, cam, point)
    # Image space: origin top-left, both axes as 0-1 fractions of the render.
    return Vector((ndc.x, 1.0 - ndc.y))


def to_gltf(v: Vector) -> list[float]:
    """Blender Z-up to glTF Y-up: (x, y, z) -> (x, z, -y), rounded to millimetres."""
    return [round(v.x, 3), round(v.z, 3), round(-v.y, 3)]


def render(palette: Palette) -> Campus:
    campus = build(palette)
    out = WEB_ROOT / "public" / "scenes" / f"overview-campus-{palette.name}.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    return campus


def export_glb(campus: Campus) -> Path:
    out = WEB_ROOT / "public" / "scenes" / "overview-campus.glb"
    bpy.ops.object.select_all(action="DESELECT")
    for obj in campus.collection.objects:
        obj.select_set(bool(obj.get("aw_export")))
    bpy.ops.export_scene.gltf(
        filepath=str(out),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_cameras=False,
        export_lights=False,
        export_extras=True,
    )
    return out


def scene_manifest(campus: Campus) -> dict[str, object]:
    cam = campus.camera
    render = bpy.context.scene.render
    forward = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    up = cam.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    # The orbit target is where the camera's view axis meets the ground plane.
    target = cam.location + forward * (-cam.location.z / forward.z)
    return {
        "model": "/scenes/overview-campus.glb",
        "camera": {
            "position": to_gltf(cam.location),
            "target": to_gltf(target),
            "up": [round(up.x, 4), round(up.z, 4), round(-up.y, 4)],
            "orthoWidth": cam.data.ortho_scale,
            "shift": [round(cam.data.shift_x, 4), round(cam.data.shift_y, 4)],
            "aspect": round(render.resolution_x / render.resolution_y, 4),
        },
        "stages": {
            key: {"anchor": to_gltf(campus.anchors[key]), "center": to_gltf(campus.centers[key])}
            for key in STAGES
        },
        "roads": [
            {"id": road_id, "stage": stage_id, "points": [to_gltf(v) for v in pts[::2] + [pts[-1]]]}
            for road_id, stage_id, pts in campus.roads
        ],
    }


def main() -> dict[str, object]:
    render(DARK)
    campus = render(LIGHT)  # light last so the open file shows the approved look
    export_glb(campus)
    anchors_2d = {
        "image": {
            "widthPx": bpy.context.scene.render.resolution_x,
            "heightPx": bpy.context.scene.render.resolution_y,
        },
        "anchors": {
            key: {"x": round(v.x, 4), "y": round(v.y, 4)}
            for key, v in ((k, project(campus.camera, a)) for k, a in campus.anchors.items())
        },
    }
    (WEB_ROOT / "lib" / "overview-scene-anchors.json").write_text(
        json.dumps(anchors_2d, indent=2) + "\n"
    )
    manifest = scene_manifest(campus)
    (WEB_ROOT / "lib" / "overview-campus-3d.json").write_text(json.dumps(manifest) + "\n")
    return {"anchors": anchors_2d["anchors"], "roads": len(manifest["roads"])}


if __name__ == "__main__":
    main()
