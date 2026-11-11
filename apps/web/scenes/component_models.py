"""Build the Overview's close-up component models, one GLB per pipeline stage.

Clicking a building on the Overview zooms the camera to it and swaps the
building for one of these models: a GPU card for the model runtime, a CPU
board for the CPU tool, a database and vector index for retrieval, a network
router for the gateway and an output store for completion. They reuse the
campus materials, so the page tints them the same way.

Conventions the page relies on (components/widgets/campus-canvas.tsx):

- nodes named ``spin_*`` rotate continuously; ``extras.aw_spin`` is the speed
  in rad/s and ``extras.aw_spin_axis`` the glTF axis ("x", "y" or "z");
- meshes named ``blink_*`` pulse their glow, staggered by ``extras.aw_phase``;
- materials named ``aw_glow_<stage>`` take the stage's status colour.

The motion is decorative; only colour carries status. Run inside Blender 5.2+:

    blender --background --python apps/web/scenes/component_models.py

Outputs, relative to ``apps/web``: ``public/scenes/component-<stage>.glb`` and
``lib/overview-component-models.json`` (packet paths in glTF metres).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
COLLECTION_NAME = "AgentWatch Component Models"
SPIN_FAST = 9.0  # rad/s, GPU fans
SPIN_COOLER = 6.0
SPIN_SLOW = 1.2


def _campus():
    """The campus generator, for its Builder, materials and helpers."""
    name = "aw_overview_campus"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "overview_campus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campus = _campus()


def reset_collection() -> bpy.types.Collection:
    existing = bpy.data.collections.get(COLLECTION_NAME)
    if existing is not None:
        for obj in list(existing.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing)
    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    return collection


class Parts:
    """Builder helpers for animated parts on top of the campus Builder."""

    def __init__(self, b, stage_id: str) -> None:
        self.b = b
        self.stage_id = stage_id
        b.glow = b.glows[stage_id]
        self.blinks = 0

    def spin(self, name: str, center: tuple[float, float, float], speed: float, axis: str):
        """An empty that the page rotates; children placed relative to it spin with it."""
        empty = bpy.data.objects.new(f"spin_{name}", None)
        empty.location = center
        empty["aw_spin"] = speed
        empty["aw_spin_axis"] = axis
        return self.b.place(empty)

    def child(self, obj, parent) -> None:
        obj.parent = parent

    def blink(self, name: str, center, size) -> None:
        obj = self.b.box(f"blink_{name}", center, size, self.b.glow, 0)
        obj["aw_phase"] = round((self.blinks * 0.37) % 1.0, 3)
        self.blinks += 1

    def fan(self, name: str, center, radius: float, axis: str, speed: float, blades: int = 7):
        """Fan in the XY plane (axis "y" in glTF) or the XZ plane (axis "z" in glTF)."""
        b = self.b
        flat = axis == "y"
        well_center = center
        well = b.cylinder(f"{name}_well", well_center, radius, 0.04, b.recess)
        if not flat:
            well.rotation_euler = (math.radians(90), 0, 0)
        hub_spin = self.spin(name, center, speed, axis)
        hub = b.cylinder(f"{name}_hub", (0, 0, 0), radius * 0.26, 0.07, b.shade)
        self.child(hub, hub_spin)
        if not flat:
            hub.rotation_euler = (math.radians(90), 0, 0)
        for i in range(blades):
            angle = i * 2 * math.pi / blades
            r = radius * 0.58
            if flat:
                blade = b.box(
                    f"{name}_blade_{i}",
                    (r * math.cos(angle), r * math.sin(angle), 0.02),
                    (radius * 0.62, radius * 0.2, 0.015),
                    b.white,
                    0,
                )
                blade.rotation_euler = (0.35, 0, angle)
            else:
                blade = b.box(
                    f"{name}_blade_{i}",
                    (r * math.cos(angle), -0.03, r * math.sin(angle)),
                    (radius * 0.62, 0.015, radius * 0.2),
                    b.white,
                    0,
                )
                blade.rotation_euler = (0, -angle, 0.35)
            self.child(blade, hub_spin)

    def dash_ring(self, name: str, center, radius: float, speed: float, dashes: int = 18) -> None:
        """A dashed glowing ring in the XY plane that rotates (a scan or signal ring)."""
        ring = self.spin(name, center, speed, "y")
        for i in range(dashes):
            angle = i * 2 * math.pi / dashes
            dash = self.b.box(
                f"{name}_dash_{i}",
                (radius * math.cos(angle), radius * math.sin(angle), 0),
                (0.22, 0.05, 0.03),
                self.b.glow,
                0,
            )
            dash.rotation_euler = (0, 0, angle + math.pi / 2)
            self.child(dash, ring)


def path(points: list[tuple[float, float, float]]) -> list[list[float]]:
    pts = campus.catmull_rom([Vector(p) for p in points], samples_per_segment=10)
    return [campus.to_gltf(v) for v in pts]


# ------------------------------------------------------------------ models


def gpu(p: Parts) -> list[list[list[float]]]:
    b = p.b
    campus.podium(b, "gpu", 0, 0, 3.8, 2.6)
    for x in (-1.15, 1.15):
        b.box(f"gpu_stand_{x}", (x, 0, 0.5), (0.24, 1.0, 0.6), b.shade, 0.03)
    b.box("gpu_backplate", (0, 0, 0.82), (3.2, 1.35, 0.06), b.shade, 0.02)
    b.box("gpu_shroud", (0, 0, 1.02), (3.2, 1.35, 0.34), b.white, 0.06)
    for i in range(24):
        b.box(f"gpu_fin_{i}", (-1.45 + i * 0.126, -0.69, 0.96), (0.03, 0.02, 0.22), b.recess, 0)
    b.box("gpu_bracket", (1.66, 0, 1.0), (0.06, 1.35, 0.7), b.shade, 0.01)
    b.box("gpu_led_strip", (0, -0.69, 1.16), (2.9, 0.03, 0.05), b.glow, 0)
    for x in (-1.0, 0.0, 1.0):
        p.fan(f"gpu_fan_{int(x)}", (x, 0, 1.2), 0.44, "y", SPIN_FAST)
    p.blink("gpu_power", (1.7, -0.4, 1.2), (0.03, 0.1, 0.05))
    p.blink("gpu_link", (1.7, 0.4, 1.2), (0.03, 0.1, 0.05))
    # Work arriving over PCIe from the left of the plinth.
    return [
        path([(-1.85, -1.05, 0.26), (-1.3, -0.9, 0.3), (-1.15, -0.4, 0.55), (-0.6, -0.3, 1.22)])
    ]


def cpu(p: Parts) -> list[list[list[float]]]:
    b = p.b
    campus.podium(b, "cpu", 0, 0, 3.8, 3.2)
    b.box("cpu_board", (0, 0, 0.27), (3.2, 2.7, 0.08), b.shade, 0.02)
    b.box("cpu_socket", (-0.45, -0.2, 0.34), (1.1, 1.1, 0.07), b.recess, 0.01)
    b.box("cpu_ihs", (-0.45, -0.2, 0.4), (0.92, 0.92, 0.05), b.white, 0.02)
    for row in range(4):
        for col in range(4):
            p.blink(
                f"cpu_core_{row}_{col}",
                (-0.78 + col * 0.22, -0.53 + row * 0.22, 0.44),
                (0.16, 0.16, 0.02),
            )
    for i in range(4):
        x = -1.35 + i * 0.16
        b.box(f"cpu_ram_{i}", (x, 0.75, 0.55), (0.07, 1.1, 0.48), b.white, 0.01)
        p.blink(f"cpu_ram_led_{i}", (x, 0.75, 0.8), (0.07, 1.0, 0.02))
    # Tower cooler beside the chip, its fan facing the camera.
    for i in range(14):
        b.box(f"cpu_cooler_fin_{i}", (0.95, 0.45, 0.5 + i * 0.09), (0.9, 0.7, 0.025), b.white, 0)
    for x in (0.7, 0.95, 1.2):
        b.cylinder(f"cpu_heatpipe_{x}", (x, 0.45, 1.05), 0.035, 1.3, b.shade)
    b.box("cpu_cooler_frame", (0.95, 0.08, 1.07), (0.95, 0.05, 0.95), b.shade, 0.02)
    p.fan("cpu_cooler_fan", (0.95, 0.02, 1.07), 0.4, "z", SPIN_COOLER)
    b.box("cpu_nic", (1.45, -1.0, 0.42), (0.3, 0.5, 0.22), b.white, 0.02)
    for i in range(2):
        b.box(f"cpu_nic_port_{i}", (1.61, -1.1 + i * 0.2, 0.42), (0.02, 0.14, 0.12), b.recess, 0)
    # Network and memory traffic across the board.
    return [
        path(
            [
                (1.85, -1.1, 0.33),
                (1.45, -1.0, 0.33),
                (0.6, -0.9, 0.33),
                (-0.45, -0.75, 0.33),
                (-0.45, -0.2, 0.47),
            ]
        ),
        path([(-0.45, -0.2, 0.47), (-0.9, 0.25, 0.33), (-1.15, 0.6, 0.33), (-1.15, 0.75, 0.82)]),
        path([(-0.45, -0.2, 0.47), (0.2, 0.0, 0.33), (0.7, 0.45, 0.33), (0.95, 0.45, 0.5)]),
    ]


def retrieval(p: Parts) -> list[list[list[float]]]:
    b = p.b
    campus.podium(b, "retrieval", 0, 0, 3.6, 3.0)
    for i in range(3):
        z = 0.5 + i * 0.5
        b.cylinder(f"db_tier_{i}", (-0.6, 0.15, z), 0.72, 0.4, b.white)
        b.cylinder(f"db_band_{i}", (-0.6, 0.15, z + 0.22), 0.735, 0.04, b.recess)
        p.blink(f"db_led_{i}", (-0.6, -0.58, z), (0.22, 0.03, 0.05))
    p.dash_ring("db_scan", (-0.6, 0.15, 1.92), 0.95, SPIN_SLOW)
    b.box("index_rack", (0.95, 0.15, 0.95), (0.9, 1.2, 1.45), b.white, 0.05)
    for i in range(6):
        b.box(f"index_slot_{i}", (0.95, -0.46, 0.45 + i * 0.2), (0.7, 0.03, 0.12), b.recess, 0)
        p.blink(f"index_led_{i}", (1.24, -0.48, 0.45 + i * 0.2), (0.06, 0.02, 0.05))
    return [
        path([(-1.7, -1.3, 0.26), (-1.2, -1.0, 0.3), (-0.6, -0.65, 0.4)]),
        path([(-0.6, 0.15, 1.75), (0.1, 0.1, 2.0), (0.6, 0.15, 1.85), (0.95, 0.15, 1.7)]),
        path([(0.95, -0.5, 0.3), (0.3, -1.0, 0.27), (-0.4, -0.9, 0.3)]),
    ]


def gateway(p: Parts) -> list[list[list[float]]]:
    b = p.b
    campus.podium(b, "gateway", 0, 0, 3.6, 2.8)
    b.box("router_body", (0, 0.1, 0.55), (2.7, 1.4, 0.5), b.white, 0.08)
    b.box("router_top", (0, 0.1, 0.83), (2.5, 1.2, 0.06), b.shade, 0.02)
    for i in range(8):
        x = -1.05 + i * 0.3
        b.box(f"router_port_{i}", (x, -0.61, 0.5), (0.2, 0.03, 0.16), b.recess, 0)
        p.blink(f"router_port_led_{i}", (x, -0.62, 0.66), (0.08, 0.02, 0.04))
    for x, tilt in ((-0.9, -0.25), (0.0, 0.0), (0.9, 0.25)):
        antenna = b.cylinder(f"router_antenna_{x}", (x, 0.65, 1.3), 0.05, 1.0, b.shade)
        antenna.rotation_euler = (0.15, tilt, 0)
    p.dash_ring("router_signal", (0, 0.65, 1.95), 0.55, SPIN_SLOW * 1.5, dashes=14)
    b.box("router_status", (0, -0.62, 0.78), (2.4, 0.02, 0.04), b.glow, 0)
    return [
        path([(-1.8, -1.2, 0.26), (-1.3, -1.0, 0.3), (-0.75, -0.8, 0.45), (-0.75, -0.63, 0.5)]),
        path([(0.75, -0.63, 0.5), (0.9, -0.85, 0.35), (1.4, -1.0, 0.27), (1.8, -1.15, 0.26)]),
    ]


def complete(p: Parts) -> list[list[list[float]]]:
    b = p.b
    campus.podium(b, "complete", 0, 0, 3.2, 2.8)
    b.box("store_rack", (0.1, 0.15, 0.95), (1.7, 1.25, 1.5), b.white, 0.06)
    for row in range(3):
        for col in range(2):
            x, z = -0.32 + col * 0.84, 0.55 + row * 0.42
            b.box(f"store_bay_{row}_{col}", (x, -0.48, z), (0.72, 0.03, 0.3), b.recess, 0)
            p.blink(f"store_led_{row}_{col}", (x + 0.28, -0.5, z), (0.06, 0.02, 0.06))
    emblem = p.spin("store_check", (0.1, 0.15, 2.15), SPIN_SLOW, "y")
    for name, center, size, angle in (
        # A tick in the vertical XZ plane: the short stroke falls to the joint
        # at (-0.07, -0.13) and the long stroke rises from it.
        ("short", (-0.2, 0, 0.0), (0.37, 0.1, 0.1), 45),
        ("long", (0.18, 0, 0.12), (0.7, 0.1, 0.1), -45),
    ):
        stroke = b.box(f"store_check_{name}", center, size, b.glow, 0.02)
        stroke.rotation_euler = (0, math.radians(angle), 0)
        stroke.parent = emblem
    return [
        path([(-1.55, -1.2, 0.26), (-1.0, -0.95, 0.3), (-0.6, -0.65, 0.5), (-0.32, -0.52, 0.55)])
    ]


MODELS = {
    "model": gpu,
    "cpuTool": cpu,
    "retrieval": retrieval,
    "gateway": gateway,
    "complete": complete,
}


def export(collection: bpy.types.Collection, out: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in collection.objects:
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


def main() -> dict[str, object]:
    manifest: dict[str, object] = {}
    for stage_id, build in MODELS.items():
        collection = reset_collection()
        b = campus.Builder(collection, campus.LIGHT)
        paths = build(Parts(b, stage_id))
        bpy.context.view_layer.update()
        url = f"/scenes/component-{stage_id}.glb"
        export(collection, WEB_ROOT / "public" / url.lstrip("/"))
        manifest[stage_id] = {"model": url, "packetPaths": paths}
    out = WEB_ROOT / "lib" / "overview-component-models.json"
    out.write_text(json.dumps(manifest) + "\n")
    return {stage: len(entry["packetPaths"]) for stage, entry in manifest.items()}


if __name__ == "__main__":
    main()
