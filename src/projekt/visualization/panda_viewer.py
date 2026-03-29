from __future__ import annotations

from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Filename,
    GeoMipTerrain,
    KeyboardButton,
    TextureStage,
    Vec3,
    WindowProperties,
    ClockObject
)


class PandaTerrainViewer(ShowBase):
    def __init__(
        self,
        heightmap_path: str | Path,
        texture_path: str | Path | None = None,
        z_scale: float = 60.0,
    ) -> None:
        super().__init__()

        self.heightmap_path = Path(heightmap_path)
        self.texture_path = Path(texture_path) if texture_path is not None else None
        self.z_scale = z_scale

        self._setup_window()
        self._setup_camera()
        self._setup_lights()
        self._setup_terrain()
        self._setup_controls()

        self.taskMgr.add(self.update_task, "update_task")

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Terrain AI - Panda3D Viewer")
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        self.setBackgroundColor(0.6, 0.8, 0.95, 1.0)

    def _setup_camera(self) -> None:
        self.disableMouse()

        self.camera_speed = 150.0
        self.mouse_sensitivity = 0.2

        self.camera.setPos(128, -300, 120)
        self.camera.lookAt(128, 128, 0)

        self.key_map = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.45, 0.45, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((0.9, 0.9, 0.85, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -50, 0)
        self.render.setLight(sun_np)

    def _setup_terrain(self) -> None:
        self.terrain = GeoMipTerrain("terrain")
        self.terrain.setHeightfield(
            Filename.from_os_specific(str(self.heightmap_path))
        )
        self.terrain.setBlockSize(32)
        self.terrain.setNear(500)
        self.terrain.setFar(1000)
        self.terrain.setFocalPoint(self.camera)

        self.terrain_root = self.terrain.getRoot()
        self.terrain_root.reparentTo(self.render)
        self.terrain_root.setSz(self.z_scale)

        if self.texture_path is not None and self.texture_path.exists():
            texture = self.loader.loadTexture(
                Filename.from_os_specific(str(self.texture_path))
            )
            self.terrain_root.setTexture(TextureStage.getDefault(), texture)

        self.terrain.generate()

    def _setup_controls(self) -> None:
        self.accept("w", self._set_key, ["forward", True])
        self.accept("w-up", self._set_key, ["forward", False])

        self.accept("s", self._set_key, ["backward", True])
        self.accept("s-up", self._set_key, ["backward", False])

        self.accept("a", self._set_key, ["left", True])
        self.accept("a-up", self._set_key, ["left", False])

        self.accept("d", self._set_key, ["right", True])
        self.accept("d-up", self._set_key, ["right", False])

        self.accept("q", self._set_key, ["down", True])
        self.accept("q-up", self._set_key, ["down", False])

        self.accept("e", self._set_key, ["up", True])
        self.accept("e-up", self._set_key, ["up", False])

        self.accept("escape", self.userExit)

    def _set_key(self, key: str, value: bool) -> None:
        self.key_map[key] = value

    def update_task(self, task: Task) -> int:
        dt = ClockObject.getGlobalClock().getDt()

        move = Vec3(0, 0, 0)

        if self.key_map["forward"]:
            move.y += 1
        if self.key_map["backward"]:
            move.y -= 1
        if self.key_map["left"]:
            move.x -= 1
        if self.key_map["right"]:
            move.x += 1
        if self.key_map["up"]:
            move.z += 1
        if self.key_map["down"]:
            move.z -= 1

        if move.length_squared() > 0:
            move.normalize()
            self.camera.setPos(self.camera, move * self.camera_speed * dt)

        self.terrain.update()
        return Task.cont