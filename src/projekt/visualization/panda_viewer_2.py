from __future__ import annotations

from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    Filename,
    GeoMipTerrain,
    TextureStage,
    Vec3,
    WindowProperties,
)


class PandaTerrainComparisonViewer(ShowBase):
    def __init__(
        self,
        heightmap_path_1: str | Path,
        heightmap_path_2: str | Path,
        texture_path_1: str | Path | None = None,
        texture_path_2: str | Path | None = None,
        z_scale: float = 40.0,
        terrain_spacing: float = 320.0,
    ) -> None:
        super().__init__()

        self.heightmap_path_1 = Path(heightmap_path_1)
        self.heightmap_path_2 = Path(heightmap_path_2)
        self.texture_path_1 = Path(texture_path_1) if texture_path_1 is not None else None
        self.texture_path_2 = Path(texture_path_2) if texture_path_2 is not None else None

        self.z_scale = z_scale
        self.terrain_spacing = terrain_spacing

        self._setup_window()
        self._setup_camera()
        self._setup_lights()
        self._setup_terrains()
        self._setup_controls()

        self.taskMgr.add(self.update_task, "update_task")

    def _setup_window(self) -> None:
        props = WindowProperties()
        props.setTitle("Terrain Comparison Viewer")
        props.setSize(1600, 900)
        self.win.requestProperties(props)
        self.setBackgroundColor(0.6, 0.8, 0.95, 1.0)

    def _setup_camera(self) -> None:
        self.disableMouse()
        self.camera_speed = 250.0

        center_x = self.terrain_spacing / 2.0
        self.camera.setPos(center_x, -700, 220)
        self.camera.lookAt(center_x, 128, 20)

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
        ambient.setColor((0.5, 0.5, 0.5, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((0.95, 0.95, 0.90, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -50, 0)
        self.render.setLight(sun_np)

    def _create_terrain(
        self,
        name: str,
        heightmap_path: Path,
        texture_path: Path | None,
        pos_x: float,
    ) -> GeoMipTerrain:
        terrain = GeoMipTerrain(name)
        terrain.setHeightfield(Filename.from_os_specific(str(heightmap_path)))
        terrain.setBruteforce(True)

        root = terrain.getRoot()
        root.reparentTo(self.render)
        root.setPos(pos_x, 0, 0)
        root.setSz(self.z_scale)

        if texture_path is not None and texture_path.exists():
            texture = self.loader.loadTexture(
                Filename.from_os_specific(str(texture_path))
            )
            root.setTexture(TextureStage.getDefault(), texture)

        terrain.generate()
        return terrain

    def _setup_terrains(self) -> None:
        self.terrain_1 = self._create_terrain(
            name="terrain_ground_truth",
            heightmap_path=self.heightmap_path_1,
            texture_path=self.texture_path_1,
            pos_x=0.0,
        )

        self.terrain_2 = self._create_terrain(
            name="terrain_prediction",
            heightmap_path=self.heightmap_path_2,
            texture_path=self.texture_path_2,
            pos_x=self.terrain_spacing,
        )

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

    def update_task(self, task: Task):
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

        self.terrain_1.update()
        self.terrain_2.update()

        return Task.cont