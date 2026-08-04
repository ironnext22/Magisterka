# from __future__ import annotations
#
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Final
#
# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont, ImageOps
#
# from panda3d.core import (
#     AmbientLight,
#     DirectionalLight,
#     Filename,
#     GeoMipTerrain,
#     Vec3,
#     Vec4,
#     loadPrcFileData,
# )
# from direct.showbase.ShowBase import ShowBase
#
# from projekt.generation.procedural_maps import (
#     ProceduralMapConfig,
#     generate_maps,
#     save_segmentation_png,
#     save_texture_png,
# )
# from projekt.models.unet import UNetTerrainModel
#
#
# # ---------------------------------------------------------------------------
# # Konfiguracja renderowania
# # ---------------------------------------------------------------------------
#
# WINDOW_WIDTH: Final[int] = 1400
# WINDOW_HEIGHT: Final[int] = 900
#
# PANEL_WIDTH: Final[int] = 900
# PANEL_HEIGHT: Final[int] = 580
# TITLE_HEIGHT: Final[int] = 54
# PANEL_GAP: Final[int] = 18
#
# Z_SCALE: Final[float] = 42.0
#
#
# @dataclass(frozen=True)
# class ExampleConfig:
#     """
#     Konfiguracja jednego przykładu do pracy.
#     """
#
#     name: str
#     seed: int
#     map_config: ProceduralMapConfig
#
#
# EXAMPLES: Final[list[ExampleConfig]] = [
#     ExampleConfig(
#         name="mountain_island",
#         seed=42,
#         map_config=ProceduralMapConfig(
#             width=256,
#             height=256,
#             sea_level=0.42,
#             beach_width=3,
#             grass_level=0.52,
#             rock_level=0.68,
#             snow_level=0.82,
#             continent_strength=0.72,
#             mountain_strength=0.45,
#         ),
#     ),
#     ExampleConfig(
#         name="low_archipelago",
#         seed=1337,
#         map_config=ProceduralMapConfig(
#             width=256,
#             height=256,
#             sea_level=0.50,
#             beach_width=4,
#             grass_level=0.57,
#             rock_level=0.74,
#             snow_level=0.89,
#             continent_strength=0.50,
#             mountain_strength=0.24,
#         ),
#     ),
# ]
#
#
# # ---------------------------------------------------------------------------
# # Obsługa danych modelu
# # ---------------------------------------------------------------------------
#
# def load_generated_input_tensor(
#     texture_path: Path,
#     segmentation_path: Path,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Wczytuje wygenerowaną teksturę i segmentację.
#
#     Zwraca tensor o wymiarach:
#     [1, 4, H, W]
#
#     Kanały:
#     - 3 kanały tekstury RGB,
#     - 1 kanał mapy segmentacji.
#     """
#     with Image.open(texture_path) as image:
#         texture = np.asarray(
#             image.convert("RGB"),
#             dtype=np.float32,
#         ) / 255.0
#
#     with Image.open(segmentation_path) as image:
#         segmentation = np.asarray(
#             image.convert("L"),
#             dtype=np.float32,
#         ) / 255.0
#
#     texture = np.transpose(texture, (2, 0, 1))
#     segmentation = np.expand_dims(segmentation, axis=0)
#
#     model_input = np.concatenate(
#         [texture, segmentation],
#         axis=0,
#     )
#
#     return (
#         torch.from_numpy(model_input)
#         .unsqueeze(0)
#         .float()
#         .to(device)
#     )
#
#
# def save_heightmap_png(
#     height_tensor: torch.Tensor,
#     output_path: Path,
# ) -> None:
#     """
#     Zapisuje mapę wysokości wygenerowaną przez model jako PNG.
#     """
#     if height_tensor.ndim == 3:
#         height_tensor = height_tensor[0]
#
#     heightmap = height_tensor.detach().cpu().numpy()
#     heightmap = np.clip(heightmap, 0.0, 1.0)
#
#     minimum = float(heightmap.min())
#     maximum = float(heightmap.max())
#
#     if maximum > minimum:
#         heightmap = (heightmap - minimum) / (maximum - minimum)
#     else:
#         heightmap = np.zeros_like(heightmap)
#
#     heightmap_uint8 = np.round(heightmap * 255.0).astype(np.uint8)
#
#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     Image.fromarray(
#         heightmap_uint8,
#         mode="L",
#     ).save(output_path)
#
#
# def load_model(
#     model_path: Path,
#     device: torch.device,
# ) -> UNetTerrainModel:
#     """
#     Wczytuje wytrenowany model U-Net.
#     """
#     if not model_path.exists():
#         raise FileNotFoundError(
#             f"Nie znaleziono pliku modelu:\n{model_path}"
#         )
#
#     model = UNetTerrainModel(
#         in_channels=4,
#         out_channels=1,
#     ).to(device)
#
#     state_dict = torch.load(
#         model_path,
#         map_location=device,
#     )
#
#     model.load_state_dict(state_dict)
#     model.eval()
#
#     return model
#
#
# # ---------------------------------------------------------------------------
# # Renderowanie pojedynczego terenu w Panda3D
# # ---------------------------------------------------------------------------
#
# class ThesisTerrainRenderer(ShowBase):
#     """
#     Prosty renderer Panda3D przeznaczony do automatycznego wykonywania
#     zrzutów pojedynczego modelu terenu.
#     """
#
#     def __init__(
#         self,
#         heightmap_path: Path,
#         texture_path: Path,
#         z_scale: float = Z_SCALE,
#     ) -> None:
#         super().__init__(windowType="offscreen")
#
#         self.disableMouse()
#
#         self.setBackgroundColor(
#             0.55,
#             0.76,
#             0.90,
#             1.0,
#         )
#
#         self.heightmap_path = heightmap_path
#         self.texture_path = texture_path
#         self.z_scale = z_scale
#
#         self.terrain_size = self._read_terrain_size()
#         self.terrain_center = Vec3(0.0, 0.0, z_scale * 0.23)
#
#         self._create_terrain()
#         self._create_lighting()
#         self._configure_lens()
#
#     def _read_terrain_size(self) -> int:
#         with Image.open(self.heightmap_path) as image:
#             width, height = image.size
#
#         if width != height:
#             raise ValueError(
#                 "Mapa wysokości powinna być kwadratowa. "
#                 f"Otrzymano: {width}x{height}"
#             )
#
#         return width
#
#     def _create_terrain(self) -> None:
#         terrain = GeoMipTerrain("generated-terrain")
#
#         terrain.setHeightfield(
#             Filename.fromOsSpecific(
#                 str(self.heightmap_path.resolve())
#             )
#         )
#
#         terrain.setBlockSize(32)
#         terrain.setNear(40)
#         terrain.setFar(200)
#         terrain.setFocalPoint(self.camera)
#
#         # Pełna geometria jest lepsza do statycznych zrzutów.
#         terrain.setBruteforce(True)
#         terrain.generate()
#
#         terrain_root = terrain.getRoot()
#         terrain_root.reparentTo(self.render)
#
#         terrain_root.setScale(
#             1.0,
#             1.0,
#             self.z_scale,
#         )
#
#         half_size = (self.terrain_size - 1) / 2.0
#
#         terrain_root.setPos(
#             -half_size,
#             -half_size,
#             0.0,
#         )
#
#         texture = self.loader.loadTexture(
#             Filename.fromOsSpecific(
#                 str(self.texture_path.resolve())
#             )
#         )
#
#         if texture is None:
#             raise RuntimeError(
#                 f"Nie udało się wczytać tekstury: {self.texture_path}"
#             )
#
#         texture.setMinfilter(texture.FTLinearMipmapLinear)
#         texture.setMagfilter(texture.FTLinear)
#
#         terrain_root.setTexture(texture, 1)
#
#         self.terrain = terrain
#         self.terrain_root = terrain_root
#
#     def _create_lighting(self) -> None:
#         ambient = AmbientLight("ambient-light")
#         ambient.setColor(Vec4(0.52, 0.52, 0.52, 1.0))
#
#         ambient_node = self.render.attachNewNode(ambient)
#         self.render.setLight(ambient_node)
#
#         sun = DirectionalLight("sun-light")
#         sun.setColor(Vec4(1.0, 0.97, 0.90, 1.0))
#
#         sun_node = self.render.attachNewNode(sun)
#         sun_node.setHpr(-35.0, -55.0, 0.0)
#
#         self.render.setLight(sun_node)
#         self.render.setShaderAuto()
#
#     def _configure_lens(self) -> None:
#         self.camLens.setFov(48.0)
#         self.camLens.setNearFar(1.0, 3000.0)
#
#     def set_angled_camera(self) -> None:
#         """
#         Ustawia kamerę w perspektywie ukośnej.
#         """
#         distance = self.terrain_size * 1.18
#
#         self.camera.setPos(
#             distance,
#             -distance,
#             self.terrain_size * 0.72,
#         )
#
#         self.camera.lookAt(self.terrain_center)
#
#     def set_top_camera(self) -> None:
#         """
#         Ustawia kamerę niemal dokładnie nad terenem.
#         """
#         self.camera.setPos(
#             0.0,
#             -0.5,
#             self.terrain_size * 1.60,
#         )
#
#         self.camera.lookAt(
#             0.0,
#             0.0,
#             0.0,
#         )
#
#     def save_view(
#         self,
#         output_path: Path,
#     ) -> None:
#         """
#         Renderuje kilka klatek i zapisuje aktualny widok.
#         """
#         output_path.parent.mkdir(
#             parents=True,
#             exist_ok=True,
#         )
#
#         # Kilka klatek pozwala Panda3D zakończyć ładowanie tekstury.
#         for _ in range(6):
#             self.graphicsEngine.renderFrame()
#
#         success = self.win.saveScreenshot(
#             Filename.fromOsSpecific(
#                 str(output_path.resolve())
#             )
#         )
#
#         if not success:
#             raise RuntimeError(
#                 f"Nie udało się zapisać zrzutu: {output_path}"
#             )
#
#
# def render_panda_views(
#     heightmap_path: Path,
#     texture_path: Path,
#     angled_output_path: Path,
#     top_output_path: Path,
# ) -> None:
#     """
#     Tworzy dwa zrzuty tego samego świata:
#     - widok pod kątem,
#     - widok z góry.
#     """
#     renderer = ThesisTerrainRenderer(
#         heightmap_path=heightmap_path,
#         texture_path=texture_path,
#     )
#
#     try:
#         renderer.set_angled_camera()
#         renderer.save_view(angled_output_path)
#
#         renderer.set_top_camera()
#         renderer.save_view(top_output_path)
#
#     finally:
#         renderer.destroy()
#
#
# # ---------------------------------------------------------------------------
# # Składanie gotowej ilustracji
# # ---------------------------------------------------------------------------
#
# def load_label_font(size: int) -> ImageFont.ImageFont:
#     """
#     Próbuje wczytać popularną czcionkę systemową.
#     """
#     possible_fonts = [
#         Path("C:/Windows/Fonts/arial.ttf"),
#         Path("C:/Windows/Fonts/calibri.ttf"),
#         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
#     ]
#
#     for font_path in possible_fonts:
#         if font_path.exists():
#             return ImageFont.truetype(
#                 str(font_path),
#                 size=size,
#             )
#
#     return ImageFont.load_default()
#
#
# def create_panel(
#     image_path: Path,
#     title: str,
#     resize_method: Image.Resampling,
# ) -> Image.Image:
#     """
#     Tworzy pojedynczy podpisany panel.
#     """
#     with Image.open(image_path) as image:
#         source = image.convert("RGB")
#
#     fitted = ImageOps.fit(
#         source,
#         (PANEL_WIDTH, PANEL_HEIGHT),
#         method=resize_method,
#         centering=(0.5, 0.5),
#     )
#
#     panel = Image.new(
#         "RGB",
#         (
#             PANEL_WIDTH,
#             PANEL_HEIGHT + TITLE_HEIGHT,
#         ),
#         "white",
#     )
#
#     panel.paste(
#         fitted,
#         (0, TITLE_HEIGHT),
#     )
#
#     draw = ImageDraw.Draw(panel)
#     font = load_label_font(28)
#
#     bounding_box = draw.textbbox(
#         (0, 0),
#         title,
#         font=font,
#     )
#
#     text_width = bounding_box[2] - bounding_box[0]
#     text_height = bounding_box[3] - bounding_box[1]
#
#     text_x = (PANEL_WIDTH - text_width) // 2
#     text_y = (TITLE_HEIGHT - text_height) // 2 - 2
#
#     draw.text(
#         (text_x, text_y),
#         title,
#         fill="black",
#         font=font,
#     )
#
#     return panel
#
#
# def create_three_panel_figure(
#     segmentation_path: Path,
#     angled_view_path: Path,
#     top_view_path: Path,
#     output_path: Path,
# ) -> None:
#     """
#     Łączy:
#     - mapę segmentacji,
#     - widok Panda3D pod kątem,
#     - widok Panda3D z góry.
#     """
#     segmentation_panel = create_panel(
#         segmentation_path,
#         "Mapa segmentacji semantycznej",
#         Image.Resampling.NEAREST,
#     )
#
#     angled_panel = create_panel(
#         angled_view_path,
#         "Widok modelu pod kątem",
#         Image.Resampling.LANCZOS,
#     )
#
#     top_panel = create_panel(
#         top_view_path,
#         "Widok modelu z góry",
#         Image.Resampling.LANCZOS,
#     )
#
#     total_width = (
#         PANEL_WIDTH * 3
#         + PANEL_GAP * 2
#     )
#
#     total_height = PANEL_HEIGHT + TITLE_HEIGHT
#
#     final_image = Image.new(
#         "RGB",
#         (
#             total_width,
#             total_height,
#         ),
#         "white",
#     )
#
#     final_image.paste(
#         segmentation_panel,
#         (0, 0),
#     )
#
#     final_image.paste(
#         angled_panel,
#         (
#             PANEL_WIDTH + PANEL_GAP,
#             0,
#         ),
#     )
#
#     final_image.paste(
#         top_panel,
#         (
#             2 * (PANEL_WIDTH + PANEL_GAP),
#             0,
#         ),
#     )
#
#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     final_image.save(
#         output_path,
#         format="PNG",
#         optimize=True,
#     )
#
#
# # ---------------------------------------------------------------------------
# # Generowanie jednego przykładu
# # ---------------------------------------------------------------------------
#
# def generate_example(
#     example: ExampleConfig,
#     model: UNetTerrainModel,
#     device: torch.device,
#     outputs_root: Path,
# ) -> Path:
#     """
#     Generuje wszystkie dane i finalną ilustrację dla jednego przykładu.
#     """
#     generated_inputs_dir = outputs_root / "thesis_inputs"
#     predictions_dir = outputs_root / "thesis_predictions"
#     screenshots_dir = outputs_root / "thesis_screenshots"
#     figures_dir = outputs_root / "thesis_figures"
#
#     print(
#         f"\nGenerowanie przykładu: {example.name}, "
#         f"seed={example.seed}"
#     )
#
#     _, segmentation_map, texture_map = generate_maps(
#         seed=example.seed,
#         config=example.map_config,
#     )
#
#     texture_path = (
#         generated_inputs_dir
#         / f"{example.name}_texture.png"
#     )
#
#     segmentation_path = (
#         generated_inputs_dir
#         / f"{example.name}_segmentation.png"
#     )
#
#     predicted_heightmap_path = (
#         predictions_dir
#         / f"{example.name}_predicted_heightmap.png"
#     )
#
#     angled_screenshot_path = (
#         screenshots_dir
#         / f"{example.name}_angled.png"
#     )
#
#     top_screenshot_path = (
#         screenshots_dir
#         / f"{example.name}_top.png"
#     )
#
#     final_figure_path = (
#         figures_dir
#         / f"{example.name}_thesis_figure.png"
#     )
#
#     save_texture_png(
#         texture_map,
#         texture_path,
#     )
#
#     save_segmentation_png(
#         segmentation_map,
#         segmentation_path,
#     )
#
#     model_input = load_generated_input_tensor(
#         texture_path=texture_path,
#         segmentation_path=segmentation_path,
#         device=device,
#     )
#
#     with torch.no_grad():
#         prediction = model(model_input)
#         prediction = torch.clamp(
#             prediction,
#             min=0.0,
#             max=1.0,
#         )[0]
#
#     save_heightmap_png(
#         prediction,
#         predicted_heightmap_path,
#     )
#
#     render_panda_views(
#         heightmap_path=predicted_heightmap_path,
#         texture_path=texture_path,
#         angled_output_path=angled_screenshot_path,
#         top_output_path=top_screenshot_path,
#     )
#
#     create_three_panel_figure(
#         segmentation_path=segmentation_path,
#         angled_view_path=angled_screenshot_path,
#         top_view_path=top_screenshot_path,
#         output_path=final_figure_path,
#     )
#
#     print(f"Segmentacja: {segmentation_path}")
#     print(f"Heightmapa:   {predicted_heightmap_path}")
#     print(f"Widok ukośny: {angled_screenshot_path}")
#     print(f"Widok z góry: {top_screenshot_path}")
#     print(f"Gotowy rysunek: {final_figure_path}")
#
#     return final_figure_path
#
#
# # ---------------------------------------------------------------------------
# # Główna funkcja
# # ---------------------------------------------------------------------------
#
# def main() -> None:
#     # Ustawienia okna należy przekazać przed utworzeniem ShowBase.
#     loadPrcFileData(
#         "",
#         f"""
#         window-type offscreen
#         win-size {WINDOW_WIDTH} {WINDOW_HEIGHT}
#         framebuffer-multisample 1
#         multisamples 4
#         sync-video 0
#         show-frame-rate-meter 0
#         texture-anisotropic-degree 8
#         """,
#     )
#
#     # Jeżeli plik znajduje się w katalogu scripts/, parents[1]
#     # powinno wskazywać katalog główny projektu.
#     project_root = Path(__file__).resolve().parents[1]
#     outputs_root = project_root / "outputs"
#
#     model_path = (
#         outputs_root
#         / "unet_segmentation"
#         / "unet_segmentation.pt"
#     )
#
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#
#     print(f"Urządzenie obliczeniowe: {device}")
#
#     model = load_model(
#         model_path=model_path,
#         device=device,
#     )
#
#     generated_figures: list[Path] = []
#
#     for example in EXAMPLES:
#         figure_path = generate_example(
#             example=example,
#             model=model,
#             device=device,
#             outputs_root=outputs_root,
#         )
#
#         generated_figures.append(figure_path)
#
#     print("\nUtworzono gotowe ilustracje:")
#
#     for path in generated_figures:
#         print(path)
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from panda3d.core import Filename

from projekt.generation.procedural_maps import (
    ProceduralMapConfig,
    generate_maps,
    save_segmentation_png,
    save_texture_png,
)
from projekt.models.unet import UNetTerrainModel
from projekt.visualization.panda_viewer_2 import (
    PandaTerrainComparisonViewer,
)


def load_generated_input_tensor(
    texture_path: Path,
    segmentation_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Tworzy tensor wejściowy [1, 4, H, W]:

    - kanały 0-2: tekstura RGB,
    - kanał 3: segmentacja w skali szarości.
    """
    with Image.open(texture_path) as image:
        texture = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    with Image.open(segmentation_path) as image:
        segmentation = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        ) / 255.0

    texture = np.transpose(texture, (2, 0, 1))
    segmentation = np.expand_dims(segmentation, axis=0)

    model_input = np.concatenate(
        [texture, segmentation],
        axis=0,
    )

    return (
        torch.from_numpy(model_input)
        .unsqueeze(0)
        .float()
        .to(device)
    )


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalizuje tablicę do zakresu 0-1.
    """
    array = np.asarray(array, dtype=np.float32)

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)

    return (array - minimum) / (maximum - minimum)


def save_numpy_heightmap(
    heightmap: np.ndarray,
    output_path: Path,
) -> None:
    """
    Zapisuje referencyjną mapę wysokości jako 8-bitowy PNG.
    """
    normalized = normalize_array(heightmap)
    image_array = np.round(normalized * 255.0).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image_array,
        mode="L",
    ).save(output_path)


def save_tensor_heightmap(
    height_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """
    Zapisuje mapę wysokości wygenerowaną przez U-Net.
    """
    if height_tensor.ndim == 3:
        height_tensor = height_tensor[0]

    heightmap = height_tensor.detach().cpu().numpy()
    heightmap = np.clip(heightmap, 0.0, 1.0)
    heightmap = normalize_array(heightmap)

    image_array = np.round(heightmap * 255.0).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image_array,
        mode="L",
    ).save(output_path)


def load_model(
    model_path: Path,
    device: torch.device,
) -> UNetTerrainModel:
    """
    Wczytuje wytrenowany model U-Net.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu:\n{model_path}"
        )

    model = UNetTerrainModel(
        in_channels=4,
        out_channels=1,
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_font(size: int) -> ImageFont.ImageFont:
    """
    Wczytuje czcionkę dostępną w systemie.
    """
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


def add_comparison_labels(
    screenshot_path: Path,
    output_path: Path,
) -> None:
    """
    Dodaje nad terenami podpisy:

    - teren referencyjny,
    - teren wygenerowany przez U-Net.
    """
    with Image.open(screenshot_path) as image:
        screenshot = image.convert("RGB")

    # Usuwa część nadmiarowego nieba.
    # W razie potrzeby zmień wartości 0.18 i 0.92.
    width, height = screenshot.size

    crop_top = int(height * 0.18)
    crop_bottom = int(height * 0.92)

    screenshot = screenshot.crop(
        (
            0,
            crop_top,
            width,
            crop_bottom,
        )
    )

    header_height = 90

    final_image = Image.new(
        "RGB",
        (
            screenshot.width,
            screenshot.height + header_height,
        ),
        "white",
    )

    final_image.paste(
        screenshot,
        (0, header_height),
    )

    draw = ImageDraw.Draw(final_image)
    font = get_font(34)

    left_text = "Teren referencyjny"
    right_text = "Teren wygenerowany przez U-Net"

    left_center = screenshot.width * 0.30
    right_center = screenshot.width * 0.72

    def draw_centered_text(
        text: str,
        center_x: float,
    ) -> None:
        box = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = box[2] - box[0]
        text_height = box[3] - box[1]

        draw.text(
            (
                center_x - text_width / 2,
                (header_height - text_height) / 2 - 3,
            ),
            text,
            fill="black",
            font=font,
        )

    draw_centered_text(
        left_text,
        left_center,
    )

    draw_centered_text(
        right_text,
        right_center,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_image.save(
        output_path,
        format="PNG",
        optimize=True,
    )


def run_comparison_viewer(
    reference_heightmap_path: Path,
    predicted_heightmap_path: Path,
    texture_path: Path,
    raw_screenshot_path: Path,
) -> None:
    """
    Wyświetla teren referencyjny i predykcję obok siebie,
    wykonuje zrzut ekranu i zamyka aplikację.
    """
    raw_screenshot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    app = PandaTerrainComparisonViewer(
        heightmap_path_1=reference_heightmap_path,
        heightmap_path_2=predicted_heightmap_path,
        texture_path_1=texture_path,
        texture_path_2=texture_path,
        z_scale=40.0,
        terrain_spacing=310.0,
    )

    def save_screenshot(task):
        # Kilka dodatkowych klatek pozwala dokończyć
        # ładowanie geometrii i tekstur.
        for _ in range(5):
            app.graphicsEngine.renderFrame()

        panda_filename = Filename.fromOsSpecific(
            str(raw_screenshot_path.resolve())
        )

        success = app.win.saveScreenshot(
            panda_filename
        )

        if not success:
            raise RuntimeError(
                f"Nie udało się zapisać zrzutu: "
                f"{raw_screenshot_path}"
            )

        print(
            f"Zapisano surowy zrzut: "
            f"{raw_screenshot_path}"
        )

        app.userExit()
        return task.done

    app.taskMgr.doMethodLater(
        2.0,
        save_screenshot,
        "save-reference-prediction-comparison",
    )

    app.run()


def main() -> None:
    # Jeżeli skrypt znajduje się w katalogu scripts/.
    project_root = Path(__file__).resolve().parents[1]

    outputs_root = project_root / "outputs"

    generated_dir = (
        outputs_root
        / "comparison_inputs"
    )

    prediction_dir = (
        outputs_root
        / "comparison_predictions"
    )

    screenshot_dir = (
        outputs_root
        / "comparison_screenshots"
    )

    figure_dir = (
        outputs_root
        / "thesis_figures"
    )

    model_path = (
        outputs_root
        / "unet_segmentation"
        / "unet_segmentation.pt"
    )

    seed = 42

    config = ProceduralMapConfig(
        width=256,
        height=256,
        sea_level=0.42,
        beach_width=3,
        grass_level=0.52,
        rock_level=0.68,
        snow_level=0.82,
        continent_strength=0.65,
        mountain_strength=0.35,
    )

    print(
        f"Generowanie przykładu porównawczego "
        f"dla seed={seed}..."
    )

    reference_heightmap, segmentation_map, texture_map = (
        generate_maps(
            seed=seed,
            config=config,
        )
    )

    texture_path = (
        generated_dir
        / f"texture_seed_{seed}.png"
    )

    segmentation_path = (
        generated_dir
        / f"segmentation_seed_{seed}.png"
    )

    reference_heightmap_path = (
        generated_dir
        / f"reference_heightmap_seed_{seed}.png"
    )

    predicted_heightmap_path = (
        prediction_dir
        / f"predicted_heightmap_seed_{seed}.png"
    )

    raw_screenshot_path = (
        screenshot_dir
        / f"comparison_raw_seed_{seed}.png"
    )

    final_figure_path = (
        figure_dir
        / f"reference_vs_unet_seed_{seed}.png"
    )

    save_texture_png(
        texture_map,
        texture_path,
    )

    save_segmentation_png(
        segmentation_map,
        segmentation_path,
    )

    save_numpy_heightmap(
        reference_heightmap,
        reference_heightmap_path,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Urządzenie: {device}")

    model = load_model(
        model_path=model_path,
        device=device,
    )

    model_input = load_generated_input_tensor(
        texture_path=texture_path,
        segmentation_path=segmentation_path,
        device=device,
    )

    with torch.no_grad():
        prediction = model(model_input)

        prediction = torch.clamp(
            prediction,
            min=0.0,
            max=1.0,
        )[0]

    save_tensor_heightmap(
        prediction,
        predicted_heightmap_path,
    )

    print(
        f"Referencyjna mapa wysokości: "
        f"{reference_heightmap_path}"
    )

    print(
        f"Predykcja U-Net: "
        f"{predicted_heightmap_path}"
    )

    run_comparison_viewer(
        reference_heightmap_path=reference_heightmap_path,
        predicted_heightmap_path=predicted_heightmap_path,
        texture_path=texture_path,
        raw_screenshot_path=raw_screenshot_path,
    )

    if not raw_screenshot_path.exists():
        raise FileNotFoundError(
            f"Nie utworzono zrzutu:\n"
            f"{raw_screenshot_path}"
        )

    add_comparison_labels(
        screenshot_path=raw_screenshot_path,
        output_path=final_figure_path,
    )

    print(
        f"\nGotowy rysunek do pracy:\n"
        f"{final_figure_path}"
    )


if __name__ == "__main__":
    main()