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