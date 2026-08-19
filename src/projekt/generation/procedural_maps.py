from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

def slope_limit(height, max_diff=0.05):

    h = height.copy()
    H, W = h.shape

    for _ in range(2):
        for y in range(1, H-1):
            for x in range(1, W-1):

                neighbors = [
                    h[y+1,x], h[y-1,x],
                    h[y,x+1], h[y,x-1]
                ]

                avg = sum(neighbors)/4

                if abs(h[y,x] - avg) > max_diff:
                    h[y,x] = avg + np.sign(h[y,x]-avg)*max_diff

    return normalize(h)

def remove_small_water(height_map, sea_level, min_size=200):

    water = height_map < sea_level
    visited = np.zeros_like(water, dtype=bool)

    h, w = water.shape

    for y in range(h):
        for x in range(w):

            if not water[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            region = []

            while stack:
                cy, cx = stack.pop()

                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue

                if visited[cy, cx] or not water[cy, cx]:
                    continue

                visited[cy, cx] = True
                region.append((cy, cx))

                stack.extend([
                    (cy+1, cx),
                    (cy-1, cx),
                    (cy, cx+1),
                    (cy, cx-1),
                ])

            if len(region) < min_size:
                for ry, rx in region:
                    height_map[ry, rx] = sea_level + 0.01

    return height_map

@dataclass
class ProceduralMapConfig:
    width: int = 256
    height: int = 256

    sea_level: float = 0.42
    beach_width: int = 3

    grass_level: float = 0.52
    rock_level: float = 0.68
    snow_level: float = 0.82

    continent_strength: float = 0.65
    mountain_strength: float = 0.35

    texture_variation: int = 10


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())

    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - arr_min) / (arr_max - arr_min)


def blur_simple(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    result = arr.astype(np.float32).copy()

    for _ in range(passes):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
        result = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    return result


def fbm_noise(
    seed: int,
    width: int,
    height: int,
    octaves: list[tuple[int, float]],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float32)

    for blur_passes, weight in octaves:
        base = rng.random((height, width), dtype=np.float32)
        layer = blur_simple(base, passes=blur_passes)
        result += weight * layer

    return normalize(result)


def ridged_noise(base_noise: np.ndarray) -> np.ndarray:
    ridged = 1.0 - np.abs(2.0 * base_noise - 1.0)
    ridged = ridged ** 1.5
    return normalize(ridged)


def radial_continent_mask(width: int, height: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt(xx**2 + yy**2)
    mask = 1.0 - np.clip(dist, 0.0, 1.0)
    mask = mask ** 0.9

    return normalize(mask)


def generate_height_seed_map(seed: int, config: ProceduralMapConfig) -> np.ndarray:
    width = config.width
    height = config.height

    continent_noise = fbm_noise(
        seed=seed,
        width=width,
        height=height,
        octaves=[
            (30, 0.55),
            (18, 0.30),
            (10, 0.15),
        ],
    )

    coast_noise = fbm_noise(
        seed=seed + 888,
        width=width,
        height=height,
        octaves=[
            (12, 0.6),
            (5, 0.4),
        ],
    )

    continent_mask = radial_continent_mask(width, height)
    continents = normalize(
        config.continent_strength * continent_mask
        + (1.0 - config.continent_strength) * continent_noise
    )

    continents = continents + 0.08 * (coast_noise - 0.5)
    continents = normalize(continents)

    mountain_base = fbm_noise(
        seed=seed + 101,
        width=width,
        height=height,
        octaves=[
            (12, 0.45),
            (6, 0.35),
            (3, 0.20),
        ],
    )
    mountains = ridged_noise(mountain_base)
    mountains = blur_simple(mountains, passes=2)

    # Local detail
    detail = fbm_noise(
        seed=seed + 202,
        width=width,
        height=height,
        octaves=[
            (6, 0.50),
            (3, 0.30),
            (1, 0.20),
        ],
    )

    mountain_mask = fbm_noise(
        seed=seed + 777,
        width=width,
        height=height,
        octaves=[
            (20, 0.7),
            (10, 0.3),
        ],
    )

    mountain_mask = mountain_mask ** 2.5

    height_seed = (
            0.78 * continents
            + 0.18 * mountains * mountain_mask
            + 0.04 * detail
    )

    def erosion_smooth(height: np.ndarray, iterations: int = 3) -> np.ndarray:
        h = height.copy()

        for _ in range(iterations):
            blurred = blur_simple(h, passes=1)
            h = 0.6 * h + 0.4 * blurred

        return normalize(h)

    height_seed = normalize(height_seed)
    height_seed = height_seed ** 1.15
    height_seed = erosion_smooth(height_seed, iterations=3)
    height_seed = remove_small_water(height_seed, config.sea_level)
    height_seed = slope_limit(height_seed)
    return height_seed


def estimate_beach_mask(
    water_mask: np.ndarray,
    max_distance: int = 3,
) -> np.ndarray:
    beach = np.zeros_like(water_mask, dtype=bool)

    padded_water = np.pad(water_mask, ((max_distance, max_distance), (max_distance, max_distance)), mode="edge")

    h, w = water_mask.shape
    for y in range(h):
        for x in range(w):
            if water_mask[y, x]:
                continue

            y0 = y
            x0 = x
            patch = padded_water[y0:y0 + 2 * max_distance + 1, x0:x0 + 2 * max_distance + 1]
            if np.any(patch):
                beach[y, x] = True

    return beach


def generate_moisture_map(seed: int, config: ProceduralMapConfig) -> np.ndarray:
    return fbm_noise(
        seed=seed + 303,
        width=config.width,
        height=config.height,
        octaves=[
            (20, 0.50),
            (9, 0.30),
            (4, 0.20),
        ],
    )


def generate_segmentation_map(
    height_seed_map: np.ndarray,
    moisture_map: np.ndarray,
    config: ProceduralMapConfig,
) -> np.ndarray:
    seg = np.full(height_seed_map.shape, 2, dtype=np.uint8)

    water_mask = height_seed_map < config.sea_level
    seg[water_mask] = 0

    beach_mask = estimate_beach_mask(
        water_mask=water_mask,
        max_distance=config.beach_width,
    )
    seg[beach_mask & (~water_mask)] = 1

    rock_mask = (height_seed_map >= config.rock_level)
    seg[rock_mask] = 3

    snow_mask = height_seed_map >= config.snow_level
    seg[snow_mask] = 4

    dry_rock_mask = (
        (height_seed_map >= config.grass_level)
        & (height_seed_map < config.rock_level)
        & (moisture_map < 0.28)
        & (~beach_mask)
        & (~water_mask)
    )
    seg[dry_rock_mask] = 3

    return seg


def generate_texture_map(
    seed: int,
    segmentation_map: np.ndarray,
    height_seed_map: np.ndarray,
    config: ProceduralMapConfig,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000)

    h, w = segmentation_map.shape
    texture = np.zeros((h, w, 3), dtype=np.uint8)

    color_map = {
        0: np.array([50, 105, 160], dtype=np.int32),   # water
        1: np.array([210, 194, 132], dtype=np.int32),  # sand
        2: np.array([110, 160, 90], dtype=np.int32),   # grass
        3: np.array([128, 128, 128], dtype=np.int32),  # rock
        4: np.array([235, 235, 235], dtype=np.int32),  # snow
    }

    variation = rng.integers(
        -config.texture_variation,
        config.texture_variation + 1,
        size=(h, w, 3),
        dtype=np.int32,
    )

    # Height-based shading
    depth = (config.sea_level - height_seed_map)
    depth = np.clip(depth, 0, 1)

    shade = (height_seed_map * 30.0 - depth * 60.0).astype(np.int32)
    shade = np.repeat(shade[:, :, None], 3, axis=2)

    for class_id, base_color in color_map.items():
        mask = segmentation_map == class_id
        class_pixels = base_color + variation[mask] + shade[mask]
        class_pixels = np.clip(class_pixels, 0, 255)
        texture[mask] = class_pixels.astype(np.uint8)

    return texture


def save_segmentation_png(segmentation_map: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    value_map = {
        0: 0,
        1: 64,
        2: 128,
        3: 192,
        4: 255,
    }

    seg_vis = np.zeros_like(segmentation_map, dtype=np.uint8)

    for class_id, value in value_map.items():
        seg_vis[segmentation_map == class_id] = value

    Image.fromarray(seg_vis, mode="L").save(path)


def save_texture_png(texture_map: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture_map, mode="RGB").save(path)


def generate_maps(
    seed: int,
    config: ProceduralMapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height_seed_map = generate_height_seed_map(seed, config)
    moisture_map = generate_moisture_map(seed, config)
    segmentation_map = generate_segmentation_map(
        height_seed_map=height_seed_map,
        moisture_map=moisture_map,
        config=config,
    )
    texture_map = generate_texture_map(
        seed=seed,
        segmentation_map=segmentation_map,
        height_seed_map=height_seed_map,
        config=config,
    )

    return height_seed_map, segmentation_map, texture_map