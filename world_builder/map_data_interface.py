from enum import StrEnum
from functools import cache
import io
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pytmx import TiledMap, TiledTileset

from PIL import Image
from gstk.creation.graph_registry import Message


class GIDData(BaseModel):
    description: Optional[str] = Field(default=None, description="A description of the tile.")
    image_data: Optional[bytes] = Field(default=None, description="The image data for the tile.")


def get_gid_tile_properties(tiled_map: TiledMap):
    gid_tile_properties: dict[int, dict[str, Any]] = {}
    for tileset in tiled_map.tilesets:
        assert isinstance(tileset, TiledTileset)
        for gid in range(tileset.firstgid, tileset.firstgid+tileset.tilecount):
            gid_tile_properties[gid] = tiled_map.get_tile_properties_by_gid(gid)
    return gid_tile_properties


def get_gid_data(tiled_map: TiledMap) -> dict[int, GIDData]:
    gid_data: dict[int, dict[str, GIDData]] = {}
    for tileset in tiled_map.tilesets:
        assert isinstance(tileset, TiledTileset)
        path_to_asset_file: Path = Path(tiled_map.filename).parent / tileset.source
        for gid in range(tileset.firstgid, tileset.firstgid+tileset.tilecount):
            gid_data[gid] = GIDData(**{
                "description": tiled_map.get_tile_properties_by_gid(gid).get("description", None),
                "image_data": get_tileset_tile_image(path_to_asset_file, *(tiled_map.get_tile_image_by_gid(gid)[1]))
            })
    return gid_data


def get_tile_matrix_from_csv(csv_string: str|Path) -> list[list[int]]:
    df: pd.DataFrame = pd.read_csv(csv_string, header=None)
    df = df.astype(int)
    return df.values.tolist()


@cache
def get_image_from_filepath(image_filepath: Path) -> Image:
    return Image.open(image_filepath)


@cache
def get_tileset_tile_image(image_filepath: Path, x: int, y: int, width: int, height: int) -> bytes:
    image = get_image_from_filepath(image_filepath)
    crop_box = (x, y, x+width, y+height)
    cropped_image = image.crop(crop_box)
    buffer = io.BytesIO()
    cropped_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


def get_image_from_tile_matrix(tile_matrix: list[list[int]], tiled_map: TiledMap) -> Image:
    gid_data: dict[int, GIDData] = get_gid_data(tiled_map)
    height, width = np.array(tile_matrix).shape
    tile_width = tiled_map.tilewidth
    tile_height = tiled_map.tileheight
    buffer_width_px = width * tile_width
    buffer_height_px = height * tile_height
    new_image_buffer: Image = Image.new('RGB', (buffer_width_px, buffer_height_px), 'white')

    for cell_y, row in enumerate(tile_matrix):
        for cell_x, gid in enumerate(row):
            if gid not in gid_data:
                print(f"gid: {gid} not found in gid_data.")
                continue
            if not gid_data[gid].image_data:
                print(f"gid: {gid} has no image data.")
                continue
            cell_image = Image.open(io.BytesIO(gid_data[gid].image_data))
            px_x = cell_x * tile_width
            px_y = cell_y * tile_height
            new_image_buffer.paste(cell_image, (px_x, px_y))

    return new_image_buffer


def image_from_csv_tile_matrix(csv_string: str|Path, tiled_map: TiledMap) -> Image:
    tile_matrix: list[list[int]] = get_tile_matrix_from_csv(csv_string)
    return get_image_from_tile_matrix(tile_matrix, tiled_map)