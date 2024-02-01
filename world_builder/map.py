from enum import StrEnum
from pathlib import Path
from pytmx import TiledMap, TiledTileset

from gstk.creation.graph_registry import Message
import world_builder.map_metadata


MAPS_DIRECTORY: str = "maps"
MAP_FILENAME: str = "default.tmx"


class LayerNames(StrEnum):
    STRUCTURES = "Structures"
    TERRAIN = "Terrain"
    ALL_TILES = "AllTiles"


def set_layer_data(layer_name, data: list[list[int]]):
    pass

def get_tiled_map_by_asset_name(asset_name: str = "MiniWorldSprites") -> TiledMap:
    return TiledMap(filename=Path(world_builder.map_metadata.__file__).parent / asset_name / MAPS_DIRECTORY / MAP_FILENAME)

def generate_map_chat_context(map: TiledMap):
    for tileset in map.tilesets:
        assert isinstance(tileset, TiledTileset)
        for gid in range(tileset.firstgid, tileset.firstgid+tileset.tilecount):
            print(map.get_tile_properties_by_gid(gid))

# Write file.

def main():
    map: TiledMap = get_tiled_map_by_asset_name("MiniWorldSprites")
    generate_map_chat_context(map)

if __name__ == "__main__":
    main()