from enum import StrEnum
from pathlib import Path
from pytmx import TiledMap, TiledTileset

from gstk.creation.graph_registry import Message
import world_builder.map_metadata
from world_builder.project import WorldBuilderProjectLocator


class MapDataLoader():
    def __init__(self, resource_locator: WorldBuilderProjectLocator, project_id: str):
        self.resource_locator = resource_locator

def generate_map_chat_context(map: TiledMap):
    for tileset in map.tilesets:
        assert isinstance(tileset, TiledTileset)
        for gid in range(tileset.firstgid, tileset.firstgid+tileset.tilecount):
            print(map.get_tile_properties_by_gid(gid))


