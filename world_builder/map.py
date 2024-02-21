from enum import StrEnum
from itertools import product
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterator, Optional
from pytmx import TiledMap, TiledTileset

from gstk.creation.graph_registry import Message
from gstk.graph.graph import Node
import world_builder
from world_builder.graph_registry import MapRect, MapRootData, WorldBuilderNodeType
from world_builder.map_data_interface import get_gid_tile_properties


MAP_METADATA_DIRNAME: str = "map_metadata"
MAP_METADATA_TEMPLATES_DIRNAME: str = "map_metadata_templates"
MAP_FILENAME: str = "map.tmx"


class MapHierarchy(object):
    def __init__(self, map_root_data: MapRootData):
        self._map_root_data: MapRootData = map_root_data
        # Check that draw_diameter is a valid power of the width and height.
        # This is a complexity hedge and keeps the map traversal algorithm
        # and context selection rooted in fairly simple algebra.
        if not math.log(self._map_root_data.width, self._map_root_data.draw_diameter).is_integer() or \
                not math.log(self._map_root_data.height, self._map_root_data.draw_diameter).is_integer():
            raise ValueError(f"Invalid map.")
    @property
    def draw_diameter(self) -> int:
        return self._map_root_data.draw_diameter

    @property
    def global_row_count(self) -> int:
        return self._map_root_data.height

    @property
    def global_column_count(self) -> int:
        return self._map_root_data.width

    def get_tree_height(self) -> int:
        return math.ceil(math.log(max(self.global_column_count, self.global_row_count), self.draw_diameter))

    def get_shape_at_depth(self, depth: int) -> tuple[float, float]:
        """
        Return the matrix shape at a given depth in the map.
        """
        # Calculate the maximum number of levels based on the larger dimension
        max_dimension = max(self.global_column_count, self.global_row_count)
        distance_from_leaf_level: int = math.log(max_dimension, self.draw_diameter) - depth - 1
        # This check is bearing a lot of weight from the rest of the code.
        if distance_from_leaf_level < 0:
            raise ValueError(f"Invalid depth for map: {depth}")
        cur_width = self.global_column_count / (self.draw_diameter ** distance_from_leaf_level)
        cur_height = self.global_row_count / (self.draw_diameter ** distance_from_leaf_level)
        return cur_height / self.draw_diameter, cur_width / self.draw_diameter

    def get_rect_level(self, map_rect: MapRect) -> int:
        return int(math.log(max(self.global_column_count, self.global_row_count), self.draw_diameter)) - math.log(max(map_rect.width, map_rect.height), self.draw_diameter)

    def get_rect_level_coordinates(self, map_rect: MapRect):
        depth: int = self.get_rect_level(map_rect)
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth)
        return map_rect.y / leaf_count[0], map_rect.x / leaf_count[1]

    def get_leaf_count_per_tile(self, depth: int) -> tuple[int, int]:
        row_count, column_count = self.get_shape_at_depth(depth)
        return self.global_row_count / row_count, self.global_column_count / column_count

    def list_rects(self, depth: int) -> Iterator[MapRect]:
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth)
        for row, column in self.list_level_coordinates(depth):
            yield MapRect(y=row*leaf_count[0], x=column*leaf_count[1], height=leaf_count[0], width=leaf_count[1])

    def list_level_coordinates(self, depth: int) -> Iterator[tuple[int, int]]:
        row_count, column_count = self.get_shape_at_depth(depth)
        if not row_count.is_integer() or not column_count.is_integer():
            raise ValueError(f"Invalid map.")
        return product(range(int(row_count)), range(int(column_count)))

    def list_child_rects(self, map_rect: MapRect) -> Iterator[MapRect]:
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(self.get_rect_level(map_rect) + 1)
        if map_rect.width % leaf_count[0] or map_rect.height % leaf_count[1]:
            raise ValueError(f"MapRect {map_rect} is not a multiple of the leaf count per tile.")
        for y in range(int(map_rect.width / leaf_count[0])):
            for x in range(int(map_rect.height / leaf_count[1])):
                yield MapRect(x=map_rect.x + x*leaf_count[1], y=map_rect.y + y*leaf_count[0], width=leaf_count[1], height=leaf_count[0])

    def get_parent_rect(self, map_rect: MapRect) -> MapRect:
        """
        Parent rect will be at a multiple offset of the leaf count per tile
        of the next level up.
        """
        depth: int = self.get_rect_level(map_rect)
        if depth == 0:
            raise ValueError("MapRect is already at the top level.")
        parent_level_leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth - 1)
        return MapRect(x=map_rect.x - (map_rect.x % parent_level_leaf_count[1]), y=map_rect.y - (map_rect.y % parent_level_leaf_count[0]), width=parent_level_leaf_count[1], height=parent_level_leaf_count[0])



class MapNodeManager:
    def __init__(self, map_root_node: Node, map_hierarchy: MapHierarchy):
        self._map_root_node: MapRoot = map_root_node
        self._map_hierarchy: MapHierarchy = map_hierarchy

    def list_nodes_for_processing(self, skip_non_empty: bool = True, parent_node: Optional[Node] = None, commit: bool = False) -> Iterator[Node]:
        pass


class MapRoot:
    """
    Can add asset from template or it can be added manually.


    Args:
        CreationGroup (_type_): _description_
    """
    def __init__(self, storage_node: Node, resource_location: Path):
        if storage_node.node_type != WorldBuilderNodeType.MAP_ROOT:
            raise ValueError(f"Node {storage_node.id} is not a MapRoot.")
        self._storage_node: Node = storage_node
        self._resource_location: Path = resource_location

    @property
    def data(self) -> MapRootData:
        return self._storage_node.data

    def node_manager(self) -> MapNodeManager:
        return MapNodeManager(self._storage_node, MapHierarchy(self.data))

    def generate_map_from_prompt(self, prompt: str, update_existing: bool = False):
        # Currently not supported:
        #    layers,
        #    draw_width draw_height not equaling width and height
        #    update_existing
        if self.data.layer_names is not None:
            raise ValueError("Layers are not supported yet.")
        if self.data.draw_width != self.data.width or self.data.draw_height != self.data.height:
            raise ValueError("Drawing width and height must equal width and height.")
        if update_existing:
            raise ValueError("Update existing is not supported yet.")

    # Methods below are related to Asset Management
    def get_image_buffer_from_tile_matrix(self, tile_matrix: list[list[int]]) -> Any:
        tiled_map: TiledMap = self.get_tiled_map()
        gid_tile_properties: dict[int, dict[str, Any]] = get_gid_tile_properties(tiled_map)
        tiled_map.get_tile_image_by_gid(3)

    def _path_to_map(self) -> Path:
        return self._resource_location / MAP_METADATA_DIRNAME / str(self._storage_node.id) / MAP_FILENAME

    def has_asset(self) -> bool:
        return self._path_to_map().exists()

    def get_tiled_map(self) -> TiledMap:
        if not self.has_asset():
            raise ValueError("Map does not exist.")
        return TiledMap(filename=str(self._path_to_map()))

    def list_asset_map_templates(self) -> Iterator[str]:
        print(Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME)
        for entry in os.listdir(Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME):
            if not entry.startswith(".") and not entry.startswith("_"):
                yield entry

    def add_asset_map_from_template(self, asset_name: str):
        if asset_name not in self.list_asset_templates():
            raise ValueError(f"Asset template {asset_name} does not exist.")
        asset_path: Path = Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME / asset_name
        shutil.copytree(asset_path, self._resource_location / MAP_METADATA_DIRNAME / str(self._storage_node.id))

