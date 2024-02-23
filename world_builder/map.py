from enum import Enum, StrEnum
from itertools import product
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterator, Optional, Union
from gstk.graph.registry import GraphRegistry
from pydantic import BaseModel
from pytmx import TiledMap, TiledTileset

import numpy as np
from gstk.creation.graph_registry import Message
from gstk.graph.graph import Node
import world_builder
from world_builder.graph_registry import MapRect, MapRootData, WorldBuilderNodeType, MapRectMetadata, MapMatrixData, DescriptionMatrixData
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
        """
        Leaf depth is tree height minus 1.  Like index vs length.
        """
        return math.ceil(math.log(max(self.global_column_count, self.global_row_count), self.draw_diameter))

    def get_shape_at_depth(self, depth: int) -> tuple[float, float]:
        """
        Return the matrix shape at a given depth in the map.
        """
        # Calculate the maximum number of levels based on the larger dimension
        max_dimension = max(self.global_column_count, self.global_row_count)
        distance_from_leaf_level: int = math.log(max_dimension, self.draw_diameter) - depth - 1
        # This check is relied on by several methods.
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

    def check_rect(self, map_rect: MapRect):
        row_count, column_count = self.get_shape_at_depth(self.get_rect_level(map_rect))
        if not row_count.is_integer() or not column_count.is_integer():
            raise ValueError(f"Invalid rect {map_rect} at depth {self.get_rect_level(map_rect)}.")

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

    def walk_rects(self, map_rect: MapRect) -> Iterator[MapRect]:
        yield map_rect
        if self.get_tree_height() - 1 > self.get_rect_level(map_rect):
            for child_rect in self.list_child_rects(map_rect):
                yield from self.walk_rects(child_rect)


class SparseMapNode:
    def __init__(self, map_rect: MapRect, node: Optional[Node] = None):
        self._map_rect: MapRect = map_rect
        self._node: Optional[Node] = node

    @property
    def node(self) -> Optional[Node]:
        return self._node

    @node.setter
    def node(self, node: Node):
        self._node = node

    @property
    def rect(self) -> MapRect:
        return self._map_rect

    @property
    def data(self):
        if not self.has_data():
            raise ValueError("Node has no data.")
        return self._node.data 

    def has_data(self) -> bool:
        return self._node is not None


class SparseMapTree:
    def __init__(self, map_root_node: Node):
        if map_root_node.node_type != WorldBuilderNodeType.MAP_ROOT:
            raise ValueError("Root node is not a MapRoot.")
        self._map_root_node: Node = map_root_node
        self._map_hierarchy: MapHierarchy = MapHierarchy(self._map_root_node.data)
        assert isinstance(self._map_root_node.data, MapRootData)
        self._rect_data_node_dict: Optional[dict[tuple, Node]] = None

    def list_nodes_for_processing(self,
                                  skip_non_empty: bool = False,
                                  parent_node: Optional[Node] = None,
                                  commit_changes: bool = False) -> Iterator[Node]:
        for node in self.walk_tree(root_node=parent_node):
            if node.has_data() and skip_non_empty:
                continue
            if not node.has_data():
                node.node = self.get_or_create_data_node(self._get_empty_model_instance(node.rect))
            assert isinstance(node.node.data, (MapMatrixData, DescriptionMatrixData))
            data_before: dict = node.node.data.model_dump()
            yield node.node
            if commit_changes and data_before != node.node.data.model_dump():
                self.check_data(node.node.data)
                node.save()
                node._node.session.commit()

    @property
    def root_node(self) -> Node:
        return self._map_root_node

    def get_data_node(self, map_rect: MapRect) -> Optional[Node]:
        self.ensure_rect_node_dict()
        return self._rect_data_node_dict.get(map_rect.to_tuple())

    def check_data(self, data: Union[MapMatrixData, DescriptionMatrixData]):
        self._map_hierarchy.check_rect(data.map_rect)
        if not isinstance(data, (MapMatrixData, DescriptionMatrixData)):
            raise ValueError(f"Data {data} is not a valid type.")
        if isinstance(data, MapMatrixData) and self._map_hierarchy.get_rect_level(data.map_rect) != self._map_hierarchy.get_tree_height() - 1:
            raise ValueError(f"MapMatrixData {data} is not at the leaf level.")
        elif isinstance(data, DescriptionMatrixData) and self._map_hierarchy.get_rect_level(data.map_rect) == self._map_hierarchy.get_tree_height() - 1:
            raise ValueError(f"DescriptionMatrixData {data} is at the leaf level.")

    def _get_empty_model_instance(self, map_rect: MapRect) -> Union[MapMatrixData, DescriptionMatrixData]:
        level: int = self._map_hierarchy.get_rect_level(map_rect)
        if level == 0:
            raise ValueError("Cannot create an empty model instance for the root node.")
        elif self._map_hierarchy.get_tree_height() -1 == level:
            return MapMatrixData(map_rect=map_rect, tiles=np.zeros((map_rect.height, map_rect.width), dtype=np.int32).tolist())
        else:
            return DescriptionMatrixData(map_rect=map_rect, tiles=np.empty((map_rect.height, map_rect.width), dtype=str).tolist())

    def _ensure_parents_exist(self, map_rect: MapRect) -> Node:
        parent_chain: list[MapRect] = []
        current_rect: MapRect = map_rect

        if self._map_hierarchy.get_rect_level(map_rect) > 0:
            current_rect = self._map_hierarchy.get_parent_rect(map_rect)
            # Ascend the tree until we find a node that has a data node.
            while not self.has_data_node(current_rect) and self._map_hierarchy.get_rect_level(current_rect) >= 0:
                parent_chain.append(current_rect)
                current_rect = self._map_hierarchy.get_parent_rect(current_rect)

        parent_rect_node: Node = self.get_data_node(current_rect) if self._map_hierarchy.get_rect_level(current_rect) != 0 else self._map_root_node
        # Create in descending order.
        for rect in reversed(parent_chain):
            new_node: Node = parent_rect_node.create_child(self._get_empty_model_instance(rect))
            self._rect_data_node_dict[rect.to_tuple()] = new_node
            parent_rect_node = new_node
        return parent_rect_node

    def get_or_create_data_node(self, data: Union[MapMatrixData, DescriptionMatrixData]) -> Node:
        self.ensure_rect_node_dict()
        if data.map_rect.to_tuple() in self._rect_data_node_dict:
            return self._rect_data_node_dict[data.map_rect.to_tuple()]

        self.check_data(data)
        parent_node: Node = self._ensure_parents_exist(data.map_rect)
        node: Node = parent_node.create_child(data)
        parent_node.session.commit()
        self._rect_data_node_dict[data.map_rect.to_tuple()] = node
        return node

    def update_data_node(self, data: Union[MapMatrixData, DescriptionMatrixData]):
        self.ensure_rect_node_dict()
        if data.map_rect.to_tuple() not in self._rect_data_node_dict:
            raise ValueError(f"Node {data.map_rect} does not exist.")
        self.check_data(data)
        node: Node = self._rect_data_node_dict[data.map_rect.to_tuple()]
        node.data = data
        node.session.commit()

    def has_data_node(self, map_rect: MapRect) -> bool:
        return map_rect.to_tuple() in self._rect_data_node_dict

    def list_children(self, parent_node: Node) -> Iterator[Node]:
        assert isinstance(self._map_root_node.data, MapRectMetadata)
        for rect in self._map_hierarchy.list_child_rects(parent_node.data.map_rect):
            yield SparseMapNode(rect, self.get_data_node(rect))

    def walk_tree(self, root_node: Optional[Node] = None) -> Iterator[SparseMapNode]:
        if root_node is None or root_node.id == self._map_root_node.id:
            root_node_rect = list(self._map_hierarchy.list_rects(0))[0]
        else:
            assert isinstance(root_node.data, MapRectMetadata)
            root_node_rect = root_node.data.map_rect
        for rect in self._map_hierarchy.walk_rects(root_node_rect):
            yield SparseMapNode(rect, self.get_data_node(rect))
 
    def ensure_rect_node_dict(self):
        if self._rect_data_node_dict is not None:
            return
        self._rect_data_node_dict = {}
        for node in self._map_root_node.walk_tree():
            if isinstance(node.data, MapRootData):
                continue
            assert isinstance(node.data, MapRectMetadata)
            self._rect_data_node_dict[node.data.map_rect.to_tuple()] = node


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
        self._tree: SparseMapTree = SparseMapTree(storage_node)

    @property
    def data(self) -> MapRootData:
        return self._storage_node.data

    @property
    def tree(self) -> SparseMapTree:
        return self._tree

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
        if asset_name not in self.list_asset_map_templates():
            raise ValueError(f"Asset template {asset_name} does not exist.")
        asset_path: Path = Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME / asset_name
        shutil.copytree(asset_path, self._resource_location / MAP_METADATA_DIRNAME / str(self._storage_node.id))

