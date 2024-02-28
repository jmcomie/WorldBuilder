"""
This is beautiful code and it needs to be beautifully documented.
"""
from collections import deque
from enum import Enum, StrEnum
from itertools import product
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator, Optional, TypeVar, Union
from gstk.graph.registry import GraphRegistry
from pydantic import BaseModel
from pytmx import TiledMap, TiledTileset

import numpy as np
from gstk.creation.graph_registry import Message
from gstk.graph.graph import Node, breadth_first_traversal
import world_builder
from world_builder.graph_registry import MapRect, MapRootData, WorldBuilderNodeType, MapRectMetadata, MapMatrixData, DescriptionMatrixData
from world_builder.map_data_interface import get_gid_tile_properties


MAP_METADATA_DIRNAME: str = "map_metadata"
MAP_METADATA_TEMPLATES_DIRNAME: str = "map_metadata_templates"
MAP_FILENAME: str = "map.tmx"


def extract_2d_array_subset(arr: np.ndarray, x_y_coords: tuple[int, int], diameter: int, fill_value: Any = None) -> np.ndarray:
    if diameter % 2 == 0:
        raise ValueError(f"diameter must be odd number. got: {diameter}")
    if isinstance(arr, list):
        arr = np.array(arr)
    x, y = x_y_coords
    output = np.full((diameter, diameter), fill_value, dtype=object)

    # Calculate start and end indices in the original array
    radius = (diameter - 1) // 2
    start_x, end_x = (x - radius, x + radius + 1)
    start_y, end_y = (y - radius, y + radius + 1)

    overlap_start_x, overlap_start_y = (max(0, -start_x), max(0, -start_y))
    overlap_end_x, overlap_end_y = (diameter - max(0, end_x - arr.shape[0]), diameter - max(0, end_y - arr.shape[1]))

    output[overlap_start_x:overlap_end_x, overlap_start_y:overlap_end_y] = arr[max(start_x, 0):min(end_x, arr.shape[0]), max(start_y, 0): min(end_y, arr.shape[1])]
    return output

# have list rects produce a matrix

class MapHierarchy(object):
    def __init__(self, map_root_data: MapRootData):
        self._map_root_data: MapRootData = map_root_data
        # Check that draw_diameter is a valid power of the width and height.
        # This is a complexity hedge and keeps the map traversal algorithm
        # and context selection rooted in fairly simple algebra.
        if not math.log(self._map_root_data.width, self.draw_diameter).is_integer() or \
                not math.log(self._map_root_data.height, self.draw_diameter).is_integer():
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

    def get_rect_level_coordinates(self, map_rect: MapRect) -> tuple[int, int]:
        depth: int = self.get_rect_level(map_rect)
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth)
        return int(map_rect.y // leaf_count[0]), int(map_rect.x // leaf_count[1])

    def get_coordinates_in_parent(self, map_rect: MapRect) -> tuple[int, int]:
        depth: int = self.get_rect_level(map_rect)
        if depth == 0:
            raise ValueError("MapRect is already at the top level.")
        parent_rect: MapRect = self.get_parent_rect(map_rect)
        return (map_rect.y - parent_rect.y) // map_rect.height, (map_rect.x - parent_rect.x) // map_rect.width

    def get_leaf_count_per_tile(self, depth: int) -> tuple[int, int]:
        row_count, column_count = self.get_shape_at_depth(depth)
        return self.global_row_count / row_count, self.global_column_count / column_count

    def get_rect_matrix(self, depth: int) -> np.ndarray[MapRect]:
        level_shape = self.get_shape_at_depth(depth)
        arr: list[list[MapRect]] = np.ndarray((int(level_shape[0]), int(level_shape[1]),), dtype=object)
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth)
        for row, column in self.list_level_coordinates(depth):
            arr[row, column] = MapRect(y=row*leaf_count[0], x=column*leaf_count[1], height=leaf_count[0], width=leaf_count[1])
        return arr

    def check_rect(self, map_rect: MapRect):
        row_count, column_count = self.get_shape_at_depth(self.get_rect_level(map_rect))
        if not row_count.is_integer() or not column_count.is_integer():
            raise ValueError(f"Invalid rect {map_rect} at depth {self.get_rect_level(map_rect)}.")

    def list_level_coordinates(self, depth: int) -> Iterator[tuple[int, int]]:
        row_count, column_count = self.get_shape_at_depth(depth)
        if not row_count.is_integer() or not column_count.is_integer():
            raise ValueError(f"Invalid map.")
        return product(range(int(row_count)), range(int(column_count)))

    def get_rect_child_matrix(self, map_rect: MapRect) -> np.ndarray[MapRect]:
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(self.get_rect_level(map_rect) + 1)
        if map_rect.width % leaf_count[0] or map_rect.height % leaf_count[1]:
            raise ValueError(f"MapRect {map_rect} is not a multiple of the leaf count per tile.")
        arr: np.array = np.ndarray((self.draw_diameter, self.draw_diameter), dtype=object)
        for y in range(int(map_rect.width / leaf_count[0])):
            for x in range(int(map_rect.height / leaf_count[1])):
                arr[y, x] = MapRect(x=map_rect.x + x*leaf_count[1], y=map_rect.y + y*leaf_count[0], width=leaf_count[1], height=leaf_count[0])
        return arr

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

    def walk_rects(self, map_rect: Optional[MapRect] = None) -> Iterator[MapRect]:
        """
        Breadth first tree iteration.
        """
        if map_rect is None:
            root_level_rects: list[MapRect] = self.get_level_coordinates_rect((0, 0), 0)
            assert len(root_level_rects) == 1
            map_rect = root_level_rects[0]
        def get_child_rects(parent_rect: MapRect) -> Iterator[MapRect]:
            if self.get_tree_height() - 1 > self.get_rect_level(parent_rect):
                for rect in self.get_rect_child_matrix(parent_rect).flatten():
                    yield rect
        yield from breadth_first_traversal(map_rect, get_child_rects)

    def get_level_coordinates_rect(self, coords: tuple[int, int], level: int) -> MapRect:
        # XXX Verify this logic. (test that it's in the inverse of get_rect_level_coordinates)
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(level)
        return MapRect(x=coords[1]*leaf_count[1], y=coords[0]*leaf_count[0], width=leaf_count[1], height=leaf_count[0])

    def get_rect_neighbors(self, map_rect: MapRect, diameter: int) -> list[list[Optional[MapRect]]]:
        """
        Returns a 2d matrix of neighbors of shape (diameter, diameter). If a neighbor is off
        the map, it is None. The given rect is in the center of the returned matrix.
        """
        return extract_2d_array_subset(np.array(self.get_rect_matrix(self.get_rect_level(map_rect))),
                                       self.get_rect_level_coordinates(map_rect), diameter)

    def get_coord_neighbors(self, level: int, coordinates: tuple[int, int], diameter:int) -> np.ndarray[MapRect]:
        return extract_2d_array_subset(self.get_rect_matrix(level), coordinates, diameter)

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

    def list_data_for_processing(self,
                                  skip_non_empty: bool = False,
                                  parent_node: Optional[Node] = None,
                                  commit_changes: bool = False) -> Iterator[Union[MapMatrixData, DescriptionMatrixData]]:
        self.ensure_rect_node_dict()
        for sparse_node in self.walk_tree(root_node=parent_node):
            if sparse_node.has_data() and skip_non_empty:
                continue
            if not sparse_node.has_data():
                print(sparse_node.rect)
                sparse_node.node = self.get_or_create_data_node(self._get_empty_model_instance(sparse_node.rect))
            assert isinstance(sparse_node.node.data, (MapMatrixData, DescriptionMatrixData))
            data: Union[MapMatrixData, DescriptionMatrixData] = sparse_node.node.data
            yield data
            if commit_changes and data.model_dump() != sparse_node.node.data.model_dump():
                self.check_data(sparse_node.node.data)
                sparse_node.node.data = data
                self._rect_data_node_dict[sparse_node.node.data.map_rect.to_tuple()] = sparse_node.node
                sparse_node.node.save()
                sparse_node.node.session.commit()

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
        if self._map_hierarchy.get_tree_height() -1 == self._map_hierarchy.get_rect_level(map_rect):
            return MapMatrixData(map_rect=map_rect, tiles=np.zeros((self._map_root_node.data.draw_diameter, self._map_root_node.data.draw_diameter), dtype=np.int32).tolist())
        else:
            return DescriptionMatrixData(map_rect=map_rect, tiles=np.empty((self._map_root_node.data.draw_diameter, self._map_root_node.data.draw_diameter), dtype=str).tolist())

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
        print(data.map_rect.to_tuple())
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
        for rect in self._map_hierarchy.get_rect_child_matrix(parent_node.data.map_rect).flatten():
            yield SparseMapNode(rect, self.get_data_node(rect))

    def walk_tree(self, root_node: Optional[Node] = None) -> Iterator[SparseMapNode]:
        """
        Breadth first sparse iteration. Yields SparseMapNode for each entry in tree
        including root. If data node in tree exists for sparse node it is available, 
        """
        if root_node is None or root_node.id == self._map_root_node.id:
            root_node_rect = self._map_hierarchy.get_level_coordinates_rect((0, 0), 0)
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

    def get_node_neighbors(self, map_rect: MapRect, diameter: int) -> np.ndarray[Optional[Node]]:
        rect_neighbors = self._map_hierarchy.get_rect_neighbors(map_rect, diameter)
        def _get_node(rect: MapRect) -> Optional[Node]:
            if rect is None:
                return None
            return self.get_data_node(rect)
        return np.vectorize(_get_node)(rect_neighbors)

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

