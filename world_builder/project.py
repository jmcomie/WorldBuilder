from enum import StrEnum
from itertools import product
from lib2to3.pytree import BasePattern
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Optional, Type
from gstk.graph.project_locator import ProjectLocator

from pydantic import BaseModel
import shutil
from pytmx import TiledMap
import world_builder
from gstk.graph.graph import Node, SystemEdgeType
from gstk.creation.group import new_group, GroupProperties, CreationGroup

from world_builder.graph_registry import MapRect, MapRootData, WorldBuilderNodeType
from world_builder.map_data_interface import get_gid_tile_properties


EXPECTED_CONTENTS: list[str] = [
    "map_metadata",
    "graph.sqlite"
]

MAP_METADATA_DIRNAME: str = "map_metadata"
MAP_METADATA_TEMPLATES_DIRNAME: str = "map_metadata_templates"
MAP_FILENAME: str = "map.tmx"


class WorldBuilderProjectDirectory(ProjectLocator):
    base_directory = Path("world_builder_data")
    projects_directory_name = Path("projects")
    assets_directory = Path("assets_for_world_builder")
    base_directory_dot_filename = Path(".world_builder_root")

    def __init__(self, base_path: Optional[Path] = None):
        self._base_path: Path = base_path if base_path is not None else self._ensure_base_path()
        print(f"base_path: {self._base_path}")

    def get_ancestor_path_with_dot_file(self) -> Optional[Path]:
        current_path = Path.cwd()
        # Evaluate all ascending paths except the root.
        while current_path != current_path.parent:
            if (current_path / self.base_directory_dot_filename).exists():
                return current_path
            current_path = current_path.parent
        return None

    def _ensure_base_path(self) -> BasePattern:
        base_path: Optional[Path] = self.get_ancestor_path_with_dot_file()
        if base_path is None:
            base_path = self.base_directory
            if not base_path.exists():
                base_path.mkdir(parents=True)
            (base_path / self.base_directory_dot_filename).touch()
            (base_path / self.assets_directory).mkdir(exist_ok=True)
            (base_path / self.projects_directory_name).mkdir(exist_ok=True)
        return base_path

    def list_project_ids(self) -> list[str]:
        # Iterate all directories in cwd and check for the expected contents.
        return [d for d in os.listdir(Path(self.base_path) / self.projects_directory_name) if all([os.path.exists(
                os.path.join(self.base_path, self.projects_directory_name, d, f)) for f in EXPECTED_CONTENTS])]

    def project_id_exists(self, project_id: str) -> bool:
        """
        Return boolean indicating whether the project exists. Prints warnings to stderr
        if the project directory is empty or contains unexpected files.
        """
        project_path: Path = self.base_path / self.projects_directory_name / project_id
        if not project_path.is_dir():
            return False
        if not all([os.path.exists(os.path.join(project_path, f)) for f in EXPECTED_CONTENTS]):
            print(f"Warning: Project {project_id} does not contain all expected files.", file=sys.stderr)
        exist_mask: list[bool] = [os.path.exists(os.path.join(project_path, f)) for f in EXPECTED_CONTENTS]
        if not all(exist_mask):
            print(f"Warning: Project {project_id} does not contain all expected files.", file=sys.stderr)
        if set(os.listdir(project_path)) - set(EXPECTED_CONTENTS):
            print(f"Warning: Project {project_id} contains unexpected files.", file=sys.stderr)
        return all(exist_mask)

    def get_project_resource_location(self, project_id: str) -> Path:
        print(f"returning {self.base_path / self.projects_directory_name / project_id}")
        return self.base_path / self.projects_directory_name / project_id


import math
import pprint
from typing import Iterator

# Given values


class MapHierarchy(object):
    def __init__(self, map_root_data: MapRootData):
        self._map_root_data: MapRootData = map_root_data

    @property
    def draw_diameter(self) -> int:
        return self._map_root_data.draw_diameter

    @property
    def height(self) -> int:
        return self._map_root_data.height

    @property
    def width(self) -> int:
        return self._map_root_data.width

    def get_shape_at_depth(self, depth: int) -> tuple[float, float]:
        """
        Return the matrix shape at a given depth in the map.
        """
        # Calculate the maximum number of levels based on the larger dimension
        max_dimension = max(self.width, self.height)
        distance_from_leaf_level: int = math.log(max_dimension, self.draw_diameter) - depth - 1
        # This check is bearing a lot of weight from the rest of the code.
        if distance_from_leaf_level < 0:
            raise ValueError(f"Invalid depth for map: {depth}")
        cur_width = self.width / (self.draw_diameter ** distance_from_leaf_level)
        cur_height = self.height / (self.draw_diameter ** distance_from_leaf_level)
        return cur_height / self.draw_diameter, cur_width / self.draw_diameter

    def get_rect_level(self, map_rect: MapRect) -> int:
        return int(math.log(max(self.width, self.height), self.draw_diameter)) - math.log(max(map_rect.width, map_rect.height), self.draw_diameter)

    def get_rect_level_coordinates(self, map_rect: MapRect):
        depth: int = self.get_rect_level(map_rect)
        leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth)
        return map_rect.y / leaf_count[0], map_rect.x / leaf_count[1]

    def get_leaf_count_per_tile(self, depth: int) -> tuple[int, int]:
        row_count, column_count = self.get_shape_at_depth(depth)
        return self.height / row_count, self.width / column_count

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

    def get_parent_rect(self, map_rect: MapRect):
        """
        Parent rect will be at a multiple offset of the leaf count per tile
        of the next level up.
        """
        depth: int = self.get_rect_level(map_rect)
        if depth == 0:
            raise ValueError("MapRect is already at the top level.")
        parent_level_leaf_count: tuple[int, int] = self.get_leaf_count_per_tile(depth - 1)
        return MapRect(x=map_rect.x - (map_rect.x % parent_level_leaf_count[1]), y=map_rect.y - (map_rect.y % parent_level_leaf_count[0]), width=parent_level_leaf_count[1], height=parent_level_leaf_count[0])


class MapController:
    def get_context_for_cell(cell_node: Node):
        pass

    def get_child_type_for_cell(cell_node: Node):
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


class WorldBuilderProject:
    def __init__(self, node: Node, resource_location: Path):
        self._storage_node: Node = node
        self._resource_location: Path = resource_location

    def list_map_roots(self) -> Iterator[MapRoot]:
        for _edge, node in self.root_group.node.get_out_nodes(
            edge_type_filter=[SystemEdgeType.contains],
            node_type_filter=[WorldBuilderNodeType.MAP_ROOT]
        ):
            yield MapRoot(self._resource_location, node)

    def get_map_root_dict(self) -> dict[str, MapRoot]:
        return {map_root.data.name: map_root for map_root in self.list_map_roots()}

    def new_map(self, map_root_data: MapRootData) -> MapRoot:
        map_root: MapRoot = MapRoot(self._storage_node.create_child(map_root_data), self._resource_location)
        self._storage_node.session.commit()
        return map_root