from enum import StrEnum
from lib2to3.pytree import BasePattern
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Optional, Type

from pydantic import BaseModel
import shutil
from pytmx import TiledMap
import world_builder
from gstk.graph.interface.resource_locator.local_file import LocalFileLocator
from gstk.graph.registry_context_manager import graph_registries
from gstk.creation.api import CreationProject, get_creation_project
from gstk.graph.registry import EdgeRegistry, NodeRegistry
from gstk.creation.group import new_group, GroupProperties, CreationGroup
from gstk.graph.interface.graph.graph import Node
from gstk.graph.system_graph_registry import SystemEdgeType

from world_builder.graph_registry import MapRoot, WorldBuilderNodeRegistry, WorldBuilderEdgeRegistry, WorldBuilderNodeType
from world_builder.map_data_interface import get_gid_tile_properties

def world_builder_registry(fn):
    """
    Decorator to ensure that the WorldBuilderNodeRegistry and WorldBuilderEdgeRegistry
    are used when calling the function.
    """
    def wrapper(*args, **kwargs):
        with graph_registries(WorldBuilderNodeRegistry, WorldBuilderEdgeRegistry):
            return fn(*args, **kwargs)
    return wrapper


EXPECTED_CONTENTS: list[str] = [
    "map_metadata",
    "graph.sqlite"
]

MAP_METADATA_DIRNAME: str = "map_metadata"
MAP_METADATA_TEMPLATES_DIRNAME: str = "map_metadata_templates"
MAP_FILENAME: str = "map.tmx"

# Supplemental data.



# We should have a single map matrix data object.


class WorldBuilderProjectDirectory(LocalFileLocator):
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
        # print a warning to standard error if the project directory contains unexpected files or no files.
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


class MapTileGroup(CreationGroup):
    def __init__(self,  _resource_location: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resource_location: Path = _resource_location

    def iterate_children(self):
        pass

    @property
    @world_builder_registry
    def data(self) -> MapRoot:
        return self.node.data

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

    def get_image_buffer_from_tile_matrix(self, tile_matrix: list[list[int]]) -> Any:
        tiled_map: TiledMap = self.get_tiled_map()
        gid_tile_properties: dict[int, dict[str, Any]] = get_gid_tile_properties(tiled_map)
        tiled_map.get_tile_image_by_gid(3)

    def _path_to_map(self) -> Path:
        return self._resource_location / MAP_METADATA_DIRNAME / str(self.node.id) / MAP_FILENAME

    def has_asset(self) -> bool:
        return self._path_to_map().exists()

    def get_tiled_map(self) -> TiledMap:
        if not self.has_asset():
            raise ValueError("Map does not exist.")
        return TiledMap(filename=str(self._path_to_map()))

    def list_asset_templates(self) -> list[str]:
        print(Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME)
        for entry in os.listdir(Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME):
            if not entry.startswith(".") and not entry.startswith("_"):
                yield entry

    def add_asset_from_template(self, asset_name: str):
        if asset_name not in self.list_asset_templates():
            raise ValueError(f"Asset template {asset_name} does not exist.")
        asset_path: Path = Path(world_builder.__file__).parent / MAP_METADATA_TEMPLATES_DIRNAME / asset_name
        shutil.copytree(asset_path, self._resource_location / MAP_METADATA_DIRNAME / str(self.node.id))


class WorldBuilderProject(CreationProject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @world_builder_registry
    def list_map_roots(self) -> Iterator[MapTileGroup]:
        for _edge, node in self.root_group.node.get_out_nodes(
            edge_type_filter=[SystemEdgeType.contains],
            node_type_filter=[WorldBuilderNodeType.MAP_ROOT]
        ):
            yield MapTileGroup(self.resource_location, node)

    @world_builder_registry
    def get_map_root_dict(self) -> dict[str, MapTileGroup]:
        return {map_root.data.name: map_root for map_root in self.list_map_roots()}

    @world_builder_registry
    def new_map(self, map_root: MapRoot) -> MapTileGroup:
        node: Node = self.root_group.add_new_node(WorldBuilderNodeType.MAP_ROOT, map_root)
        # create asset group and an entry for every asset type
        # how do we list assets
        return MapTileGroup(self.resource_location, node)


def get_project(project_id: str) -> MapTileGroup:
    project_locator: WorldBuilderProjectDirectory = WorldBuilderProjectDirectory()
    project: WorldBuilderProject = get_creation_project(project_id, resource_locator=project_locator, project_class=WorldBuilderProject)
    return project

