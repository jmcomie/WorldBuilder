from enum import StrEnum
from itertools import product
from lib2to3.pytree import BasePattern
import math
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
from world_builder.map import MapRoot
from world_builder.map_data_interface import get_gid_tile_properties


EXPECTED_CONTENTS: list[str] = [
    "graph.sqlite",
]


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



class WorldBuilderProject:
    def __init__(self, node: Node, resource_location: Path):
        self._storage_node: Node = node
        self._resource_location: Path = resource_location

    @property
    def resource_location(self) -> Path:
        return self._resource_location

    def list_map_roots(self) -> Iterator[MapRoot]:
        for node in self._storage_node.list_children():
            if isinstance(node.data, MapRootData):
                yield MapRoot(node, self._resource_location)

    def get_map_root_dict(self) -> dict[str, MapRoot]:
        return {map_root.data.name: map_root for map_root in self.list_map_roots()}

    def get_map_root(self, name: str) -> MapRoot:
        return self.get_map_root_dict().get(name)

    def new_map_root(self, map_root_data: MapRootData) -> MapRoot:
        print(f"type: {type(map_root_data)}")
        if map_root_data.name in self.get_map_root_dict():
            raise Exception(f"Map root with name {map_root_data.name} already exists.")
        map_root: MapRoot = MapRoot(self._storage_node.create_child(map_root_data), self._resource_location)
        self._storage_node.session.commit()
        return map_root
