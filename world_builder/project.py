from enum import StrEnum
from lib2to3.pytree import BasePattern
import os
from pathlib import Path
import sys
from typing import Iterator, Optional

from pydantic import BaseModel

from gstk.graph.interface.resource_locator.local_file import LocalFileLocator
from gstk.graph.registry_context_manager import graph_registries
from gstk.creation.api import CreationProject
from gstk.graph.registry import EdgeRegistry, NodeRegistry
from world_builder.graph_registry import MapRoot, WorldBuilderNodeRegistry, WorldBuilderEdgeRegistry, WorldBuilderNodeType
from gstk.creation.group import new_group
from gstk.creation.group import GroupProperties, CreationGroup
from gstk.graph.interface.graph.graph import Node
from gstk.graph.system_graph_registry import SystemEdgeType


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

# Supplemental data.


class MapTileGroup(CreationGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.node = node

    def iterate_children(self):
        pass

    @property
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


        # We should have a single map matrix data object.



class WorldBuilderProject(CreationProject):

    def __init__(self, creation_project: CreationProject):
        self._creation_project = creation_project

    def list_map_roots(self) -> Iterator[MapTileGroup]:
        for _edge, node in self._creation_project.root_group.node.get_out_nodes(
            edge_type_filter=[SystemEdgeType.contains],
            node_type_filter=[WorldBuilderNodeType.MAP_ROOT]
        ):
            yield MapTileGroup(node)

    @world_builder_registry
    def new_map(self, map_root: MapRoot) -> MapTileGroup:
        if (map_root.asset_name):
            pass
        node: Node = self._creation_project.root_group.add_new_node(WorldBuilderNodeType.MAP_ROOT, map_root)
        # create asset group and an entry for every asset type
        # how do we list assets
        return MapTileGroup(node)


class WorldBuilderProjectDirectory(LocalFileLocator):
    base_directory = Path("world_builder_data")
    projects_directory_name = Path("projects")
    assets_directory = Path("assets_for_world_builder")
    base_directory_dot_filename = Path(".world_builder_root")

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

    def __init__(self, base_path: Optional[Path] = None):
        self._base_path: Path = base_path if base_path is not None else self._ensure_base_path()

    def list_project_ids(self) -> list[str]:
        # Iterate all directories in cwd and check for the expected contents.
        return [d for d in os.listdir(Path(self.base_path) / self.projects_directory_name) if all([os.path.exists(
                os.path.join(self.base_path, d, f)) for f in EXPECTED_CONTENTS])]

    def project_id_exists(self, project_id: str) -> bool:
        # print a warning to standard error if the project directory contains unexpected files or no files.
        project_path: Path = self.base_path / self.projects_directory_name / project_id
        if not project_path.is_dir():
            return False
        if not all([os.path.exists(os.path.join(project_path, f)) for f in EXPECTED_CONTENTS]):
            print(f"Warning: Project {project_id} does not contain all expected files.", file=sys.stderr)
        exist_mask: list[bool] = [os.path.exists(os.path.join(self.base_path, project_id, f)) for f in EXPECTED_CONTENTS]
        if not all(exist_mask):
            print(f"Warning: Project {project_id} does not contain all expected files.", file=sys.stderr)
        if set(os.listdir(os.path.join(self.base_path, project_id))) - set(EXPECTED_CONTENTS):
            print(f"Warning: Project {project_id} contains unexpected files.", file=sys.stderr)
        return all(exist_mask)

    def get_project_resource_location(self, project_id: str) -> Path:
        return self.base_path / self.projects_directory_name / project_id
