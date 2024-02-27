from functools import cache
from typing import Optional
from gstk.graph.registry import ProjectProperties
import numpy as np
from world_builder.project import WorldBuilderProject
from world_builder.map import MapRoot, SparseMapTree, SparseMapNode
from world_builder.graph_registry import DescriptionMatrixData, MapMatrixData, MapRect, MapRootData, WorldBuilderNodeType
from gstk.graph.graph import Node, get_project, new_project
from world_builder.project import WorldBuilderProjectDirectory

project_locator = WorldBuilderProjectDirectory()

PROJECT_ID: str = "testproject"
MAP_ROOT_ID: str = "testing"
#project: WorldBuilderProject = WorldBuilderProject(get_project(project_id, project_locator), project_locator.get_project_resource_location(project_id))

#map_root: MapRoot = 
#tree: SparseMapTree = map_root.tree


@cache
def ensure_project(project_id: str = PROJECT_ID) -> WorldBuilderProject:
    project: Node = get_project(project_id, project_locator)
    if project is None:
        project = new_project(ProjectProperties(id=project_id), project_locator)
    return WorldBuilderProject(project, project_locator.get_project_resource_location(project_id))

@cache
def ensure_map(map_root_id: str = MAP_ROOT_ID) -> MapRoot:
    project = ensure_project()
    map_root: MapRoot = project.get_map_root(map_root_id)
    if map_root is None:
        map_root = project.new_map_root(MapRootData(name=map_root_id, draw_diameter=3, width=81, height=81))
    return map_root


def get_parent_data_context(tree: SparseMapTree, node: Node):
    parent: Node = node.parent
    lst: list[str] = []
    if parent:
        while parent.node_type != WorldBuilderNodeType.MAP_ROOT:
            coords_in_parent = tree._map_hierarchy.get_coordinates_in_parent(node.data.map_rect)
            lst.insert(0, np.array(parent.data.tiles)[coords_in_parent[1], coords_in_parent[0]])
            node = parent
            parent = parent.parent
    assert parent is None or parent.node_type == WorldBuilderNodeType.MAP_ROOT
    if parent:
        lst.insert(0, parent.data.description)
    return lst


def get_neighbor_data_context(tree: SparseMapTree, node: Node, diameter: int = 3) -> np.ndarray[object]:
    neighbor_matrix: np.ndarray = tree.get_node_neighbors(node.data.map_rect, diameter)
    return np.vectorize(lambda n: n and n.data.tiles or None)(neighbor_matrix)


project: WorldBuilderProject = ensure_project()
map_root: MapRoot = ensure_map()
tree: SparseMapTree = map_root.tree


for data in tree.list_data_for_processing(commit_changes=True):
    if isinstance(data, MapMatrixData):
        data.tiles[0][0] = data.map_rect.x
    elif isinstance(data, DescriptionMatrixData):
        data.tiles[0][0] = str(data.map_rect.to_tuple())

node = tree.get_data_node(MapRect(x=0, y=63, width=9, height=9, layer=None))

