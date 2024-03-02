from functools import cache
from typing import Iterator
from gstk.graph.registry import ProjectProperties

# Import Message
from gstk.models.chatgpt import Message, Role
import numpy as np
from world_builder.project import WorldBuilderProject
from world_builder.context import get_description_matrix_context_messages
from world_builder.map import MapRoot, SparseMapTree, SparseMapNode
from world_builder.graph_registry import DescriptionMatrixData, MapMatrixData, MapRect, MapRootData, WorldBuilderNodeType
from gstk.graph.graph import Node, get_project, new_project
from gstk.graph.registry import GraphRegistry
from world_builder.project import WorldBuilderProjectDirectory
from gstk.llmlib.object_generation import get_chat_completion_object_response
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


def get_map_matrix_context_messages(tree: SparseMapTree, data: MapMatrixData) -> Iterator[Message]:
    pass

async def process_tree(map_root: MapRoot, root_node: Node):
    for data in map_root.tree.list_data_for_processing(commit_changes=True):
        print("MAP RECT")
        print(data.map_rect)
        if isinstance(data, MapMatrixData):
            data.tiles[0][0] = data.map_rect.x
        elif isinstance(data, DescriptionMatrixData):
            messages: list[Message] = get_description_matrix_context_messages(map_root, data)
            preamble: str = ""# "Draw a three by three matrix with a sentence or two description of each area that adheres to this directive and the system directives: "
            messages.append(Message(role=Role.USER, content=f"Draw an island that has some beach, covers about 80% of the map, is round but not a perfect circle, and has a small village in the center among trees, with beaches on the edges."))
            #messages.append(Message(role=Role.USER, content=r"Draw a three by three matrix with a sentence or two description of each area that adheres to this directive and the system directive: Draw an island where the island is circular and extends out to 80% of the map. Note that the upper left element in the matrix corresponds to the upper left part of the map and so on for the rest. If the area described by the sentence is partial or subdivided, provide that information in the description."))
            res: list = await get_chat_completion_object_response(list(GraphRegistry.get_node_types(data.__class__))[0], messages)
            print(res)
            data.tiles = res.tiles
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")


#project: WorldBuilderProject = ensure_project()
#map_root: MapRoot = ensure_map()
#tree: SparseMapTree = map_root.tree

#node = tree.get_data_node(MapRect(x=0, y=63, width=9, height=9, layer=None))

