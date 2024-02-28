import asyncio
from functools import cache
import os
from typing import Iterator, Optional
from gstk.graph.registry import ProjectProperties
import pprint
# Import Message
from gstk.models.chatgpt import Message, Role
import numpy as np
from world_builder.project import WorldBuilderProject
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
    if parent and parent.data.description:
        lst.insert(0, parent.data.description)
    return lst


def get_neighbor_data_context(tree: SparseMapTree, map_rect: MapRect, diameter: int = 3) -> Optional[np.ndarray[object]]:
    if tree._map_hierarchy.get_rect_level(map_rect) == 0:
        return None
    def get_neighbor_description(_map_rect: MapRect):
        parent_rect: MapRect = tree._map_hierarchy.get_parent_rect(_map_rect)
        coords_in_parent: tuple[int, int] = tree._map_hierarchy.get_coordinates_in_parent(_map_rect)
        node: Node = tree.get_data_node(parent_rect)
        if node:
            return node.data.tiles[coords_in_parent[1], coords_in_parent[0]]

    map_rect_neighbors = tree._map_hierarchy.get_rect_neighbors(map_rect, diameter)
    return np.vectorize(get_neighbor_description)(map_rect_neighbors)



def get_description_matrix_context_messages(map_root: MapRoot, data: DescriptionMatrixData) -> Iterator[Message]:
    neighbor_data = get_neighbor_data_context(tree, data.map_rect)
    parent_data: list[str] = get_parent_data_context(map_root.tree, map_root.tree.get_data_node(data.map_rect))
    map_root.data.height
    map_root.data.width
    map_root.data.draw_diameter
    messages = [
        Message(
            role=Role.SYSTEM,
            content=f"""\
You are creating a map as part of a recursive divide and conquer system in which you are tasked with
taking a description of an area on the map, and creating a tile matrix of shape {map_root.data.draw_diameter}, {map_root.data.draw_diameter}.

Create a {map_root.data.draw_diameter}x{map_root.data.draw_diameter} matrix with a sentence or two description of each area that adheres to the system directive. Note that the upper left of the matrix corresponds to the upper left part of the map and so on for the rest. If the area described by the sentence is partial or subdivided, provide that information in the description.
"""),
        Message(
            role=Role.SYSTEM,
            content=f"""\
The total width of the map is {map_root.data.width} and the total height of the map is {map_root.data.height}.
The area you are segmenting is a {data.map_rect.width}x{data.map_rect.height} area at ({data.map_rect.x}, {data.map_rect.y}).
You are describing the root layout.

"""),
    ]
    if neighbor_data is not None:
        messages.append(
            Message(
                role=Role.SYSTEM,
                content=f"""\
                    The prompt context is below, which includes the given node's prompt in the center and neighbor prompts around it.
                    {neighbor_data.tolist()}
                    """
            )
        )
    # TODO: fill this out.
    if parent_data:
        print(parent_data)
        messages.append(
            Message(
                role=Role.USER,
                content=f"""\
The recursive prompts are listed below, from the root to the immediate parent of the created area. The root prompt
represents a description of the entire map.

The root prompt:
parent_data[0]


{("Successive child prompts, up to the immediate parent: " + os.linesep.join(parent_data[1:])) if len(parent_data) > 1 else ""}

""")
        )
    return messages


def get_map_matrix_context_messages(tree: SparseMapTree, data: MapMatrixData) -> Iterator[Message]:
    pass

async def run_prompt_through_tree(prompt: str, map_root: MapRoot):
    for data in tree.list_data_for_processing(commit_changes=True):
        print("MAP RECT")
        print(data.map_rect)
        if isinstance(data, MapMatrixData):
            data.tiles[0][0] = data.map_rect.x
        elif isinstance(data, DescriptionMatrixData):
            messages: list[Message] = get_description_matrix_context_messages(map_root, data)
            preamble: str = ""# "Draw a three by three matrix with a sentence or two description of each area that adheres to this directive and the system directives: "
            messages.append(Message(role=Role.USER, content=f"{preamble} Draw an island that has some beach, covers about 80% of the map, is round but not a perfect circle, and has a small village in the center among trees, with beaches on the edges."))
            #messages.append(Message(role=Role.USER, content=r"Draw a three by three matrix with a sentence or two description of each area that adheres to this directive and the system directive: Draw an island where the island is circular and extends out to 80% of the map. Note that the upper left element in the matrix corresponds to the upper left part of the map and so on for the rest. If the area described by the sentence is partial or subdivided, provide that information in the description."))
            res: list = await get_chat_completion_object_response(list(GraphRegistry.get_node_types(data.__class__))[0], messages)
            print(res)
            data.tiles = res.tiles

        else:
            raise ValueError(f"Unexpected data type: {type(data)}")


project: WorldBuilderProject = ensure_project()
map_root: MapRoot = ensure_map()
tree: SparseMapTree = map_root.tree



node = tree.get_data_node(MapRect(x=0, y=63, width=9, height=9, layer=None))

