import numpy as np
import os
from typing import Iterator, Optional

from gstk.graph.graph import Node
from gstk.llmlib.object_generation import get_chat_completion_object_response
from gstk.models.chatgpt import Message, Role

from world_builder.map import MapRoot, SparseMapTree, MapRect, WorldBuilderNodeType, get_cell_prompt
from world_builder.graph_registry import DescriptionMatrixData

# Implement view context chain.
# Implement generate child matrix.


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
    def get_neighbor_description(_map_rect: Optional[MapRect]):
        if _map_rect is None:
            return
        parent_rect: MapRect = tree._map_hierarchy.get_parent_rect(_map_rect)
        coords_in_parent: tuple[int, int] = tree._map_hierarchy.get_coordinates_in_parent(_map_rect)
        node: Node = tree.get_data_node(parent_rect)
        if node:
            return np.array(node.data.tiles)[*coords_in_parent]

    map_rect_neighbors = tree._map_hierarchy.get_rect_neighbors(map_rect, diameter)
    print(map_rect_neighbors)
    return np.vectorize(get_neighbor_description)(map_rect_neighbors)


def get_description_matrix_context_messages(map_root: MapRoot, map_rect: MapRect) -> Iterator[Message]:
    neighbor_data = get_neighbor_data_context(map_root.tree, map_rect)
    parent_data: list[str] = get_parent_data_context(map_root.tree, map_root.tree.get_data_node(map_rect))
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
The area you are segmenting is a {map_rect.width}x{map_rect.height} area at ({map_rect.x}, {map_rect.y}).
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
                role=Role.SYSTEM,
                content=f"""\
The recursive prompts are listed below, from the root to the immediate parent of the created area. The root prompt
represents a description of the entire map.

The root prompt:
{parent_data[0]}


{("Successive child prompts, up to the immediate parent: " + os.linesep.join(parent_data[1:])) if len(parent_data) > 1 else ""}

""")
        )

    messages.append(
        Message(
            role=Role.USER,
            content=get_cell_prompt(map_root, map_root.tree._map_hierarchy.get_map_rect_cell_identifier(map_rect))
        )
    )
    return messages
