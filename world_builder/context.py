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

LEVEL_LABEL_MAP: dict[int, str] = {
    0: "first",
    1: "second",
    2: "third",
    3: "fourth",
    4: "fifth",
    5: "sixth",
    6: "seventh",
    7: "eighth",
    8: "ninth",
    9: "tenth"
}

def get_system_message(depth: 0):
    system_message_base: str = """\
The root prompt is a description of the entire map. It is the first prompt in the recursive chain of prompts that describe the map.

You are assisting in the creation of an 81x81 tile map.  To better utilize your internal semantic vector space, rather than being asked to create the 81x81 tile map in one go, you will create a 3x3 prompt matrix that divides the entire map into 9 cells. Each cell is to contain a prompt describing its corresponding projected area on the map, with a linguistic complexity appropriate to the map description. From there, you will create a 3x3 prompt matrix representing the areas below each of the 9 initial cells, and so on, until you have created the 81x81 tile map.  Note that, accordingly, each level of the map contains nine times more total cells than the level above it, and each cell corresponds to nine times less area of the tile map than the cells in the level above it.

To assist in your creation of the map via these matrices, you are provided the following context: the overall map description, each recursive parent prompt above the matrix being created, and the position of each recursive parent prompt in the 3x3 matrix in which it was created. For example, a map root description of a circle of trees spanning the entire map might have in position 1,1 of its initial prompt matrix a directive for creating the left middle of the circle of trees (approximating one eighth the arc of the circle), and in position 3,3 of its initial prompt matrix a directive for creating the bottom right of the circle of trees, and so on. The recursive parent prompts are provided to help you maintain consistency with the map description as you create the 3x3 prompt matrices.

The following illustrates the structure of the parent context provided to you and highlights the recursive nature of the prompts and their increasing specificity with respect to tile map area:

Map description: A description of the entire map.
"""

    depth_descriptions: list[str] = []
    if depth > 0:
        for i in range(0, depth):
            if i == 0:
                depth_descriptions.append("First level prompt describes subset area one ninth the size of the entire map, at the provided position in the 3x3 inital matrix.")
            else:
                depth_descriptions.append(f"{LEVEL_LABEL_MAP[i]} level prompt describes subset area nine times smaller than f{LEVEL_LABEL_MAP[i-1]} level prompt at the provided position in the 3x3 parent matrix.")
    ]
    return system_message_base + os.linesep.join(depth_descriptions[:depth]) + os.linesep



def get_parent_data_context(tree: SparseMapTree, node: Node) -> Message:
    return Message(role=Role.SYSTEM, content=get_system_message(tree.hierarchy.get_rect_level(node.data.map_rect)))

"""
def get_neighbor_data_context(tree: SparseMapTree, map_rect: MapRect, diameter: int = 3) -> Optional[np.ndarray[object]]:
    if tree.hierarchy.get_rect_level(map_rect) == 0:
        return None
    def get_neighbor_description(_map_rect: Optional[MapRect]):
        if _map_rect is None:
            return
        parent_rect: MapRect = tree.hierarchy.get_parent_rect(_map_rect)
        coords_in_parent: tuple[int, int] = tree.hierarchy.get_coordinates_in_parent(_map_rect)
        node: Node = tree.get_data_node(parent_rect)
        if node:
            return np.array(node.data.tiles)[*coords_in_parent]

    map_rect_neighbors = tree.hierarchy.get_rect_neighbors(map_rect, diameter)
    print(map_rect_neighbors)
    return np.vectorize(get_neighbor_description)(map_rect_neighbors)
"""


def get_cell_prompt_with_context(map_root: MapRoot, map_rect: MapRect) -> Message:
    cell_prompt: str = get_cell_prompt(map_root,
                                       map_root.tree.hierarchy.get_map_rect_cell_identifier(map_rect))

    print(f"Cell identifier: {map_root.tree.hierarchy.get_map_rect_cell_identifier(map_rect)}")
    print(f"Cell prompt: {cell_prompt}")
    node: Node = map_root.tree.ensure_data_node(map_rect)
    parent: Node = node.parent
    prompt_str: str = ""
    level: int = map_root.tree.hierarchy.get_rect_level(map_rect)
    if level == 0:
        prompt_str += f"Create the {LEVEL_LABEL_MAP[level]} 3x3 matrix for the following map description: " + cell_prompt
    elif level < 4:
        prompt_str += f"Create a {LEVEL_LABEL_MAP[level]} level 3x3 matrix for the following prompt: " + cell_prompt
    else:
        # This is an arbitrary limit, though even 10 level map is probably
        # prohibitively large.
        raise ValueError("Level too deep.")

    parent_context: list[str] = []
    if parent:
        while parent.node_type != WorldBuilderNodeType.MAP_ROOT:
            parent_level = map_root.tree.hierarchy.get_rect_level(parent.data.map_rect)
            assert parent_level != level and parent_level < level
            coords_in_parent = map_root.tree.hierarchy.get_coordinates_in_parent(node.data.map_rect)
            parent_context.insert(0, f"{LEVEL_LABEL_MAP[parent_level]} level prompt: {np.array(parent.data.tiles)[*coords_in_parent]}")
            node = parent
            parent = parent.parent
    assert parent is None or parent.node_type == WorldBuilderNodeType.MAP_ROOT
    if parent and parent.data.description:
        parent_context.insert(0, "Map description: " + parent.data.description)
    if level > 0:
        assert parent_context, "Parent context should not be empty for levels > 0"
        prompt_str += f"\n\nParent context:\n\n {os.linesep.join(parent_context)}"
    return prompt_str

def get_description_matrix_context_messages(map_root: MapRoot, map_rect: MapRect) -> Iterator[Message]:
    #neighbor_data = get_neighbor_data_context(map_root.tree, map_rect)

    system_message: Message = get_parent_data_context(
            map_root.tree, map_root.tree.ensure_data_node(map_rect))

    messages: list[Message] = [system_message]

    #messages.append(
    #    Message(
    #        role=Role.USER,
    #        content=get_cell_prompt_with_context(map_root, map_rect)
    #    )
    #)   

    messages.append(
        Message(
            role=Role.USER,
            content=get_cell_prompt_with_context(map_root, map_rect)
        )
    )
    return messages
