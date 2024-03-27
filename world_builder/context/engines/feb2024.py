import numpy as np
import os
from typing import Iterator, Optional

from gstk.graph.graph import Node
from gstk.models.chatgpt import Message, Role

from world_builder.context.base import ContextEngineBase
from world_builder.context.engine import register_context_engine
from world_builder.map import MapRoot, SparseMapTree, MapRect, WorldBuilderNodeType, get_cell_prompt
from world_builder.graph_registry import DescriptionMatrixData
from world_builder.map_data_interface import get_gid_description_context_string

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

def get_system_message(map_root: MapRoot, depth: int):
    if depth != map_root.tree.hierarchy.get_tree_height() - 1:
        system_message_base: str = f"""\
The map description is a description of the entire map. It is the first prompt in the recursive chain of prompts that describe the map.

You are assisting in the creation of an {map_root.data.width}x{map_root.data.height} tile map.  To better utilize your internal semantic vector space, rather than being asked to create the {map_root.data.width}x{map_root.data.height} tile map in one go, you will create a {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrix that divides the entire map into 9 cells. Each cell is to contain a prompt describing its corresponding projected area on the map, with a linguistic complexity appropriate to the map description. From there, you will create a {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrix representing the areas below each of the 9 initial cells, and so on, until you have created the {map_root.data.width}x{map_root.data.height} tile map.  Note that, accordingly, each level of the map contains nine times more total cells than the level above it, and each cell corresponds to nine times less area of the tile map than the cells in the level above it.

To assist in your creation of the map via these matrices, you are provided the following context: the overall map description, each recursive parent prompt above the matrix being created, and the position of each recursive parent prompt in the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} matrix in which it was created. For example, a map root description of a circle of trees spanning the entire map might have in y,x position {map_root.data.draw_diameter // 2 },0 of its initial prompt matrix a directive for creating the left middle of the circle of trees, and in position 6,6 of its initial prompt matrix a directive for creating the bottom right of the circle of trees, and so on. The recursive parent prompts are provided to help you maintain consistency with the map description as you create the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrices.

The following illustrates the structure of the parent context provided to you and highlights the recursive nature of the prompts and their increasing specificity with respect to tile map area:

Map description: A description of the entire map.
"""
    else:
        system_message_base: str = f"""\
The map description is a description of the entire map. It is the first prompt in the recursive chain of prompts that describe the map.

You are assisting in the creation of an {map_root.data.width}x{map_root.data.height} tile GID map.  To better utilize your internal semantic vector space, rather than being asked to create the {map_root.data.width}x{map_root.data.height} tile map in one go, you will create a {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrix that divides the entire map into 9 cells. Each cell is to contain a prompt describing its corresponding projected area on the map, with a linguistic complexity appropriate to the map description. From there, you will create a {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrix representing the areas below each of the 9 initial cells, and so on, until you have created the {map_root.data.width}x{map_root.data.height} tile map. The non-leaf level values you create are 3x3 prompt string matrices, and leaf level values are {map_root.data.draw_diameter}x{map_root.data.draw_diameter} integer tile map GID values.  Note that each level of the map contains nine times more total cells than the level above it, and each cell corresponds to nine times less area of the tile map than the cells in the level above it.

To assist in your creation of the tile GID map via these matrices, you are provided tile map context and for which each GID in the tile GID matrix output is contained in the tile map context.

Further you provided the following key context: the overall map description, each recursive parent prompt above the matrix being created, and the position of each recursive parent prompt in the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} matrix in which it was created. For example, a map root description of a circle of trees spanning the entire map might have in position 1,1 of its initial prompt matrix a directive for creating the left middle of the circle of trees (approximating one eighth the arc of the circle), and in position 3,3 of its initial prompt matrix a directive for creating the bottom right of the circle of trees, and so on. The recursive parent prompts are provided to help you maintain consistency with the map description as you create the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} prompt matrices.

The following illustrates the structure of the parent context provided to you and highlights the recursive nature of the prompts and their increasing specificity with respect to tile map area:

Map description: A description of the entire map.
"""

#     print(f"""You are creating a 2D tilemap for a 2D game represented as integers in a CSV.
# Create a 3x3 matrix in which each cell is one of the gid integers below, and that adheres to description.
#
        


    depth_descriptions: list[str] = []
    if depth > 0:
        for i in range(0, depth):
            if i == 0:
                depth_descriptions.append(f"First level prompt describes subset area one ninth the size of the entire map, at the provided position in the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} inital matrix.")
            else:
                depth_descriptions.append(f"{LEVEL_LABEL_MAP[i]} level prompt describes subset area nine times smaller than {LEVEL_LABEL_MAP[i-1]} level prompt at the provided position in the {map_root.data.draw_diameter}x{map_root.data.draw_diameter} parent matrix.")
    print(depth_descriptions)
    return system_message_base + os.linesep.join(depth_descriptions) + os.linesep




def get_parent_data_context(map_root: MapRoot, map_rect: MapRect) -> Message:
    return Message(role=Role.SYSTEM, content=get_system_message(map_root, map_root.tree.hierarchy.get_rect_level(map_rect)))

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
        prompt_str += f"Create the {LEVEL_LABEL_MAP[level]} {map_root.data.draw_diameter}x{map_root.data.draw_diameter} matrix for the following map description: " + cell_prompt
    elif level < 4:
        if level == map_root.tree.hierarchy.get_tree_height() - 1:
            prompt_str += f"Create a leaf level {map_root.data.draw_diameter}x{map_root.data.draw_diameter} tile GID matrix for the following prompt: " + cell_prompt
        else:
            prompt_str += f"Create a {LEVEL_LABEL_MAP[level]} level {map_root.data.draw_diameter}x{map_root.data.draw_diameter} matrix for the following prompt: " + cell_prompt
    else:
        # This is an arbitrary limit, though even 10 level map is probably
        # prohibitively large.
        raise ValueError("Level too deep.")

    parent_context: list[str] = []
    if parent:
        print(f"PARENT NODE TYPE: {parent.node_type}")
        while parent.node_type != WorldBuilderNodeType.MAP_ROOT:
            parent_level = map_root.tree.hierarchy.get_rect_level(parent.data.map_rect)
            print(f"Parent level: {parent_level}")
            #assert parent_level != level and parent_level < level
            coords_in_parent = map_root.tree.hierarchy.get_coordinates_in_parent(node.data.map_rect)
            parent_context.insert(0, f"{LEVEL_LABEL_MAP[parent_level][0].upper() + LEVEL_LABEL_MAP[parent_level][1:]} level prompt [y,x coords in parent: {coords_in_parent[0], coords_in_parent[1]}]: {np.array(parent.data.tiles)[*coords_in_parent]}")
            node = parent
            parent = parent.parent
    assert parent is None or parent.node_type == WorldBuilderNodeType.MAP_ROOT
    if parent and parent.data.description:
        parent_context.insert(0, "Map description: " + parent.data.description)
    if level > 0:
        assert parent_context, "Parent context should not be empty for levels > 0"
        prompt_str += f"\n\nParent context:\n\n{os.linesep.join(parent_context)}"
    if level == map_root.tree.hierarchy.get_tree_height() - 1:
        prompt_str += f"\n\nTile map context:\n\n{get_gid_description_context_string(map_root.get_tiled_map())}"
    return prompt_str

def get_map_parent_chain(map_root: MapRoot, map_rect: MapRect) -> Iterator[MapRect]:
    if map_root.tree.hierarchy.get_rect_level(map_rect) == 0:
        return
    current_map_rect: MapRect = map_root.tree.hierarchy.get_parent_rect(map_rect)
    while map_root.tree.hierarchy.get_rect_level(current_map_rect) >= 0:
        yield current_map_rect
        if map_root.tree.hierarchy.get_rect_level(current_map_rect) == 0:
            break
        current_map_rect = map_root.tree.hierarchy.get_parent_rect(current_map_rect)

@register_context_engine("feb2024")
class Feb2024ContextEngine(ContextEngineBase):
    def get_child_matrix_creation_context(self,
            map_rect: MapRect) -> Iterator[Message]:
        messages: list[Message] = []
        current_map_rect: MapRect = map_rect

        """
        parent_map_rects = reversed(list(get_map_parent_chain(map_root, map_rect)))
        for parent_map_rect in parent_map_rects:
            node: Node = map_root.tree.get_data_node(parent_map_rect)
            if node is None:
                continue
            messages.extend([
                get_parent_data_context(map_root, parent_map_rect),
                Message(
                    role=Role.USER,
                    content=get_cell_prompt_with_context(map_root, parent_map_rect)
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=f"tiles:\n{node.data.tiles}"
                )
            ])
        """

        #messages.append(
        #    Message(
        #        role=Role.USER,
        #        content=get_cell_prompt_with_context(map_root, map_rect)
        #    )
        #)

        messages.extend([
            get_parent_data_context(self.map_root, map_rect),
            Message(
                role=Role.USER,
                content=get_cell_prompt_with_context(self.map_root, map_rect)
            )
        ])
        return messages

    def can_create_child_matrix(self, map_rect: MapRect) -> bool:
        return True

