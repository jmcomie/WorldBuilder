

from enum import Enum, StrEnum
from typing import Iterator, Optional, Type

from gstk.graph.graph import Node
from gstk.models.chatgpt import Message, Role
import numpy as np

from world_builder.context.base import ContextEngineBase
from world_builder.context.engine import InvalidCellIdentifier, register_context_engine
from world_builder.graph_registry import MapRect
from world_builder.map import MapRoot


#    print(f"""You are creating a 2D tilemap for a 2D game represented as integers in a CSV.
# Create a ten by ten CSV file's contents in which each cell is one of the
# gid integers below, and that adheres to the following description: Create a map of the beach where the shore is aligned vertically near the center of the map and in which grass is on the left side of the map and in which water is on the right side of the map, with an appropriate gradation.
# GID Descriptions:

class CardinalityIndex(Enum):
    NORTH = ("north", (0, -1))
    NORTHEAST = ("northeast", (1, -1))
    EAST = ("east", (1, 0))
    SOUTHEAST = ("southeast", (1, 1))
    SOUTH = ("south", (0, 1))
    SOUTHWEST = ("southwest", (-1, 1))
    WEST = ("west", (-1, 0))
    NORTHWEST = ("northwest", (-1, -1))

def get_edge_cardinality(map_root: MapRoot, translated_x_y_coordinates) -> Optional[CardinalityIndex]:
    if translated_x_y_coordinates[0] < 0 and translated_x_y_coordinates[1] < 0:
        return CardinalityIndex.NORTHWEST
    elif translated_x_y_coordinates[0] == map_root.data.width and translated_x_y_coordinates[1] < 0:
        return CardinalityIndex.NORTHEAST
    elif translated_x_y_coordinates[0] < 0 and translated_x_y_coordinates[1] == map_root.data.height:
        return CardinalityIndex.SOUTHWEST
    elif translated_x_y_coordinates[0] == map_root.data.width and translated_x_y_coordinates[1] == map_root.data.height:
        return CardinalityIndex.SOUTHEAST
    elif translated_x_y_coordinates[0] < 0:
        return CardinalityIndex.WEST
    elif translated_x_y_coordinates[0] == map_root.data.width:
        return CardinalityIndex.EAST
    elif translated_x_y_coordinates[1] < 0:
        return CardinalityIndex.NORTH
    elif translated_x_y_coordinates[1] == map_root.data.height:
        return CardinalityIndex.SOUTH
    return None

def get_user_message_neighbor_context_str(map_rect: MapRect, map_root: MapRoot) -> str:
    base_str: str = ""
    rect_matrix: np.array[MapRect] = map_root.tree.hierarchy.get_rect_matrix(map_root.tree.hierarchy.get_rect_level(map_rect))
    coordinates_list = list(zip(*np.where(rect_matrix == map_rect)))
    assert len(coordinates_list) == 1
    x_y = np.array((coordinates_list[0][1], coordinates_list[0][0]))

    for cardinality_index in CardinalityIndex:
        cardinality: str = cardinality_index.value[0]
        x_y_translation: np.array[int] = np.array(cardinality_index.value[1])
        translated_x_y_coordinates: np.array[int] = x_y + x_y_translation
        edge_cardinality: Optional[CardinalityIndex] = get_edge_cardinality(map_root, translated_x_y_coordinates)
        if edge_cardinality is not None:
            base_str += f"The direct {cardinality} is off the {edge_cardinality.value[0]}ern edge of the map.\n"
            continue
        neighbor_map_rect: MapRect = rect_matrix[translated_x_y_coordinates[1], translated_x_y_coordinates[0]]
        neighbor_node: Optional[Node] = map_root.tree.get_data_node(neighbor_map_rect)
        if neighbor_node is None:
            # No data in this direction.
            base_str += f"There is not yet data to the {cardinality}.\n"
            continue
        base_str += f"The {cardinality} neighbor tiles are:\n{neighbor_node.data.tiles}\n"

    return base_str

def get_system_message(map_rect: MapRect, map_root: MapRoot) -> Message:
    content: str = f"""\
    The coordinates given are x,y.
    The map origin is at the top left corner and the x-axis increases to the right and the y-axis increases downward.
    Try to fit the produced data into the context of the map.
    """
    return Message(
        role=Role.SYSTEM,
        content=content
    )

def get_user_message(map_rect: MapRect, map_root: MapRoot) -> Message:
    system_message_base: str = f"""\
You are filling in a {map_root.data.draw_diameter} by {map_root.data.draw_diameter} subset of a {map_root.data.width} by {map_root.data.height} (width, height) 2D tilemap for a 2D game represented as an integer matrix.
Create a {map_root.data.draw_diameter} by {map_root.data.draw_diameter} int rect/matrix in which each cell is one of the gid integers below, that comports with the neighbor matrix context where available, and that implements this subset of the map appropriate to this description of the entire map: {map_root.data.description}

YOU ARE DRAWING AT X,Y COORDINATES {map_rect.x}, {map_rect.y}. The top left of the {map_root.data.draw_diameter} by {map_root.data.draw_diameter} rect you are drawing is at position {map_rect.x}, {map_rect.y} in the {map_root.data.width} by {map_root.data.height} map.

Neighbor context:
{get_user_message_neighbor_context_str(map_rect, map_root)}

GID Descriptions:

gid: 1 description: light green grass
gid: 2 description: light green grass accented with leaves arranged from lower left to upper right
gid: 3 description: light green grass accented with leaves arranged from upper left to lower right
gid: 4 description: green grass
gid: 5 description: green grass accented with leaves arranged from lower left to upper right
gid: 6 description: green grass accented with leaves arranged from upper left to lower right
gid: 7 description: sand
gid: 8 description: ankle deep water
gid: 9 description: knee deep water
gid: 10 description: shoulder deep water
gid: 11 description: water too deep to stand in

"""
    return Message(
        role=Role.USER,
        content=system_message_base
    )


@register_context_engine("mar2024")
class March2024ContextEngine(ContextEngineBase):
    def __init__(self, map_root: MapRoot):
        super().__init__(map_root)

    def get_child_matrix_creation_context(self,
            map_rect: MapRect) -> Iterator[Message]:
        if not self.can_create_child_matrix(map_rect):
            raise InvalidCellIdentifier(f"Cannot create child matrix for {map_rect}.")
        messages: list[Message] = []

        messages.extend([
            get_system_message(map_rect, self.map_root),
            get_user_message(map_rect, self.map_root)
        ])
        return messages

    def can_create_child_matrix(self, map_rect: MapRect) -> bool:
        return self.map_root.tree.hierarchy.get_rect_level(map_rect) == \
                self.map_root.tree.hierarchy.get_tree_height() -1

