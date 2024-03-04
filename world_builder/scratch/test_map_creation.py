import code
import readline


# START Monkey patch.
import collections
import collections.abc

from gstk.models.chatgpt import Message, Role
collections.Mapping = collections.abc.Mapping
# END Monkey patch.

from gstk.graph.graph import Node

#from gstk.creation.graph_registry import Message, Role

#from gstk.creation.group import get_chat_completion_object_response

from gstk.llmlib.object_generation import get_chat_completion_object_response
from gstk.graph.registry import GraphRegistry, NodeTypeData
from world_builder.graph_registry import WorldBuilderNodeType


from world_builder.map import MapRoot




prompt_str: str = """You are creating a 2D tilemap for a 2D game represented as integers in a CSV.
Create an eight by eight int matrix in which each cell is one of the gid integers below, and that adheres to the following description: Create a map of the beach on the eastern side of some geography where the shore is aligned vertically near the center of the map and in which grass is on the left side of the map and in which water is on the right side of the map, with an appropriate gradation.
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


async def xxx_map_creation_test(map_root: MapRoot):
    node_type_data: NodeTypeData = GraphRegistry.get_node_type_data(WorldBuilderNodeType.MAP_MATRIX)
    messages: list[Message] = []
    if node_type_data.system_directive:
        messages.append(Message(role=Role.system, content=node_type_data.system_directive))
    # Assuming prompt_str is defined somewhere above this snippet
    messages.append(Message(role=Role.user, content=prompt_str))
    # Ensure get_chat_completion_object_response is an async function and awaited
    response, vector = await get_chat_completion_object_response(WorldBuilderNodeType.MAP_MATRIX, messages)
    node: Node = map_root.add_new_node(WorldBuilderNodeType.MAP_MATRIX, response)
    node_updated: Node = await map_root.update(node, "Check the created output for any issues and retry, ensuring the best adherence to the original prompt.")
    node_updated_twice: Node = await map_root.update(node_updated, "Check the created output for any issues and retry, ensuring the best adherence to the original prompt.")
    print(f"Node updated twice: {node_updated_twice}")
    print(node_updated_twice.data)
    # readline.parse_and_bind and code.interact are blocking calls and should be outside of the async function or handled differently if you need to use them asynchronously.
    readline.parse_and_bind("tab: complete")
    code.interact(local=locals())

