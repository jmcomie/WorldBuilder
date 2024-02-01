from enum import StrEnum
from typing import Literal, get_args
import gstk.creation.api
import gstk.shim
from pydantic import BaseModel, Field, field_validator

from gstk.user_registries.story.graph_registry import StoryNodeRegistry, StoryEdgeRegistry
from gstk.graph.registry import EdgeRegistry, NodeRegistry
from gstk.creation.graph_registry import CreationNode
from gstk.graph.system_graph_registry import SystemEdgeType
from gstk.creation.graph_registry import Message, Role
from gstk.graph.registry_context_manager import default_registries


WorldBuilderNodeRegistry: NodeRegistry = StoryNodeRegistry.clone()
WorldBuilderEdgeRegistry: EdgeRegistry = StoryEdgeRegistry.clone()
default_registry = gstk.creation.api.default_registry

class WorldBuilderNodeType(StrEnum):
    MAP_ROOT = "world_builder.map"
    MAP_MATRIX = "world_builder.map_matrix"
    DESCRIPTION_MATRIX = "world_builder.description_matrix"
    ALL = "world_builder.*"

#system_message: Message = Message(
#    role=Role.system,
#    content="""
#You are tasked with creating a 16x16 map for use in a videogame by representing
#one or more of the nodes for the videogame ground layer.  The ground layer contains:
#trees, grass, roads, foilage, water of various, sand.  The map should adhere to the description and you will
#get updates on making it more arty.
#""")


class MapMatrixData(BaseModel):
    """Data describing the name and tiles of a map in a videogame."""
    width: int = Field(default=None, description="The width of the map.")
    height: int = Field(default=None, description="The height of the map.")
    tiles: list[list[int]] = Field(default=None, description="The tiles on the map.")


class DescriptionMatrixData(BaseModel):
    """Data describing a description of a map in a videogame."""
    width: int = Field(default=None, description="The width of the map.")
    height: int = Field(default=None, description="The height of the map.")
    tiles: list[list[str]] = Field(default=None, description="The sub-map cell description.")


# Supplemental data.
class MapRoot(BaseModel):
    name: str
    draw_width: int
    draw_height: int
    width: int
    height: int


WorldBuilderNodeRegistry.register_node(
    WorldBuilderNodeType.MAP_MATRIX,
    model=MapMatrixData,
    system_message="""You are tasked with creating a 16x16 map for use in a videogame by representing
one or more of the nodes for the videogame ground layer.  The ground layer contains:
trees, grass, roads, foilage, water of various, sand.  The map should adhere to the description and you will
get updates on making it more arty."""
)

WorldBuilderEdgeRegistry.register_connection_types(CreationNode.group, WorldBuilderNodeType.MAP_ROOT, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_ROOT, WorldBuilderNodeType.MAP_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_ROOT, WorldBuilderNodeType.DESCRIPTION_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_MATRIX, WorldBuilderNodeType.DESCRIPTION_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])

class WorldBuilderCreator:
    def __init__(self, project_id: str):
        pass