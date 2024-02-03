from enum import StrEnum
from typing import Literal, Optional, get_args
import gstk.creation.api
import gstk.shim
from pydantic import BaseModel, Field, field_validator, root_validator

from gstk.user_registries.story.graph_registry import StoryNodeRegistry, StoryEdgeRegistry
from gstk.graph.registry import EdgeRegistry, NodeRegistry
from gstk.creation.graph_registry import CreationNode
from gstk.graph.system_graph_registry import SystemEdgeType
from gstk.creation.graph_registry import Message, Role


WorldBuilderNodeRegistry: NodeRegistry = StoryNodeRegistry.clone()
WorldBuilderEdgeRegistry: EdgeRegistry = StoryEdgeRegistry.clone()

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
    
class MapRect(BaseModel):
    x: int
    y: int
    width: int
    height: int
    
class MapRectMetadata(BaseModel):
    map_rect: Optional[MapRect] = None

    class Config:
        @classmethod
        def json_schema_extra(cls, schema, model):
            # This removes 'hidden_field' from the schema
            schema['properties'].pop('map_rect', None)

        extra = "forbid"
        use_enum_values = True


class MapMatrixData(MapRectMetadata):
    """Data describing the name and tiles of a map in a videogame."""
    width: int = Field(default=None, description="The width of the map.")
    height: int = Field(default=None, description="The height of the map.")
    tiles: list[list[int]] = Field(default=None, description="The tiles on the map.")


class DescriptionMatrixData(MapRectMetadata):
    """Data describing a description of a map in a videogame."""
    width: int = Field(default=None, description="The width of the map.")
    height: int = Field(default=None, description="The height of the map.")
    tiles: list[list[str]] = Field(default=None, description="The sub-map cell description.")


DrawDimensionInt = Literal[3, 4, 5, 6, 7, 8]


class MapRoot(BaseModel):
    name: str
    asset_name: str
    layer_names: Optional[list[str]] = None
    draw_width: DrawDimensionInt
    draw_height: DrawDimensionInt
    width: int
    height: int
 
    class Config:
        extra = "forbid"
        use_enum_values = True

    @root_validator(pre=True)
    @classmethod
    def validate_all_fields_at_the_same_time(cls, field_values):
        if field_values["draw_width"] != field_values["draw_height"]:
            raise ValueError(f"Draw width {field_values['draw_width']} and draw height {field_values['draw_height']} must be the same.")
        if field_values["width"] % field_values["draw_width"] != 0:
            raise ValueError(f"Width {field_values['width']} must be a multiple of draw_width {field_values['draw_width']}.")
        if field_values["height"] % field_values["draw_height"] != 0:
            raise ValueError(f"Height {field_values['height']} must be a multiple of draw_height {field_values['draw_height']}.")
        return field_values


class MapLayerRoot(BaseModel):
    layer_name: str


WorldBuilderNodeRegistry.register_node(
    WorldBuilderNodeType.MAP_MATRIX,
    model=MapMatrixData,
    system_message="""You are tasked with creating a map for use in a videogame by representing
one or more of the nodes for the videogame ground layer.  The ground layer contains:
trees, grass, roads, foilage, water of various, sand.  The map should adhere to the description and you will
get updates on making it more arty."""
)


WorldBuilderNodeRegistry.register_node(
    WorldBuilderNodeType.DESCRIPTION_MATRIX,
    model=DescriptionMatrixData,
)

WorldBuilderEdgeRegistry.register_connection_types(CreationNode.group, WorldBuilderNodeType.MAP_ROOT, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_ROOT, WorldBuilderNodeType.MAP_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_ROOT, WorldBuilderNodeType.DESCRIPTION_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])
WorldBuilderEdgeRegistry.register_connection_types(WorldBuilderNodeType.MAP_MATRIX, WorldBuilderNodeType.DESCRIPTION_MATRIX, [SystemEdgeType.contains, SystemEdgeType.references])

class WorldBuilderCreator:
    def __init__(self, project_id: str):
        pass