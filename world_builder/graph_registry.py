from enum import StrEnum
from typing import Literal, Optional, get_args
import gstk.creation.api
import gstk.shim
from pydantic import BaseModel, Field, field_validator, root_validator

from gstk.graph.registry import GraphRegistry
from gstk.creation.graph_registry import CreationNode
from gstk.graph.registry import SystemEdgeType, SystemNodeType
from gstk.creation.graph_registry import Message, Role



class WorldBuilderNodeType(StrEnum):
    MAP_ROOT = "world_builder.map_root"
    MAP_MATRIX = "world_builder.map_matrix"
    DESCRIPTION_MATRIX = "world_builder.description_matrix"
    WORLD_BUILDER_ALL = "world_builder.*"


DrawDiameterInt = Literal[3, 4, 5, 6, 7, 8, 9, 10]


@GraphRegistry.node_type(WorldBuilderNodeType.MAP_ROOT, child_types=[WorldBuilderNodeType.DESCRIPTION_MATRIX, WorldBuilderNodeType.MAP_MATRIX])
class MapRootData(BaseModel):
    name: str
    layer_names: Optional[list[str]] = None
    draw_diameter: DrawDiameterInt
    width: int
    height: int
    description: Optional[str] = None

    class Config:
        extra = "forbid"
        use_enum_values = True

    @root_validator(pre=True)
    @classmethod
    def validate_all_fields(cls, field_values):
        if field_values["width"] % field_values["draw_diameter"] != 0:
            raise ValueError(f"Width {field_values['width']} must be a multiple of draw_diameter {field_values['draw_diameter']}.")
        if field_values["height"] % field_values["draw_diameter"] != 0:
            raise ValueError(f"Height {field_values['height']} must be a multiple of draw_diameter {field_values['draw_diameter']}.")
        return field_values


GraphRegistry.register_connection_type(SystemNodeType.PROJECT, WorldBuilderNodeType.MAP_ROOT, SystemEdgeType.contains)


class MapRect(BaseModel):
    x: int
    y: int
    width: int
    height: int


class MapRectMetadata(BaseModel):
    _map_rect: Optional[MapRect] = None

    class Config:
        extra = "forbid"
        use_enum_values = True


@GraphRegistry.node_type(WorldBuilderNodeType.MAP_MATRIX)
class MapMatrixData(MapRectMetadata):
    """Data describing the name and tiles of a map in a videogame."""
    _system_message: str = """You are tasked with creating a map for use in a videogame by representing
one or more of the nodes for the videogame ground layer.  The ground layer contains:
trees, grass, roads, foilage, water of various, sand.  The map should adhere to the description and you will
get updates on making it more arty."""
    tiles: list[list[int]] = Field(default=None, description="The tiles on the map.")


@GraphRegistry.node_type(WorldBuilderNodeType.DESCRIPTION_MATRIX, child_types=[WorldBuilderNodeType.DESCRIPTION_MATRIX, WorldBuilderNodeType.MAP_MATRIX])
class DescriptionMatrixData(MapRectMetadata):
    """Data describing a description of a map in a videogame."""
    _system_message: str = "Data describing a description of a map in a videogame."
    tiles: list[list[str]] = Field(default=None, description="The sub-map cell description.")




