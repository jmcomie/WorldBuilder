from enum import StrEnum
from typing import Literal, Optional, get_args
import gstk.creation.api
import gstk.shim
from openai import ChatCompletion
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


GraphRegistry.register_connection_type(SystemNodeType.PROJECT,
                                       WorldBuilderNodeType.MAP_ROOT, SystemEdgeType.CONTAINS)


class MapRect(BaseModel):
    x: int
    y: int
    width: int
    height: int
    layer: Optional[str] = None

    def to_tuple(self):
        # Include type as string, layer, and rect.
        return (self.x, self.y, self.width, self.height, self.layer, str(type(self)))

    @classmethod
    def from_tuple(cls, tup):
        # Include type as string, layer, and rect.
        return cls(x=tup[0], y=tup[1], width=tup[2], height=tup[3], layer=tup[4])

class MapRectMetadata(BaseModel):
    map_rect: Optional[MapRect] = None

    class Config:
        @classmethod
        def json_schema_extra(cls, schema, model):
            # Omit map_rect from JSON schema representation since we define
            # it in code, not via LLM.
            schema['properties'].pop('map_rect', None)

        extra = "forbid"
        use_enum_values = True


@GraphRegistry.node_type(WorldBuilderNodeType.MAP_ROOT, child_types=[WorldBuilderNodeType.DESCRIPTION_MATRIX, WorldBuilderNodeType.MAP_MATRIX])
class MapRootData(BaseModel):
    name: str
    layer_names: Optional[list[str]] = None
    width: int
    height: int
    draw_diameter: DrawDiameterInt
    description: Optional[str] = None
    readonly: Optional[bool] = False
    # Chat completion data key is the map rect as a string.
    # The values are lists of chat completions used.
    map_rect_chat_completions: dict[str, list[dict]] = Field(default_factory=dict, description="Chat completions for each map rect.")

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




