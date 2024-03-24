

from typing import Iterator

from gstk.models.chatgpt import Message, Role

from world_builder.context.base import ContextEngineBase
from world_builder.context.engine import register_context_engine
from world_builder.graph_registry import MapRect
from world_builder.map import MapRoot


@register_context_engine("mar2024")
class March2024ContextEngine(ContextEngineBase):
    def get_child_matrix_creation_context(self,
            map_root: MapRoot, map_rect: MapRect) -> Iterator[Message]:
        messages: list[Message] = []
        #current_map_rect: MapRect = map_rect

        messages.extend([
            Message(
                role=Role.USER,
                content="Kamehameha!"
            )
        ])
        return messages

    def can_create_child_matrix(self, _map_root: MapRoot, _map_rect: MapRect) -> bool:
        return True

