from abc import ABC, abstractmethod
from typing import Iterator
from gstk.models.chatgpt import Message

from world_builder.map import MapRoot



class ContextEngineBase(ABC):
    def __init__(self, map_root: MapRoot):
        self.map_root = map_root

    @abstractmethod
    def get_child_matrix_creation_context(cell_identifier: str) -> Iterator[Message]:
        raise NotImplemented

    @abstractmethod
    def can_create_child_matrix(cell_identifier: str) -> bool:
        """
        Behavior: return boolean indicating whether a child matrix
        can be created for the given cell identifier.
        """
        raise NotImplementedError