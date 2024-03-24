from abc import ABC, abstractmethod
from typing import Iterator
from gstk.models.chatgpt import Message

from world_builder.map import MapRoot



class ContextEngineBase(ABC):
    @abstractmethod
    def get_child_matrix_creation_context(map_root: MapRoot, cell_identifier: str) -> Iterator[Message]:
        raise NotImplemented

    @abstractmethod
    def can_create_child_matrix(map_root: MapRoot, cell_identifier: str) -> bool:
        """
        Behavior: return boolean indicating whether a child matrix
        can be created for the given cell identifier.
        """
        raise NotImplementedError