from abc import ABC, abstractmethod
import typing

from gstk.models.chatgpt import Message

class ContextBuffer(ABC):
    def __init__(self):
        self._buffer: list[Message] = []

    # Note: no seek position is maintained.
    def add(self, message: Message) -> int:
        assert isinstance(message, Message)
        self._buffer.append(message)
        return len(self._buffer) - 1

    @abstractmethod
    def save(self):
        ...

    def delete(self, index: int):
        self._buffer.pop(index)

    def list(self) -> list[Message]:
        return self._buffer

    def move(self, from_index: int, to_index: int):
        if from_index == to_index:
            return
        message: Message = self._buffer.pop(from_index)
        if from_index < to_index:
            to_index -= 1
        self._buffer.insert(to_index, message)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)
