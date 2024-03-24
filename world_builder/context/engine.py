from typing import Type
from world_builder.context.base import ContextEngineBase
from world_builder.map import MapRoot

_context_engines: dict[str, ContextEngineBase] = {}
class InvalidContextEngine(Exception): pass
class InvalidCellIdentifier(Exception): pass

DEFAULT_CONTEXT_ENGINE_NAME: str = "feb2024"


def register_context_engine(name: str):
    if name in _context_engines:
        raise InvalidContextEngine(f"Context engine {name} already registered.")
    def context_engine(context_engine_cls: Type):
        if not issubclass(context_engine_cls, ContextEngineBase):
            raise InvalidContextEngine(f"Context engine {context_engine_cls} is not a subclass of ContextEngineBase.")
        try:
            _context_engines[name] = context_engine_cls()
        except Exception as e:
            raise InvalidContextEngine(f"Error registering context engine {context_engine_cls}: {e}")
        return context_engine_cls
    return context_engine


def has_context_engine(name: str) -> bool:
    return name in _context_engines


def get_context_engine_by_name(name: str) -> ContextEngineBase:
    if name not in _context_engines:
        raise InvalidContextEngine(f"Context engine {name} not found.")
    return _context_engines[name]


def list_context_engines() -> list[str]:
    return list(_context_engines.keys())


def get_context_engine_for_map_root(map_root: MapRoot):
    if not map_root.data.context_engine_name:
        try:
            return get_context_engine_by_name(DEFAULT_CONTEXT_ENGINE_NAME)
        except:
            raise InvalidContextEngine(f"Default context engine {DEFAULT_CONTEXT_ENGINE_NAME} not found.")
    return get_context_engine_by_name(map_root.data.context_engine_name)