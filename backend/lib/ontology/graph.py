import datetime
from collections import deque
from functools import cache, reduce
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Iterator, Optional, Type, TypeVar, Union, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, case, create_engine
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, object_session, relationship, sessionmaker

from lib.ontology.registry import (
    EdgeCardinality,
    EdgeTypeData,
    GraphRegistry,
    NodeTypeData,
    ProjectProperties,
    SystemEdgeType,
    SystemNodeType,
    node_type_matches_type_in_policy_list,
)

# Define the base model
Base = declarative_base()
T = TypeVar("T")
DB_FILENAME: str = "graph.sqlite"


@cache
def get_engine(path: str):
    engine = create_engine(f"sqlite:///{path}")
    # Create all tables in the engine
    Base.metadata.create_all(engine)
    return engine


def get_session(path: str):
    Session = sessionmaker(bind=get_engine(path))
    return Session()


def breadth_first_traversal(root_node: T, node_children_getter: Callable[[T], list[T]]) -> Iterator[T]:
    queue: deque[T] = deque([root_node])
    while queue:
        current_node: T = queue.popleft()
        yield current_node
        queue.extend(node_children_getter(current_node))


class SessionBoundMixin:
    def _session(self) -> Session:
        session = object_session(self)
        if not session:
            raise RuntimeError("Object is not bound to a session.")
        return session

    def save(self):
        s = self._session()
        s.add(self)
        s.commit()

    def delete(self):
        s = self._session()
        s.delete(self)
        s.commit()


##
## Graph helper functions
##

class GraphOps:
    @staticmethod
    def add_node(session: Session, data: BaseModel, vector: Optional[list[float]] = None) -> "DataGraphNode":
        node_types: str = list(GraphRegistry.get_node_types(type(data)))
        if node_types:
            node_type: str = node_types[0]
        else:
            raise ValueError(f"No node type found for model {type(data)}")
        check_node_add(session, node_type)
        node: DataGraphNode = DataGraphNode(
            node_type=node_type,
            vector=vector,
            _data=data.model_dump() if data else None,
        )
        session.add(node)
        session.flush()
        return node

    @staticmethod
    def add_edge(
        session: Session,
        edge_type: str,
        from_id: int,
        to_id: int,
        sort_idx: Optional[int] = None,
    ) -> "DataGraphEdge":
        check_edge_add(session, edge_type, GraphOps.get_node(session, from_id), GraphOps.get_node(session, to_id))
        edge: DataGraphEdge = DataGraphEdge(edge_type=edge_type, from_id=from_id, to_id=to_id, sort_idx=sort_idx)
        session.add(edge)
        session.flush()
        return edge

    @staticmethod
    def delete_node(session: Session, node_id: int):
        node: Optional[DataGraphNode] = session.query(DataGraphNode).filter_by(id=node_id).first()
        if node:
            session.delete(node)

    @staticmethod
    def delete_edge(session: Session, edge_id: int):
        session.query(DataGraphEdge).filter_by(id=edge_id).delete()

    @staticmethod
    def delete_edge_by_nodes(session: Session, from_id: int, to_id: int):
        session.query(DataGraphEdge).filter_by(from_id=from_id, to_id=to_id).delete()

    @staticmethod
    def get_node(session: Session, node_id: int) -> Optional["DataGraphNode"]:
        return session.query(DataGraphNode).filter_by(id=node_id).first()

    @staticmethod
    def get_edge_by_id(session: Session, edge_id: int) -> Optional["DataGraphEdge"]:
        return session.query(DataGraphEdge).filter_by(id=edge_id).first()

    @staticmethod
    def get_edge(session: Session, from_id: int, to_id: int, edge_type: str) -> Optional["DataGraphEdge"]:
        result: Optional[DataGraphEdge] = (
            session.query(DataGraphEdge).filter_by(from_id=from_id, to_id=to_id, edge_type=edge_type).first()
        )
        return result

    @staticmethod
    def list_nodes(session: Session, node_type_filter: Optional[list[str]] = None) -> Iterator["DataGraphNode"]:
        query = session.query(DataGraphNode)
        if node_type_filter:
            query = query.filter(DataGraphNode.node_type.in_(node_type_filter))
        yield from query

    @staticmethod
    def list_edges(session: Session, edge_type_filter: Optional[list[str]] = None) -> Iterator["DataGraphEdge"]:
        query = session.query(DataGraphEdge)
        if edge_type_filter:
            query = query.filter(DataGraphEdge.edge_type.in_(edge_type_filter))
        yield from query

##
## End of graph helper functions
##

class DataGraph:
    """
    UserDataGraph
    """

    def __init__(self, project_resource_location: Path):
        self._project_resource_location = Path(project_resource_location)
        self._project_node_id: Optional[int] = None

    def list_project_nodes(self, session: Session) -> Generator["DataGraphNode", None, None]:
        for node in session.query(DataGraphNode).filter_by(node_type=SystemNodeType.PROJECT):
            yield node

    def get_project_node(self, session: Session) -> "DataGraphNode":
        projects: list[DataGraphNode] = list(self.list_project_nodes(session))
        if len(projects) == 0:
            raise ValueError("No project node found. Please create a new project.")
        if len(projects) > 1:
            raise ValueError("Multiple project nodes found. There should only be one.")
        return projects[0]

    def create_session(self) -> Session:
        self._project_resource_location.mkdir(parents=True, exist_ok=True)
        return get_session(str(self._project_resource_location / DB_FILENAME))


class DataGraphEdge(Base, SessionBoundMixin):
    __tablename__ = "edge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    edge_type: Mapped[str] = mapped_column(String(32))

    from_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("node.id", ondelete="CASCADE"),
        nullable=False
    )
    to_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("node.id", ondelete="CASCADE"),
        nullable=False
    )

    sort_idx: Mapped[int] = mapped_column(Integer, nullable=True)

    in_node: Mapped["DataGraphNode"] = relationship(
        "DataGraphNode",
        foreign_keys=[from_id],
        back_populates="out_edges",
        passive_deletes=True,
    )
    out_node: Mapped["DataGraphNode"] = relationship(
        "DataGraphNode",
        foreign_keys=[to_id],
        back_populates="in_edges",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("from_id", "to_id", name="uq_from_id_to_id"),
        Index("idx_from_id", "from_id"),
        Index("idx_to_id", "to_id"),
    )


class DataGraphNode(Base, Generic[T], SessionBoundMixin):
    __tablename__ = "node"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.now(datetime.timezone.utc)
    )
    modified_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )
    deleted_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=True, default=None)
    node_type: Mapped[str] = mapped_column(String(32))
    vector = Column(JSON)
    _data: Mapped[dict] = mapped_column("data", JSON, default=dict)

    in_edges: Mapped[list["DataGraphEdge"]] = relationship(
        "DataGraphEdge",
        foreign_keys=[DataGraphEdge.to_id],
        back_populates="in_node",
        order_by=case((DataGraphEdge.sort_idx != None, DataGraphEdge.sort_idx), else_=DataGraphEdge.id),
    )
    out_edges: Mapped[list["DataGraphEdge"]] = relationship(
        "DataGraphEdge",
        foreign_keys=[DataGraphEdge.from_id],
        back_populates="out_node",
        order_by=case((DataGraphEdge.sort_idx != None, DataGraphEdge.sort_idx), else_=DataGraphEdge.id),
    )

    def __repr__(self) -> str:
        return f"<Node type={self.node_type}> object at {hex(id(self))}"

    @property
    def graph(self) -> DataGraph:
        return self._graph

    @property
    def data(self) -> T:
        return cast(T, GraphRegistry.get_node_type_data(self.node_type).model.model_validate(self._data))

    @data.setter
    def data(self, value: T):
        # Should be an instance of register model -- check.
        if not isinstance(value, GraphRegistry.get_node_type_data(self.node_type).model):
            raise ValueError(f"Node data must be an instance of {self.node_type}, not {type(value)}")
        self._data = value.model_dump()

    def _check_node_data_against_filters(self, node: "DataGraphNode", filters: list[dict]) -> bool:
        data_as_dict = node.data.model_dump() if node.data else {}
        for filter in filters:
            for key, value in filter.items():
                if key not in data_as_dict or data_as_dict[key] != value:
                    return False
        return True

    def list_children(self, node_types: Optional[list[str]] = None, filters=[]) -> Iterator["DataGraphNode"]:
        for node in self.walk_tree(yield_node_types=node_types, max_depth=1):
            if node.id == self.id:
                continue
            if not self._check_node_data_against_filters(node._sqlalchemy_obj, filters):
                continue
            yield node

    @property
    def parent(self) -> Optional["DataGraphNode"]:
        for edge in self._sqlalchemy_obj.in_edges:
            if edge.edge_type == SystemEdgeType.CONTAINS:
                return edge.in_node
        return None

    def create_child(self, data: BaseModel) -> "DataGraphNode":
        node: DataGraphNode = GraphOps.add_node(self._session(), data)
        GraphOps.add_edge(self._session(), SystemEdgeType.CONTAINS, self.id, node.id)
        return node

    def create_child_reference(
        self,
        reference_to: Union["DataGraphNode", int],
        conflict_filter: Optional[dict] = None,
        overwrite_on_conflict: bool = False,
    ):
        reference_node_id: int = reference_to if isinstance(reference_to, int) else reference_to.id
        edge: DataGraphEdge = GraphOps.add_edge(self._session(), SystemEdgeType.CONTAINS, self.id, reference_node_id)

    def delete_child(self, child: Union[int, "DataGraphNode"]):
        # Note: could be optimized by using a query
        # or only loading the edge(s).
        child_id: int = child if isinstance(child, int) else child.id
        for _child in self.list_children():
            if _child.id == child_id and _child.parent and _child.parent.id == self.id:
                GraphOps.delete_node(self._session(), child_id)
                return

    def delete(self):
        GraphOps.delete_node(self._session(), self.id)

    def _get_descendent_nodes(self, filters: list[dict], seen: set) -> Iterator["DataGraphNode"]:
        for edge in self._sqlalchemy_obj.out_edges:
            assert isinstance(edge, DataGraphEdge)
            if edge.out_node.id in seen:
                continue
            seen.add(edge.out_node.id)
            assert isinstance(edge, DataGraphEdge)
            assert isinstance(edge.out_node, DataGraphNode)
            if self._check_node_data_against_filters(edge.out_node, filters):
                yield edge.out_node
            if edge.edge_type == SystemEdgeType.CONTAINS:
                yield from edge.out_node._get_descendent_nodes(filters, seen)

    def get_descendent_nodes(self, filters: list[dict]) -> Iterator["DataGraphNode"]:
        if not filters:
            raise ValueError("Filters must be provided.")
        yield from self._get_descendent_nodes(filters, set())

    def save(self, session: Optional[Session] = None, commit=False):
        assert isinstance(self.session, Session)
        if session is not None:
            self = session.merge(self)
        else:
            session = object_session(self)
        if session is None:
            raise ValueError("Session must be provided or set on the DataGraphNode instance.")
        session.flush()
        if commit:
            if not self.session.is_active:
                raise ValueError("Session is not active. Cannot commit changes.")
            session.commit()

    def walk_tree(
        self,
        yield_node_types: Optional[list[str]] = None,
        descend_into_types: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = [SystemEdgeType.CONTAINS, SystemEdgeType.REFERENCES],
        max_depth: Optional[int] = None,
    ) -> Generator["DataGraphNode", None, None]:
        """
        Cycle-safe but may traverse nodes closer to the project root than the given node.
        That logic will be inherited from a proper graph data object.
        """
        if max_depth is not None and max_depth < 0:
            return
        if yield_node_types is not None and node_type_matches_type_in_policy_list(
            self.node_type, tuple(yield_node_types)
        ):
            yield self
        if max_depth == 0:
            return
        if max_depth is not None:
            max_depth -= 1
        seen: set[int] = {self.id}
        yield from _walk_tree_helper(
            self,
            seen,
            descend_into_types=descend_into_types,
            yield_node_types=yield_node_types,
            edge_type_filter=edge_type_filter,
            max_depth=max_depth,
        )

    def build_vector_index(
        self,
        node_type_filter: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = None,
        descend_into_types: Optional[list[str]] = None,
    ) -> tuple["faiss.IndexFlatIP", list[int]]:
        """
        Build a vector index for descendent nodes.
        """
        index: Optional[faiss.IndexFlatIP] = None
        node_ids: list[int] = []
        for node in self.walk_tree(
            descend_into_types=descend_into_types,
            yield_node_types=node_type_filter,
            edge_type_filter=edge_type_filter,
        ):
            if node.vector is None:
                continue
            if index is None:
                dim: int = np.array(node.vector).shape[0]
                index = faiss.IndexFlatIP(dim)
            assert index is not None
            index.add(np.array([node.vector]))
            node_ids.append(node.id)
        assert index is not None
        return index, node_ids

    def find(
        self,
        query_vector: np.ndarray,
        count: int = 10,
        node_type_filter: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = None,
        descend_into_types: Optional[list[str]] = None,
        similarity_threshold: Optional[list[float]] = None,
    ) -> list[tuple[int, float]]:
        index, node_ids = self.build_vector_index(
            node_type_filter=node_type_filter,
            edge_type_filter=edge_type_filter,
            descend_into_types=descend_into_types,
        )
        query_vector = np.array(query_vector)
        # D: distances/similarities, I: indices
        # 0 is the query vector index
        D, I = index.search(np.array([query_vector]), count)
        assert len(I) == 1
        return [
            (node_ids[I[0][i]], D[0][i])
            for i in range(len(I[0]))
            if similarity_threshold is None or D[i] >= similarity_threshold
        ]


def _walk_tree_helper(
    iter_node: "DataGraphNode",
    seen: set,
    descend_into_types: Optional[list[str]] = None,
    yield_node_types: Optional[list[str]] = None,
    edge_type_filter: Optional[list[str]] = None,
    max_depth: Optional[int] = None,
) -> Generator["DataGraphNode", None, None]:
    if max_depth is not None:
        if max_depth < 0:
            return
        max_depth -= 1
    for edge in iter_node.out_edges:
        assert isinstance(edge, DataGraphEdge)
        # Dangling edges have been occurring. Need to review foreign key constraint
        # behavior.
        if edge.out_node is None:
            continue
        assert isinstance(edge.out_node, DataGraphNode)
        if edge.out_node.id in seen:
            continue
        seen.add(edge.out_node.id)
        if edge.out_node.deleted_at:
            continue
        if yield_node_types is None or node_type_matches_type_in_policy_list(
            edge.out_node.node_type, tuple(yield_node_types)
        ):
            yield DataGraphNode(edge.out_node, iter_node.graph, iter_node.session)
        if descend_into_types is None or node_type_matches_type_in_policy_list(
            edge.out_node.node_type, tuple(descend_into_types)
        ):
            yield from _walk_tree_helper(
                DataGraphNode(edge.out_node, iter_node.graph, iter_node.session),
                seen,
                descend_into_types=descend_into_types,
                yield_node_types=yield_node_types,
                edge_type_filter=edge_type_filter,
                max_depth=max_depth,
            )


def check_node_add(session: Session, node_type: str):
    node_type_data: NodeTypeData = GraphRegistry.get_node_type_data(node_type)

    if not node_type_data.write_allowed:
        raise ValueError(f"Node type {node_type} is not enabled")

    if node_type_data.instance_limit is not None:
        instance_count: int = len(list(GraphOps.list_nodes(session, node_type_filter=[node_type])))
        if instance_count >= node_type_data.instance_limit:
            raise ValueError(f"Node type {node_type} has reached its instance limit")


def check_edge_add(
    session: Session,
    edge_type: str,
    from_node: Optional[DataGraphNode] = None,
    to_node: Optional[DataGraphNode] = None,
):
    edge_type_data: EdgeTypeData = GraphRegistry.get_edge_type_data(edge_type)

    if from_node is None or to_node is None:
        raise ValueError("from_node and to_node must be provided")

    if not edge_type_data.write_allowed:
        raise ValueError(f"Edge type {edge_type} is not enabled")

    if edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_MANY:
        # Look at all in edges of to_node.  Should be no edges of this type.
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.ONE_TO_ONE:
        # Look at relevant edges from both nodes. Should be no edges of this type.
        for _ in from_node.get_out_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
        for _ in to_node.get_in_nodes(edge_type_filter=[edge_type]):
            raise ValueError(f"Edge type {edge_type} already exists between {from_node.id} and {to_node.id}")
    elif edge_type_data.edge_cardinality == EdgeCardinality.MANY_TO_MANY:
        pass  # No checks needed.
