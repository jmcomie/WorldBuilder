from enum import StrEnum
from pprint import pprint
from typing import Annotated
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
import sys


mcp = FastMCP(
    name="Worldbuilder Entity Graph Outline Server",
    instructions="""
    This server provides tools for creating and managing ontological outlines.
    It supports defining entity names and fundamental relationships. It's to be
    called in the initial stages of creating an ontology, where the focus is on
    establishing the basic structure and relationships between entities. It is
    also to be called in service of updating an existing ontology outline.
    """
)

# Nature of the ontological structure
# time
# relationship
# world scope
# location
# conceptual
# vertical horizontal

# orientation
# veritcal
# horizontal


class OntologicalOrientation(StrEnum):
    """How entities relate to each other structurally"""
    VERTICAL = "vertical"  # Parent-child, hierarchical relationships
    HORIZONTAL = "horizontal"  # Peer-to-peer, sibling relationships


class OntologicalCategory(StrEnum):
    """The domain or type of relationships in this ontology"""
    TIME = "time"  # Temporal relationships (before/after, during)
    HUMAN_RELATIONSHIPS = "human_relationships"  # Family, social, professional
    LOCATION = "location"  # Spatial, geographical relationships
    CONCEPTUAL = "conceptual"  # Abstract ideas, scientific concepts
    ORGANIZATIONAL = "organizational"  # Corporate, institutional structures
    BIOLOGICAL = "biological"  # Taxonomic, evolutionary relationships


class OntologicalDataStructure(StrEnum):
    """The underlying data structure for the ontology"""
    TREE = "tree"  # Hierarchical with single root (taxonomies)
    GRAPH = "graph"  # General graph (arbitrary connections)


class OntologicalOutlineBase(BaseModel):
    """Initial configuration for an ontology"""
    ontology_name: str = Field(description="Human-readable name for this ontology. Serves as unique identifier.")
    category: OntologicalCategory = Field(
        description="Primary domain - determines relationship types and inference rules." \
        "One of (with descriptions): " \
        "time (temporal): human_relationships (social), location (spatial)\n"
        "conceptual: (abstract), organizational (corporate), biological (taxonomic)\n"
        "human_relationships (social), location (spatial)\n"
        "conceptual (abstract), organizational (corporate), biological (taxonomic)\n"
        "biological (taxonomic, evolutionary)\n"
        "organizational (corporate, institutional)\n"
    )
    data_structure: OntologicalDataStructure = Field(
        description="How entities connect - affects allowed relationship patterns"
    )
    orientation: OntologicalOrientation = Field(
        description="Primary relationship direction - affects visualization and traversal"
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the ontology's purpose"
    )


class OntologyOutlineInitial(OntologicalOutlineBase):
    """Initial configuration for an ontology"""
    seed_entities: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Initial entities to bootstrap the ontology (max 10)"
    )

class OntologyOutline(OntologicalOutlineBase):
    """Initial configuration for an ontology"""
    seed_entities: list[str] = Field(
        description="Initial entities to bootstrap the ontology (max 10)"
    )


@mcp.tool()
def initialize_ontology_outline(outline: OntologyOutlineInitial) -> dict:
    """
    Initialize a new ontology with basic structure and seed entities.
    Call this tool or open an existing outline to start building an ontology.

    The user prompt informing the call of this tool can be abstract, personal, or
    in the form of a concrete directive. If it implies the creation of complex
    relationships, start here, and then move to other MCP servers and tools as
    needed.
    """
    # Store the outline and return a unique ID
    ontology_id = f"{outline.name}"

    print(f"Initializing ontology outline: {pprint(outline.model_dump())}", file=sys.stderr)
    # review code below
    return {
        "ontology_id": ontology_id,
        "status": "initialized",
        "outline": outline.model_dump(),
        "next_steps": [
            "add_entity",
            "add_relationship", 
            "define_relationship_type"
        ]
    }


#@mcp.tool
#def list_ontology_names():
#    pass


class ChainDirection(StrEnum):
    """Direction of the chain relative to the anchor entity"""
    BEFORE = "before"  # Hook the last entity to the anchor entity
    AFTER = "after"  # Hook the first entity to the anchor entity


class SupplementalOntologicalChain(BaseModel):
    """A chain of entities that can be added to an ontology outline"""
    anchor_entity: str = Field(
        description="The entity to which the chain will be attached"
    )
    new_entity_names: list[str] = Field(
        min_length=1,
        max_length=10,
        description="List of new entities to add in the chain"
    )
    chain_direction: ChainDirection = Field(
        description="Direction of the chain relative to the anchor entity"
    )


#@mcp.tool
#def add_ontological_chain_to_outline(outline_name: str, anchor_entity: str, new_entity_names: list[str], chain_direction: ChainDirection):
#    pass


#@mcp.tool
#def get_ontology_outline(outline_name: str) -> OntologyOutline:
#    pass


class ChainDirection(StrEnum):
    before = "BEFORE"
    after = "AFTER"



# Add support server selection based on description and have each server in the graph context communicate server switches or handoffs.



# This is an exploration of ontology creation for the purpose of developing
# intuitions for the relationships between text input and corresponding
# entity graphs.
#
# Option: use a sqlite database for storing and retrieving ontology
# information from the backend.


def create_entity_ontological_structure(vertical_entities):
    pass


def expand_ontology(outline):
    pass


def get_remaining():
    pass


# Entry point
if __name__ == "__main__":
    mcp.run()