"""
Worldbuilder Demo MCP Server

A demonstration MCP server showcasing FastMCP features without performing
any actual mutations. Provides tools, resources, and prompts for worldbuilding
concepts with mock data responses.
"""

import random
import json
from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from fastmcp import FastMCP, Context

# Initialize the MCP server
mcp = FastMCP(
    name="Worldbuilder Demo MCP Server",
    instructions="""
    This is a demonstration MCP server for worldbuilding tasks.
    All operations return mock data and perform no actual mutations.
    
    Available capabilities:
    - Tools for analyzing world elements and generating story seeds
    - Resources providing templates and statistics
    - Prompts for creative writing assistance
    """
)


# Pydantic models for structured data
class WorldElement(BaseModel):
    """Model for a world element (character, location, event)"""
    name: str = Field(description="Name of the element")
    type: Literal["character", "location", "event", "item", "concept"]
    description: Optional[str] = Field(default=None, description="Brief description")
    tags: List[str] = Field(default_factory=list, description="Associated tags")


class StoryParameters(BaseModel):
    """Parameters for story generation"""
    genre: Literal["fantasy", "sci-fi", "mystery", "romance", "horror"] = "fantasy"
    tone: Literal["dark", "light", "neutral", "comedic", "serious"] = "neutral"
    complexity: Annotated[int, Field(ge=1, le=10)] = 5


# Tools
@mcp.tool
async def analyze_world_element(
    element: WorldElement,
    ctx: Context,
    depth: Annotated[str, Field(description="Analysis depth level")] = "standard"
) -> Dict[str, Any]:
    """
    Analyzes a world element and returns insights about its role and connections.
    This is a demo tool that returns mock analysis data.
    """
    await ctx.info(f"Analyzing {element.type}: {element.name}")
    
    # Generate mock analysis
    analysis = {
        "element_name": element.name,
        "element_type": element.type,
        "analysis_depth": depth,
        "timestamp": datetime.now().isoformat(),
        "insights": {
            "narrative_importance": random.choice(["high", "medium", "low"]),
            "connection_count": random.randint(1, 10),
            "development_stage": random.choice(["concept", "developing", "established"]),
            "consistency_score": round(random.uniform(0.7, 1.0), 2)
        },
        "recommendations": [
            "Consider adding more backstory",
            "Explore relationships with other elements",
            "Develop unique characteristics"
        ][:random.randint(1, 3)],
        "related_elements": [
            f"Element_{i}" for i in range(random.randint(0, 5))
        ]
    }
    
    await ctx.report_progress(100, 100, "Analysis complete")
    return analysis


@mcp.tool
def calculate_narrative_distance(
    element1_name: str,
    element2_name: str,
    distance_type: Literal["temporal", "spatial", "relational", "thematic"] = "relational"
) -> Dict[str, Any]:
    """
    Calculates the narrative distance between two story elements.
    Returns mock distance calculations for demonstration.
    """
    # Mock distance calculation
    base_distance = random.uniform(0.1, 10.0)
    
    distance_modifiers = {
        "temporal": 1.2,
        "spatial": 1.0,
        "relational": 0.8,
        "thematic": 1.5
    }
    
    final_distance = base_distance * distance_modifiers[distance_type]
    
    return {
        "element1": element1_name,
        "element2": element2_name,
        "distance_type": distance_type,
        "raw_distance": round(base_distance, 2),
        "adjusted_distance": round(final_distance, 2),
        "interpretation": {
            "closeness": "very close" if final_distance < 2 else "close" if final_distance < 5 else "distant",
            "narrative_impact": "strong" if final_distance < 3 else "moderate" if final_distance < 7 else "weak",
            "suggested_connections": random.randint(1, 5)
        },
        "visualization_hint": "Use force-directed graph with edge weight inversely proportional to distance"
    }


@mcp.tool
def generate_story_seed(
    parameters: StoryParameters,
    include_conflict: bool = True,
    seed_length: Literal["brief", "standard", "detailed"] = "standard"
) -> Dict[str, Any]:
    """
    Generates a story seed based on the provided parameters.
    This is a demo tool that creates random story elements.
    """
    # Mock story components based on genre
    genre_elements = {
        "fantasy": {
            "settings": ["ancient kingdom", "floating city", "enchanted forest", "dragon's lair"],
            "characters": ["reluctant mage", "exiled prince", "forest guardian", "sky pirate"],
            "items": ["cursed amulet", "living sword", "memory crystal", "portal key"]
        },
        "sci-fi": {
            "settings": ["space station", "colony ship", "quantum lab", "alien megastructure"],
            "characters": ["rogue AI", "time refugee", "gene-hacker", "void navigator"],
            "items": ["quantum drive", "consciousness backup", "nano-swarm", "gravity lens"]
        },
        "mystery": {
            "settings": ["fog-shrouded manor", "underground city", "isolated island", "ancient library"],
            "characters": ["amnesiac detective", "cryptic informant", "shadow broker", "memory thief"],
            "items": ["encrypted journal", "missing photograph", "strange key", "coded message"]
        },
        "romance": {
            "settings": ["seaside town", "mountain retreat", "art studio", "book café"],
            "characters": ["mysterious artist", "returning traveler", "local historian", "garden designer"],
            "items": ["lost letter", "shared journal", "music box", "painted portrait"]
        },
        "horror": {
            "settings": ["abandoned asylum", "cursed village", "underground tunnels", "mirror dimension"],
            "characters": ["paranormal investigator", "possessed medium", "cult survivor", "dream walker"],
            "items": ["bone whistle", "shadow mirror", "cursed book", "soul jar"]
        }
    }
    
    elements = genre_elements[parameters.genre]
    
    seed = {
        "genre": parameters.genre,
        "tone": parameters.tone,
        "complexity": parameters.complexity,
        "seed_id": f"SEED_{random.randint(1000, 9999)}",
        "primary_setting": random.choice(elements["settings"]),
        "protagonist": random.choice(elements["characters"]),
        "key_item": random.choice(elements["items"]),
        "opening_scene": f"The {random.choice(elements['characters'])} discovers a {random.choice(elements['items'])} in the {random.choice(elements['settings'])}."
    }
    
    if include_conflict:
        conflicts = [
            "a hidden truth that changes everything",
            "a choice between two impossible options",
            "a countdown to an irreversible event",
            "a betrayal from within their circle",
            "a power that comes with a terrible cost"
        ]
        seed["central_conflict"] = random.choice(conflicts)
    
    if seed_length == "detailed":
        seed["supporting_cast"] = [random.choice(elements["characters"]) for _ in range(3)]
        seed["subplot_seeds"] = [
            "A mysterious message arrives",
            "An old enemy returns",
            "A secret alliance forms"
        ][:random.randint(1, 3)]
    
    return seed


@mcp.tool
def validate_world_consistency(
    element_names: List[str],
    validation_rules: Optional[List[str]] = None,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Validates world consistency across multiple elements.
    Returns mock validation results for demonstration.
    """
    if validation_rules is None:
        validation_rules = [
            "timeline_consistency",
            "character_relationships",
            "location_accessibility",
            "technology_levels",
            "magic_system_rules"
        ]
    
    # Generate mock validation results
    results = {
        "validation_id": f"VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "elements_checked": element_names,
        "rules_applied": validation_rules,
        "strict_mode": strict_mode,
        "overall_consistency": random.uniform(0.85, 0.98) if not strict_mode else random.uniform(0.70, 0.90),
        "issues_found": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Generate some mock issues
    possible_issues = [
        {
            "type": "timeline_conflict",
            "severity": "medium",
            "elements": random.sample(element_names, min(2, len(element_names))),
            "description": "Temporal paradox detected between elements"
        },
        {
            "type": "relationship_gap",
            "severity": "low",
            "elements": random.sample(element_names, min(2, len(element_names))),
            "description": "Missing relationship definition"
        },
        {
            "type": "rule_violation",
            "severity": "high" if strict_mode else "medium",
            "elements": random.sample(element_names, 1),
            "description": "Element violates established world rules"
        }
    ]
    
    num_issues = random.randint(0, 2 if not strict_mode else 4)
    results["issues_found"] = random.sample(possible_issues, min(num_issues, len(possible_issues)))
    
    # Add warnings for non-critical items
    if random.random() > 0.5:
        results["warnings"].append("Some elements lack detailed descriptions")
    if random.random() > 0.7:
        results["warnings"].append("Consider adding more inter-element connections")
    
    # Add suggestions
    results["suggestions"] = [
        "Run deep analysis on flagged elements",
        "Review timeline for all temporal elements",
        "Consider creating a relationship map"
    ][:random.randint(1, 3)]
    
    return results


# Resources
@mcp.resource("worldbuilder://templates/character")
def get_character_template() -> Dict[str, Any]:
    """
    Provides a character template schema for worldbuilding.
    """
    return {
        "template_version": "1.0",
        "template_type": "character",
        "required_fields": {
            "name": "string",
            "age": "number or string",
            "role": "string",
            "description": "string"
        },
        "optional_fields": {
            "background": "string",
            "personality_traits": "array of strings",
            "goals": "array of strings",
            "fears": "array of strings",
            "relationships": "object mapping names to relationship types",
            "abilities": "array of objects",
            "inventory": "array of strings",
            "notes": "string"
        },
        "example": {
            "name": "Elena Starweaver",
            "age": 28,
            "role": "Protagonist - Mage Scholar",
            "description": "A brilliant researcher of ancient magic with a mysterious past",
            "personality_traits": ["curious", "determined", "secretive"],
            "goals": ["Uncover the truth about the Lost Archives", "Master temporal magic"],
            "relationships": {
                "Marcus": "mentor",
                "Zara": "rival",
                "The Council": "distrusted by"
            }
        }
    }


@mcp.resource("worldbuilder://templates/location")
def get_location_template() -> Dict[str, Any]:
    """
    Provides a location template schema for worldbuilding.
    """
    return {
        "template_version": "1.0",
        "template_type": "location",
        "required_fields": {
            "name": "string",
            "type": "string (city, wilderness, structure, etc.)",
            "description": "string"
        },
        "optional_fields": {
            "geography": "string",
            "climate": "string",
            "population": "number or string",
            "notable_features": "array of strings",
            "history": "string",
            "current_events": "array of strings",
            "connected_locations": "array of objects with name and distance",
            "resources": "array of strings",
            "dangers": "array of strings",
            "notes": "string"
        },
        "example": {
            "name": "The Whispering Library",
            "type": "structure",
            "description": "An ancient library that exists partially in another dimension",
            "notable_features": [
                "Books that write themselves",
                "Shifting corridors",
                "The Forbidden Section"
            ],
            "connected_locations": [
                {"name": "Arcanum City", "distance": "2 days travel"},
                {"name": "The Void Gate", "distance": "through the Forbidden Section"}
            ],
            "dangers": ["Knowledge parasites", "Temporal loops", "The Librarian"]
        }
    }


@mcp.resource("worldbuilder://stats")
async def get_server_stats() -> Dict[str, Any]:
    """
    Provides server statistics and metadata.
    """
    return {
        "server_name": "Worldbuilder Demo MCP Server",
        "version": "1.0.0",
        "uptime_seconds": random.randint(1000, 100000),
        "capabilities": {
            "tools": ["analyze_world_element", "calculate_narrative_distance", "generate_story_seed", "validate_world_consistency"],
            "resources": ["templates/character", "templates/location", "stats"],
            "prompts": ["create_character_profile", "describe_location", "plot_development"]
        },
        "statistics": {
            "total_requests": random.randint(100, 10000),
            "tools_called": random.randint(50, 5000),
            "resources_accessed": random.randint(30, 3000),
            "prompts_generated": random.randint(20, 2000)
        },
        "demo_mode": True,
        "last_reset": datetime.now().isoformat()
    }


# Prompts
@mcp.prompt
def create_character_profile(
    character_name: str,
    role: str,
    world_context: Optional[str] = None,
    detail_level: Literal["basic", "detailed", "comprehensive"] = "detailed"
) -> str:
    """
    Generates a prompt for creating a character profile.
    """
    base_prompt = f"""Create a character profile for {character_name}, who serves as {role} in the story.

Please include:
1. Physical description and age
2. Personality traits and quirks  
3. Background and history
4. Motivations and goals
5. Fears and weaknesses
6. Key relationships
7. Special abilities or skills (if any)"""

    if world_context:
        base_prompt += f"\n\nWorld Context: {world_context}"
    
    detail_additions = {
        "basic": "\n\nProvide a concise overview focusing on the most essential elements.",
        "detailed": "\n\nProvide a thorough profile with specific examples and clear character voice.",
        "comprehensive": """

Additionally include:
8. Character arc potential
9. Internal conflicts
10. External challenges
11. Symbolic significance
12. Potential subplot seeds
13. Voice and speech patterns
14. Moral alignment and philosophy

Provide extensive detail with examples, potential scenes, and character development opportunities."""
    }
    
    return base_prompt + detail_additions[detail_level]


@mcp.prompt
def describe_location(
    location_name: str,
    location_type: str,
    atmosphere: Optional[str] = None,
    include_history: bool = True
) -> str:
    """
    Generates a prompt for describing a location in detail.
    """
    prompt = f"""Describe {location_name}, a {location_type}, in rich sensory detail.

Focus on:
1. Visual appearance and layout
2. Sounds, smells, and atmosphere
3. Unique or notable features
4. How characters would interact with this space
5. Time of day and weather effects"""

    if atmosphere:
        prompt += f"\n\nDesired atmosphere: {atmosphere}"
    
    if include_history:
        prompt += """

Also include:
6. Brief history of the location
7. Current state and recent changes
8. Local legends or rumors
9. Hidden areas or secrets"""
    
    prompt += "\n\nWrite in an immersive, literary style that makes readers feel present in the location."
    
    return prompt


@mcp.prompt
def plot_development(
    current_situation: str,
    protagonist_goal: str,
    antagonist_force: Optional[str] = None,
    desired_themes: Optional[List[str]] = None,
    plot_style: Literal["linear", "complex", "twist-heavy"] = "complex"
) -> str:
    """
    Generates a prompt for developing plot ideas.
    """
    prompt = f"""Given the current situation: {current_situation}

And the protagonist's goal: {protagonist_goal}

Suggest plot developments that:
1. Create meaningful obstacles and complications
2. Develop character through challenge
3. Build tension naturally
4. Offer moments of hope and despair
5. Lead toward a satisfying resolution"""

    if antagonist_force:
        prompt += f"\n\nThe main opposing force is: {antagonist_force}"
        prompt += "\nConsider how this force would realistically counter the protagonist's efforts."
    
    if desired_themes:
        prompt += f"\n\nIncorporate these themes: {', '.join(desired_themes)}"
    
    style_additions = {
        "linear": "\n\nFocus on clear cause-and-effect progression with steady escalation.",
        "complex": "\n\nInclude subplots, parallel storylines, and interconnected consequences.",
        "twist-heavy": "\n\nIncorporate surprising revelations, misdirection, and paradigm shifts while maintaining fairness to the reader."
    }
    
    prompt += style_additions[plot_style]
    prompt += "\n\nProvide 3-5 specific plot development options with brief explanations of their narrative impact."
    
    return prompt


# Entry point
if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if "--test" in sys.argv:
        print("🚀 Worldbuilder Demo MCP Server")
        print("Server configured successfully!")
        print("\nCapabilities:")
        print("- Tools: analyze_world_element, calculate_narrative_distance, generate_story_seed, validate_world_consistency")
        print("- Resources: worldbuilder://templates/character, worldbuilder://templates/location, worldbuilder://stats")
        print("- Prompts: create_character_profile, describe_location, plot_development")
        print("\nUse without --test flag to run the server.")
    else:
        mcp.run()