from world_builder.graph_registry import MapRootData
from world_builder.map import MapHierarchy


root_data: MapRootData = MapRootData(name="test", draw_diameter=3, width=9**3, height=9**3)
map_hierarchy: MapHierarchy = MapHierarchy(root_data)

