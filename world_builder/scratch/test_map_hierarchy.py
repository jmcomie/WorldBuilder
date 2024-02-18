from world_builder.project import MapHierarchy
from world_builder.graph_registry import WorldBuilderNodeType, MapRootData, MapRect, MapRectMetadata


root_data: MapRootData = MapRootData(name="test", draw_diameter=3, width=81, height=81)
map_hierarchy: MapHierarchy = MapHierarchy(root_data)
