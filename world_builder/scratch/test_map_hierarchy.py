from world_builder.graph_registry import MapRootData, MapRect
from world_builder.map import MapHierarchy, SparseMapTree


root_data: MapRootData = MapRootData(name="test", draw_diameter=3, width=81, height=81)
print(root_data.dict())
#_map_rect=MapRect(x=0,y=0, width=9**3, height=9**3)

map_hierarchy: MapHierarchy = MapHierarchy(root_data)


