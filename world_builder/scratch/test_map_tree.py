from world_builder.project import WorldBuilderProject, WorldBuilderProjectDirectory
from world_builder.map import MapRoot, SparseMapTree, MapMatrixData
from gstk.graph.graph import get_project



project_locator = WorldBuilderProjectDirectory()

# If select_project and select_map_root are async, await them. Otherwise, make sure they are synchronous calls.
project_id: str = "testproject"
project: WorldBuilderProject = WorldBuilderProject(
        get_project(
            project_id, project_locator),
            project_locator.get_project_resource_location(project_id))

map_root: MapRoot = project.get_map_root("testing")
tree: SparseMapTree = map_root.tree

for node in tree.list_children():
    print(node.data)
    print(node.data.map_rect)
    print(node.data.map_rect.to_tuple())

for data in tree.list_data_for_processing(commit_changes=True, skip_non_empty=True):
    print(tree._map_hierarchy.get_rect_level(data.map_rect))
    if isinstance(data, MapMatrixData):
        print("CHANGING")
        print(f"BEFORE CHANGING {data.tiles}")
        print(f"TYPE {type(data.tiles[0])}")
        data.tiles[0] = [3,3,3]
        print(f"AFTER CHANGING {data.tiles}")
    print(data.map_rect)
