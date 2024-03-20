



from itertools import product
from gstk.graph.graph import get_project
import numpy as np
from pydantic import BaseModel
from world_builder.graph_registry import DescriptionMatrixData, MapMatrixData
from world_builder.map import MapHierarchy, MapRoot
from world_builder.project import WorldBuilderProject, WorldBuilderProjectDirectory


def get_matrix(map_root: MapRoot, cell_identifier: str):
    sparse_node = map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier)
    if not sparse_node.has_data():
        matrix_str = "No data."
    else:
        assert isinstance(sparse_node.data, BaseModel)
        data = sparse_node.data.model_copy()
        if isinstance(data, DescriptionMatrixData):
            hierarchy: MapHierarchy = map_root.tree.hierarchy
            rect_child_matrix: np.array = hierarchy.get_rect_child_matrix(sparse_node.rect)
            arr = np.array(data.tiles)
            assert rect_child_matrix.shape == arr.shape, f"{rect_child_matrix.shape} {arr.shape}, arr: {arr}"
            #for y,x in product(range(arr.shape[0]), range(arr.shape[1])):
            #    cell_identifier = hierarchy.get_map_rect_cell_identifier(rect_child_matrix[y,x])
            #    arr[y,x] = f"{cell_identifier}: {str(arr[y,x])}"
            return arr
        elif isinstance(data, MapMatrixData):
            return np.array(data.tiles)


def main():
    project_id: str = "Scratch 2"
    map_id: str = "Island Map"

    project_locator: WorldBuilderProjectDirectory = WorldBuilderProjectDirectory()
    project: WorldBuilderProject = WorldBuilderProject(get_project(project_id, project_locator), project_locator.get_project_resource_location(project_id))
    map_root: MapRoot = project.get_map_root(map_id)
    print(get_matrix(map_root, "0:0"))


if __name__ == "__main__":
    main()