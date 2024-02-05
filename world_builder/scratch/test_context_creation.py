from io import StringIO
from typing import Any

from pytmx import TiledMap

from gstk.creation.group import get_chat_completion_object_response

from world_builder.map_data_interface import get_gid_tile_properties, get_tile_matrix_from_csv, get_gid_data, GIDData, image_from_csv_tile_matrix
from world_builder.project import WorldBuilderProjectDirectory, WorldBuilderProject, get_project, MapTileGroup


def main():
    project: WorldBuilderProject = get_project("testproject")
    map: MapTileGroup = project.get_map_root_dict()["testmap"]
    gid_tile_properties: dict[int, dict[str, Any]] = get_gid_tile_properties(map.get_tiled_map())
    gid_description: str = ""
    for gid, value in gid_tile_properties.items():
        if not value or "description" not in value:
            print(f"gid: {gid} has no description.")
            continue
        gid_description += f"gid: {gid} description: {value['description']}\n"
    print(f"""You are creating a 2D tilemap for a 2D game represented as integers in a CSV.
Create a ten by ten CSV file's contents in which each cell is one of the
gid integers below, and that adheres to the following description: Create a map of the beach where the shore is aligned vertically near the center of the map and in which grass is on the left side of the map and in which water is on the right side of the map, with an appropriate gradation.
GID Descriptions:

{gid_description}""")
    return map

MATRIX: str = """1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
1,2,6,4,7,7,8,9,10,11
"""

if __name__ == "__main__":
    project: WorldBuilderProject = get_project("testproject")
    map_root: MapTileGroup = project.get_map_root_dict()["testmap"]
    gid_data: dict[int, GIDData] = get_gid_data(map_root.get_tiled_map())
    image_from_csv_tile_matrix(StringIO(MATRIX), map_root.get_tiled_map()).show()