from enum import StrEnum
from pathlib import Path
import sys
from typing import Callable, get_args

from gstk.creation.api import CreationProject, get_creation_project, ProjectProperties, new_creation_project

# START Monkey patch.
import collections
import collections.abc
from world_builder.graph_registry import DrawDimensionInt, MapRoot
collections.Mapping = collections.abc.Mapping
# END Monkey patch.


from world_builder.project import MAP_METADATA_DIRNAME, MapTileGroup, WorldBuilderProject, WorldBuilderProjectDirectory


from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import print_function


class PromptOptions(StrEnum):
    USER_OPTION = "user_option"



class ProjectOptions(StrEnum):
    NEW_PROJECT = "New project"
    OPEN_PROJECT = "Open project"


class MapRootOptions(StrEnum):
    NEW_MAP_ROOT = "New map"
    OPEN_MAP_ROOT = "Open map"


def get_validate_new_project_id_fn(project_locator: WorldBuilderProjectDirectory):
    def validate_new_project_id(text: str):
        keep_characters: set[str] = {' ','.','_'}
        validated_text = "".join(c for c in text if c.isalnum() or c in keep_characters).rstrip()
        if text != validated_text:
            raise ValidationError(message = f"Name must include only alphanumeric characters, spaces, underscores, and periods.  You entered: {text}\n" + \
                                    f"Consider using this validated form: {validated_text}\nName of the project:",
                                    cursor_position=len(text))
        if project_locator.project_id_exists(validated_text):
                raise ValidationError(message="Project already exists.\nName of the project:",
                                    cursor_position=len(text))
        return True
    return validate_new_project_id

def get_validate_new_map_root_id_fn(map_root_names: set[str]):
    def validate_new_map_root_id(text: str):
        if text in map_root_names:
                raise ValidationError(message=f"Map name '{text}' already exists.",
                                      cursor_position=len(text))
        return True
    return validate_new_map_root_id


def get_open_project_questions_list(choices: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Welcome to WorldBuilder',
            'choices': choices
        },
    ]

def get_open_project_questions_list(available_project_ids: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select a project to open:',
            'choices': available_project_ids
        }
    ]

def get_new_project_questions_list(project_id_validator: Callable) -> list[dict]:
    return [
        {
            'type': 'input',
            'validate': project_id_validator,
            'name': 'project_id',
            'message': 'Project id:',
        },
        {
            'type': 'input',
            'name': 'project_name',
            'message': 'Project name (optional):',
        },
        {
            'type': 'input',
            'name': 'project_description',
            'message': 'Project description (optional):',
        },
    ]

def get_map_root_questions_list(choices: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Create or open a map:',
            'choices': choices
        }
    ]


new_questions = [

]


open_questions = [

]


def print_project_id_list(project_locator: WorldBuilderProjectDirectory):
    print("Projects:")
    project_ids: list[str] = project_locator.list_project_ids()
    if not project_ids:
        print("No projects found.")
    for project_id in project_ids:
        print(project_id)


def print_map_list(project: WorldBuilderProject):
    print("Maps:")
    print("No maps found.")

def get_open_map_questions_list(available_map_roots: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select a map to open:',
            'choices': available_map_roots
        }
    ]

def validate_diameter(text: str):
    if int(text) not in get_args(DrawDimensionInt):
        raise ValidationError(message=f"Invalid dimensions. Dimensions must be one of {get_args(DrawDimensionInt)}.",
                              cursor_position=len(text))
    return True

def get_new_map_name_and_diameter_questions_list(existing_map_names: list[str]) -> list[dict]:
    return [
        {
            'type': 'input',
            'name': 'name',
            'message': 'Map name:',
            'validate': get_validate_new_map_root_id_fn(existing_map_names)
        },
        {
            'type': 'input',
            'name': 'draw_diameter',
            'message': 'Draw diameter:',
            'filter': lambda val: int(val),
            'validate': validate_diameter
        },
    ]

def get_validate_width_and_height_fn(draw_diameter: int):
    def validate_width_and_height(text: str):
        if int(text) % draw_diameter != 0:
            raise ValidationError(message=f"Width and height must be a multiple of {draw_diameter}.",
                                  cursor_position=len(text))
        return True
    return validate_width_and_height



def get_map_asset_questions_list(project_resource_location: Path) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Asset selection.',
            'choices': ['Select asset', 'Check asset']
        }
    ]

def get_asset_selection_questions_list(map_tile_group: MapTileGroup) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select asset:',
            'choices': map_tile_group.list_asset_templates()
        }
    ]

def get_new_map_width_and_height_questions_list(draw_diameter: int) -> list[dict]:
    return [
        {
            'type': 'input',
            'name': 'width',
            'message': 'Map width:',
            'filter': lambda val: int(val),
            'validate': get_validate_width_and_height_fn(draw_diameter)
        },
        {
            'type': 'input',
            'name': 'height',
            'message': 'Map height:',
            'filter': lambda val: int(val),
            'validate': get_validate_width_and_height_fn(draw_diameter)
        },
    ]

def main():
    project_locator = WorldBuilderProjectDirectory()

    # Project selection
    answer: dict = prompt(get_open_project_questions_list([m.value for m in ProjectOptions if m != ProjectOptions.OPEN_PROJECT or bool(project_locator.list_project_ids())]))
    print(f"project ids {project_locator.list_project_ids()}")
    # Project selection
    project: WorldBuilderProject
    if answer[PromptOptions.USER_OPTION] == ProjectOptions.NEW_PROJECT:
        answer_new: dict = prompt(get_new_project_questions_list(get_validate_new_project_id_fn(project_locator)))
        project_properties: ProjectProperties = ProjectProperties(
            id=answer_new['project_id'],
            name=answer_new['project_name'],
            description=answer_new['project_description']
        )
        project = new_creation_project(project_properties.id, project_properties, resource_locator=project_locator, project_class=WorldBuilderProject)
        (project.resource_location / MAP_METADATA_DIRNAME).mkdir(parents=True, exist_ok=True)
    elif answer[PromptOptions.USER_OPTION] == ProjectOptions.OPEN_PROJECT:
        answer_open: dict = prompt(get_open_project_questions_list(project_locator.list_project_ids()))
        project_id: str = answer_open[PromptOptions.USER_OPTION]
        project = get_creation_project(project_id, resource_locator=project_locator, project_class=WorldBuilderProject)
    else:
        raise ValueError(f"Invalid mode option: {answer[PromptOptions.USER_OPTION]}")

    print(f"Project {project.project_node.id} opened.")

    # Map root selection
    map_root_dict: dict[str, MapTileGroup] = project.get_map_root_dict()
    answer: dict = prompt(get_map_root_questions_list([m.value for m in MapRootOptions if m != MapRootOptions.OPEN_MAP_ROOT or bool(map_root_dict)]))
    map_root: MapTileGroup
    if answer[PromptOptions.USER_OPTION] == MapRootOptions.NEW_MAP_ROOT:
        answer_new = prompt(get_new_map_name_and_diameter_questions_list(map_root_dict.keys()))
        answer_new.update(prompt(get_new_map_width_and_height_questions_list(answer_new['draw_diameter'])))
        map_root = project.new_map(MapRoot(**answer_new))
    elif answer[PromptOptions.USER_OPTION] == MapRootOptions.OPEN_MAP_ROOT:
        answer_open: dict = prompt(get_open_map_questions_list(map_root_dict.keys()))
        map_root_name: str = answer_open[PromptOptions.USER_OPTION]
        map_root: MapTileGroup = map_root_dict[map_root_name]
    else:
        raise ValueError(f"Invalid map root option: {answer[PromptOptions.USER_OPTION]}")

    print(f"Map {map_root.data.name} opened.")

    while not map_root.has_asset():
        answer: dict = prompt(get_map_asset_questions_list(project.resource_location))
        if answer[PromptOptions.USER_OPTION] == "Select asset":
            answer_select_asset: dict = prompt(get_asset_selection_questions_list(map_root))
            map_root.add_asset_from_template(answer_select_asset[PromptOptions.USER_OPTION])
        elif answer[PromptOptions.USER_OPTION] == "Check asset" and not map_root.has_asset():
            print("No asset found.")

    print("Map has asset.")

if __name__ == "__main__":
    main()