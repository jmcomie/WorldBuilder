import asyncio
from enum import Enum, StrEnum
from pathlib import Path
import sys
from typing import Callable, get_args
import code
import readline
import rlcompleter


# START Monkey patch.
import collections
import collections.abc
from gstk.llmlib.object_generation import get_chat_completion_object_response

from gstk.models.chatgpt import Message, Role
collections.Mapping = collections.abc.Mapping
# END Monkey patch.

from gstk.graph.graph import Node, get_project, new_project

#from gstk.creation.graph_registry import Message, Role

#from gstk.creation.group import get_chat_completion_object_response

from gstk.graph.registry import GraphRegistry, NodeTypeData, ProjectProperties
from world_builder.graph_registry import DrawDiameterInt, MapRootData, WorldBuilderNodeType, MapRect, MapMatrixData


from world_builder.project import MapRoot, WorldBuilderProject, WorldBuilderProjectDirectory
from world_builder.map import MAP_METADATA_DIRNAME, MapRootData, MapRoot, SparseMapTree

from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import print_function


class ExitDirectiveException(Exception): pass
class BackDirectiveException(Exception): pass


class PromptOptions(StrEnum):
    USER_OPTION = "user_option"


class ProjectOptions(StrEnum):
    NEW_PROJECT = "New project"
    OPEN_PROJECT = "Open project"


class MapRootOptions(StrEnum):
    NEW_MAP_ROOT = "New map"
    OPEN_MAP_ROOT = "Open map"

class BackNavigations(StrEnum):
    BACK = "Back"
    EXIT = "Exit"

class Views(Enum):
    LANDING = 0
    PROJECT = 1
    MAP = 2
    CELL = 3
    MATRIX = 4


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




def get_landing_ui_questions_list(choices: list[str]) -> list[dict]:
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
            'message': 'Select a project:',
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
    if int(text) not in get_args(DrawDiameterInt):
        raise ValidationError(message=f"Invalid dimensions. Dimensions must be one of {get_args(DrawDiameterInt)}.",
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
            'name': 'description',
            'message': 'Description:',
            'validate': lambda text: len(text) > 20 and len(text) < 500
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

def get_asset_selection_questions_list(map_root: MapRoot) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select asset:',
            'choices': map_root.list_asset_map_templates()
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


def select_project(project_locator: WorldBuilderProjectDirectory) -> WorldBuilderProject:
    # Project selection

    answer: dict = prompt(get_landing_ui_questions_list([m.value for m in ProjectOptions if m != ProjectOptions.OPEN_PROJECT or bool(project_locator.list_project_ids())]))
    # Project selection
    project_node: Node
    if answer[PromptOptions.USER_OPTION] == ProjectOptions.NEW_PROJECT:
        answer_new: dict = prompt(get_new_project_questions_list(get_validate_new_project_id_fn(project_locator)))
        project_properties: ProjectProperties = ProjectProperties(
            id=answer_new['project_id'],
            name=answer_new['project_name'],
            description=answer_new['project_description']
        )
        project_node = new_project(project_properties, project_locator)
        (project_locator.get_project_resource_location(project_properties.id) / MAP_METADATA_DIRNAME).mkdir(parents=True, exist_ok=True)
    elif answer[PromptOptions.USER_OPTION] == ProjectOptions.OPEN_PROJECT:
        answer_open: dict = prompt(get_open_project_questions_list(project_locator.list_project_ids()))
        project_id: str = answer_open[PromptOptions.USER_OPTION]
        project_node = get_project(project_id, project_locator)
    else:
        raise ValueError(f"Invalid mode option: {answer[PromptOptions.USER_OPTION]}")

    print(f"Project {project_node.data.id} opened.")
    return WorldBuilderProject(project_node, project_locator.get_project_resource_location(project_node.data.id))


def select_map_root(project: WorldBuilderProject) -> MapRoot:
    # Map root selection
    map_root_dict: dict[str, MapRoot] = project.get_map_root_dict()
    answer: dict = prompt(get_map_root_questions_list([m.value for m in MapRootOptions if m != MapRootOptions.OPEN_MAP_ROOT or bool(map_root_dict)]))
    map_root: MapRoot
    if answer[PromptOptions.USER_OPTION] == MapRootOptions.NEW_MAP_ROOT:
        answer_new = prompt(get_new_map_name_and_diameter_questions_list(map_root_dict.keys()))
        answer_new.update(prompt(get_new_map_width_and_height_questions_list(answer_new['draw_diameter'])))
        map_root = project.new_map(
                MapRootData(name=answer_new['name'],
                            draw_diameter=answer_new['draw_diameter'],
                            description=answer_new["description"],
                            width=answer_new['width'],
                            height=answer_new['height']))
    elif answer[PromptOptions.USER_OPTION] == MapRootOptions.OPEN_MAP_ROOT:
        answer_open: dict = prompt(get_open_map_questions_list(map_root_dict.keys()))
        map_root_name: str = answer_open[PromptOptions.USER_OPTION]
        map_root: MapRoot = map_root_dict[map_root_name]
    else:
        raise ValueError(f"Invalid map root option: {answer[PromptOptions.USER_OPTION]}")

    print(f"Map {map_root.data.name} opened.")

    while not map_root.has_asset():
        answer: dict = prompt(get_map_asset_questions_list(project.resource_location))
        if answer[PromptOptions.USER_OPTION] == "Select asset":
            answer_select_asset: dict = prompt(get_asset_selection_questions_list(map_root))
            map_root.add_asset_map_from_template(answer_select_asset[PromptOptions.USER_OPTION])
        elif answer[PromptOptions.USER_OPTION] == "Check asset" and not map_root.has_asset():
            print("No asset found.")

    return map_root


async def run():
    project_locator = WorldBuilderProjectDirectory()

    # If select_project and select_map_root are async, await them. Otherwise, make sure they are synchronous calls.
    project: WorldBuilderProject = select_project(project_locator)
    map_root: MapRoot = select_map_root(project)

    assert map_root.has_asset()

def interact_with_map(map_root: MapRoot, cell: str) -> tuple[Views, str]:
    pass

def view_matrix(map_root: MapRoot, cell: str):
    pass


SUPPORTED_VIEW_CHAIN: list[Views] = [Views.LANDING, Views.PROJECT, Views.MAP, Views.CELL, Views.MATRIX]

class CLIController():
    def __init__(self, project_locator: WorldBuilderProjectDirectory):
        self._project_locator = project_locator
        self._view_chain: list[Views] = [Views.LANDING]
        self._project: WorldBuilderProject = None
        self._map_root: MapRoot = None
        self._cell: str = None

    def run(self):
        should_exit: bool = False
        while not should_exit:
            try:
                self._interact()
            except BackDirectiveException:
                self.pop_view()
            except ExitDirectiveException:
                should_exit = True

    def append_view(self, view: Views):
        # Check if the view is a valid next view
        if list(self._view_chain) + [view] != SUPPORTED_VIEW_CHAIN[:len(self._view_chain) + 1]:
            raise ValueError(f"Invalid view chain: {list(self._view_chain) + [view]}")
        self._view_chain.append(view)

    def pop_view(self):
        if self._view_chain[-1] == Views.LANDING:
            raise ValueError("Cannot pop landing view.")
        elif self._view_chain[-1] == Views.PROJECT:
            self._project = None
        elif self._view_chain[-1] == Views.MAP:
            self._map_root = None
            self._cell = None
        elif self._view_chain[-1] == Views.CELL:
            self._cell = None
        self._view_chain.pop()

    def _interact(self):
        if self._view_chain[-1] == Views.LANDING:
            self._project = select_project()
            self.append_view(Views.PROJECT)
        elif self._view_chain[-1] == Views.PROJECT:
            # Combine MAP and CELL views for now. In the future, the map view
            # may provide options such as to copy/delete the map, etc, distinct
            # from the cell view.
            self._map_root = select_map_root(self._project)
            self._cell = "0:0" # root cell
            self.append_view(Views.MAP)
            self.append_view(Views.CELL)
        elif self._view_chain[-1] in [Views.MAP, Views.CELL]:
            view, cell = interact_with_map(self._map_root, self._cell)
            if view == Views.MATRIX:
                self.append_view(Views.MATRIX)
            if cell is not None:
                self._cell = cell
        elif self._view_chain[-1] == Views.MATRIX:
            view_matrix(self._map_root, self._cell)

"""
Open
List
List with data
Display
Up
"""

async def start():
    await run()
    #await run_in_code()

def main():
    asyncio.run(start())

if __name__ == "__main__":
    main()