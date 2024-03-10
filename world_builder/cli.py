import asyncio
from itertools import product
import devtools
from enum import Enum, StrEnum
from pathlib import Path
import pprint
import pydoc
import sys
import tempfile
from typing import Any, Callable, Optional, Type, get_args
import code
import readline
import rlcompleter


# START Monkey patch.
import collections

import numpy as np
from pydantic import BaseModel
collections.Mapping = collections.abc.Mapping
# END Monkey patch.

import collections.abc
from gstk.llmlib.object_generation import get_chat_completion_object_response
from gstk.models.chatgpt import Message, Role
from gstk.graph.graph import Node, get_project, new_project

#from gstk.creation.graph_registry import Message, Role

#from gstk.creation.group import get_chat_completion_object_response

from gstk.graph.registry import GraphRegistry, NodeTypeData, ProjectProperties
from world_builder.graph_registry import DrawDiameterInt, MapRootData, WorldBuilderNodeType, MapRect, MapMatrixData, DescriptionMatrixData
from world_builder.table import Table
from world_builder.context import get_description_matrix_context_messages
from world_builder.project import MapRoot, WorldBuilderProject, WorldBuilderProjectDirectory
from world_builder.map import MAP_METADATA_DIRNAME, MapHierarchy, MapRootData, MapRoot, SparseMapNode, SparseMapTree, CELL_IDENTIFER_RE, ROOT_CELL_IDENTIFIER, get_cell_prompt, set_cell_prompt
from world_builder.map_data_interface import get_gid_description_context_string, get_image_from_tile_matrix
from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import print_function


class ExitDirectiveException(Exception): pass
class BackDirectiveException(Exception): pass
BACK_DIRECTIVE_INVOCATION_STRING: str = "back"


class PromptOptions(StrEnum):
    USER_OPTION = "user_option"


class ProjectOptions(StrEnum):
    NEW_PROJECT = "New project"
    OPEN_PROJECT = "Open project"


class ProjectViewOptions(StrEnum):
    NEW_MAP_ROOT = "New map"
    OPEN_MAP_ROOT = "Open map"
    RENAME_MAP = "Rename Map"
    CLONE_MAP = "Clone Map"


class CellOptions(StrEnum):
    VIEW_PROMPT = "View Cell Prompt"
    EDIT_PROMPT = "Edit Cell Prompt"
    VIEW_CHILD_MATRIX = "View Child Matrix"
    GENERATE_CHILD_MATRIX = "Generate Child Matrix"
    GENERATE_RECURSIVE_MATRICES = "Generate Recursive Matrices"
    GOTO_CELL = "Goto Cell"
    VIEW_CONTEXT_CHAIN = "View Context Chain"
    VIEW_PROJECTED_IMAGE = "View Projected Image"
    VIEW_LEVEL_COUNTS = "View Level Counts"
    DELETE_SUBTREE = "Delete Subtree"

class MapOptions(StrEnum):
    GOTO_ROOT_CELL = "Goto Root Cell"
    GOTO_CELL = "Goto Cell"
    LIST_TILE_ASSETS = "List asset tiles"
    VIEW_IMAGE = "View Image"
    READONLY = "Set Readonly"


class BackNavigations(StrEnum):
    BACK = "Back"
    EXIT = "Exit"


class Views(Enum):
    LANDING = 0
    PROJECT = 1
    MAP = 2
    CELL = 3
    MATRIX = 4
    CONTEXT_CHAIN = 5


def print_burgandy(text: str):
    print(f"\033[38;5;88m{text}\033[0m")


def get_validate_new_project_id_fn(project_locator: WorldBuilderProjectDirectory):
    def validate_new_project_id(text: str):
        if text == BACK_DIRECTIVE_INVOCATION_STRING:
            raise BackDirectiveException()
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
        if text == BACK_DIRECTIVE_INVOCATION_STRING:
            raise BackDirectiveException()
        if text in map_root_names:
                raise ValidationError(message=f"Map name '{text}' already exists.",
                                      cursor_position=len(text))
        return True
    return validate_new_map_root_id


def get_validate_cell_identifier_fn(map_root: MapRoot):
    def validate_cell_identifier(text: str):
        if text == BACK_DIRECTIVE_INVOCATION_STRING:
            raise BackDirectiveException()
        try:
            # Should throw exception if cell identifier does not exist.
            map_root.tree.get_sparse_node_from_cell_identifier(text)
        except Exception as e:
            raise ValidationError(message=f"Invalid cell identifier: {text}",
                                  cursor_position=len(text))
        return True
    return validate_cell_identifier


def get_landing_ui_questions_list(choices: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Welcome to WorldBuilder',
            'choices': choices + [BackNavigations.EXIT]
        },
    ]


def get_open_project_questions_list(available_project_ids: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select a project:',
            'choices': available_project_ids + [BackNavigations.BACK]
        }
    ]


def view_projected_image(map_root: MapRoot, cell_identifier: str):
    map_rect: MapRect = map_root.tree.hierarchy.get_cell_identifier_map_rect(cell_identifier)
    tile_array: np.array = np.zeros((map_rect.width, map_rect.height)).astype(int)

    for leaf_map_rect in map_root.tree.hierarchy.get_rect_matrix(map_root.tree.hierarchy.get_tree_height() - 1).flatten():
        if leaf_map_rect.x < map_rect.width + map_rect.x and leaf_map_rect.y < map_rect.height + map_rect.y and \
                leaf_map_rect.x >= map_rect.x and leaf_map_rect.y >= map_rect.y:
            sparse_node: SparseMapNode = map_root.tree.get_sparse_node_from_cell_identifier(
                                map_root.tree.hierarchy.get_map_rect_cell_identifier(leaf_map_rect))
            if sparse_node.has_data() and np.array(sparse_node.data.tiles).shape == (map_root.data.draw_diameter, map_root.data.draw_diameter):
                tile_array[leaf_map_rect.y:leaf_map_rect.y+map_root.data.draw_diameter, leaf_map_rect.x:leaf_map_rect.x+map_root.data.draw_diameter] = np.array(sparse_node.data.tiles)
    get_image_from_tile_matrix(tile_array, map_root.get_tiled_map()).show()


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
            'choices': choices + [BackNavigations.BACK]
        }
    ]


def get_open_map_questions_list(available_map_roots: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select a map to open:',
            'choices': available_map_roots + [BackNavigations.BACK]
        }
    ]

def get_map_options_questions_list(choices: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': 'user_option',
            'message': 'Select option:',
            'choices': choices + [BackNavigations.BACK]
        }
    ]


def validate_diameter(text: str):
    if int(text) not in get_args(DrawDiameterInt):
        raise ValidationError(message=f"Invalid dimensions. Dimensions must be one of {get_args(DrawDiameterInt)}.",
                              cursor_position=len(text))
    return True


def get_new_map_name_and_diameter_questions_list(existing_map_names: list[str], map_root: Optional[MapRoot] = None) -> list[dict]:
    return [
        {
            'type': 'input',
            'name': 'name',
            'message': 'Map name:',
            'default': map_root is not None and map_root.data.name or "",
            'validate': get_validate_new_map_root_id_fn(existing_map_names)
        },
        {
            'type': 'input',
            'name': 'description',
            'message': 'Description:',
            'when': lambda x: map_root is None,
            'validate': lambda text: len(text) > 20 and len(text) < 500
        },
        {
            'type': 'input',
            'name': 'draw_diameter',
            'message': 'Draw diameter:',
            'when': lambda x: map_root is None,
            'validate': validate_diameter
        },
    ]


def get_validate_width_and_height_fn(draw_diameter: int):
    def validate_width_and_height(text: str):
        if text == BACK_DIRECTIVE_INVOCATION_STRING:
            raise BackDirectiveException()
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


def get_cell_interface_questions_list(message: str, choices: list[str]) -> list[dict]:
    return [
        {
            'type': 'list',
            'name': PromptOptions.USER_OPTION,
            'message': message,
            'choices': choices + [BackNavigations.BACK]
        }
    ]


def get_cell_identifier_questions_list(map_root: MapRoot) -> list[dict]:
    return [
        {
            'type': 'input',
            'name': PromptOptions.USER_OPTION,
            'message': 'Cell identifier:',
            'validate': get_validate_cell_identifier_fn(map_root)
        }
    ]


def select_project(project_locator: WorldBuilderProjectDirectory) -> WorldBuilderProject:
    # Project selection

    answer: dict = prompt(get_landing_ui_questions_list([m.value for m in ProjectOptions if m != ProjectOptions.OPEN_PROJECT or bool(project_locator.list_project_ids())]))
    # Project selection
    project_node: Node
    if not answer or answer[PromptOptions.USER_OPTION] == BackNavigations.EXIT:
        raise ExitDirectiveException()
    elif answer[PromptOptions.USER_OPTION] == ProjectOptions.NEW_PROJECT:
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
        if not answer_open or answer_open[PromptOptions.USER_OPTION] == BackNavigations.BACK:
            raise BackDirectiveException()
        project_id: str = answer_open[PromptOptions.USER_OPTION]
        project_node = get_project(project_id, project_locator)
    else:
        raise ValueError(f"Invalid mode option: {answer[PromptOptions.USER_OPTION]}")

    print(f"Project {project_node.data.id} opened.")
    return WorldBuilderProject(project_node, project_locator.get_project_resource_location(project_node.data.id))


def select_map_root(project: WorldBuilderProject) -> MapRoot:
    # Map root selection
    map_root_dict: dict[str, MapRoot] = project.get_map_root_dict()
    answer: dict = prompt(get_map_root_questions_list([m.value for m in ProjectViewOptions if m != ProjectViewOptions.OPEN_MAP_ROOT or bool(map_root_dict)]))
    map_root: MapRoot
    if not answer or answer[PromptOptions.USER_OPTION] == BackNavigations.BACK:
        raise BackDirectiveException()
    elif answer[PromptOptions.USER_OPTION] == ProjectViewOptions.NEW_MAP_ROOT:
        answer_new = prompt(get_new_map_name_and_diameter_questions_list(map_root_dict.keys()))
        if not answer_new:
            raise BackDirectiveException()
        answer_new.update(prompt(get_new_map_width_and_height_questions_list(answer_new['draw_diameter'])))
        map_root = project.new_map_root(
                MapRootData(name=answer_new['name'],
                            draw_diameter=answer_new['draw_diameter'],
                            description=answer_new["description"],
                            width=answer_new['width'],
                            height=answer_new['height']))
    elif answer[PromptOptions.USER_OPTION] in [ProjectViewOptions.RENAME_MAP, ProjectViewOptions.CLONE_MAP]:
        answer_select: dict = prompt(get_open_map_questions_list(list(map_root_dict.keys())))
        if not answer_select or answer_select[PromptOptions.USER_OPTION] == BackNavigations.BACK:
            return
        answer_name: dict = prompt(get_new_map_name_and_diameter_questions_list(map_root_dict.keys(), map_root=map_root_dict[answer_select[PromptOptions.USER_OPTION]]))
        if answer[PromptOptions.USER_OPTION] == ProjectViewOptions.RENAME_MAP:
            project.rename_map(answer_select[PromptOptions.USER_OPTION], answer_name['name'])
        else:
            project.clone_map(answer_select[PromptOptions.USER_OPTION], answer_name['name'])
        return
    elif answer[PromptOptions.USER_OPTION] == ProjectViewOptions.OPEN_MAP_ROOT:
        answer_open: dict = prompt(get_open_map_questions_list(list(map_root_dict.keys())))
        if not answer_open or answer_open[PromptOptions.USER_OPTION] == BackNavigations.BACK:
            return
        map_root_name: str = answer_open[PromptOptions.USER_OPTION]
        map_root: MapRoot = map_root_dict[map_root_name]
    else:
        raise ValueError(f"Invalid map root option: {answer[PromptOptions.USER_OPTION]}")

    while not map_root.has_asset():
        answer: dict = prompt(get_map_asset_questions_list(project.resource_location))
        if answer[PromptOptions.USER_OPTION] == "Select asset":
            answer_select_asset: dict = prompt(get_asset_selection_questions_list(map_root))
            map_root.add_asset_map_from_template(answer_select_asset[PromptOptions.USER_OPTION])
        elif answer[PromptOptions.USER_OPTION] == "Check asset" and not map_root.has_asset():
            print("No asset found.")

    print(f"Map {map_root.data.name} opened.")

    return map_root


def view_prompt(map_root: MapRoot, cell_identifier: str):
    pydoc.pager(get_cell_prompt(map_root, cell_identifier))

def edited_prompt(prompt_str: str):
    questions = [
    {
        'type': 'editor',
        'name': 'contents',
        'message': 'Editor',
        'default': prompt_str,
        'eargs': {
            'ext':'.txt',
            'editor':'vim'
            #'filename': filepath
        }
    }
    ]
    answer = prompt(questions)
    return answer['contents']


def edit_prompt(map_root: MapRoot, cell_identifier: str):
    if map_root.data.readonly:
        print_burgandy("Map is readonly.  Cannot edit prompt.")
        return
    print("Editing prompt")
    #with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as temp_file:
    #    temp_file.write(get_cell_prompt(map_root, cell_identifier))
    #    temp_file.flush()
    prompt: str = edited_prompt(get_cell_prompt(map_root, cell_identifier))
    print(f"Prompt: {prompt}")
    set_cell_prompt(map_root, cell_identifier, prompt)

async def generate_child_matrix(map_root: MapRoot, cell_identifier: str):
    if map_root.data.readonly:
        print_burgandy("Map is readonly.  Cannot generate child matrix.")
        return
    print(f"Generating child matrix for cell {cell_identifier}.")
    map_rect: MapRect = map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier).rect
    messages: list[Message] = get_description_matrix_context_messages(map_root, map_rect)
    map_rect_data_type: Type[DescriptionMatrixData|MapMatrixData] = map_root.tree.get_map_rect_data_type(map_rect)
    if map_rect_data_type == MapMatrixData:
        res: list = await get_chat_completion_object_response(list(GraphRegistry.get_node_types(MapMatrixData))[0], messages)
        data: MapMatrixData = MapMatrixData(map_rect=map_rect, tiles=res.tiles)
    elif map_rect_data_type == DescriptionMatrixData:
        res: list = await get_chat_completion_object_response(list(GraphRegistry.get_node_types(DescriptionMatrixData))[0], messages)
        data: DescriptionMatrixData = DescriptionMatrixData(map_rect=map_rect, tiles=res.tiles)
    else:
        raise ValueError(f"Invalid map_rect_data_type: {map_rect_data_type}")
    map_root.tree.get_or_create_data_node(data)

def get_toggle_readonly_questions_list(default_value: bool) -> list[dict]:
    return [
        {
            'type': 'confirm',
            'message': 'Set readonly value. If true, matrix creation and editing is disabled.',
            'name': 'readonly',
            'default': default_value,
        },
    ]

def clone_map_questions_list() -> list[dict]:
    return [
        {
            'type': 'input',
            'name': 'name',
            'message': 'Map name:',
            'validate': lambda text: len(text) > 0
        },
    ]


async def generate_recursive_matrices(map_root: MapRoot, cell_identifier: str):
    if map_root.data.readonly:
        print_burgandy("Map is readonly.  Cannot generate recursive matrices.")
        return

    if not get_cell_prompt(map_root, cell_identifier):
        raise ValueError(f"Cell {cell_identifier} has no prompt.")

    for sparse_node in map_root.tree.walk_tree(root_map_rect=map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier).rect):
        if sparse_node.has_data():
            tiles: np.array = np.array(sparse_node.data.tiles)
            if np.all(np.array(sparse_node.data.tiles) == "") or np.all(np.array(sparse_node.data.tiles) == 0) \
                    or tiles.shape != (map_root.data.draw_diameter, map_root.data.draw_diameter):
                await generate_child_matrix(map_root, sparse_node.cell_identifier)
                #print(f"Want to generate child matrix for cell {sparse_node.cell_identifier}.")
        else:
            #print(f"Want to generate child matrix for cell {sparse_node.cell_identifier}.")
            await generate_child_matrix(map_root, sparse_node.cell_identifier)

def view_context_chain(map_root: MapRoot, cell_identifier: str):
    map_rect: MapRect = map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier).rect
    messages: list[Message] = get_description_matrix_context_messages(map_root, map_rect)
    pydoc.pager("\n".join([str(m) for m in messages]))
    #pydoc.pager(pprint.pformat(messages))
    #pydoc.pager(pprint.pformat([instance.model_dump() for instance in get_description_matrix_context_messages(map_root, map_rect)]))

def format_human_readable_cell_description(map_root: MapRoot, cell_identifier: str) -> str:
    cell_identifier_human_readable: str
    if cell_identifier == ROOT_CELL_IDENTIFIER:
        cell_identifier_human_readable = "Map Root Cell"
    else:
        level, cell = CELL_IDENTIFER_RE.match(cell_identifier).groups()
        cell_identifier_human_readable = f"Level {level} Cell {cell}"
    map_rect: MapRect = map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier).rect
    header = f"{cell_identifier_human_readable}\n{'-' * len(cell_identifier_human_readable)}"
    header += f"\nProjected Rect: x:{map_rect.x} y:{map_rect.y} w:{map_rect.width} h:{map_rect.height}"
    header += f"\nCell Identifier: {cell_identifier}"
    if map_root.tree.hierarchy.get_rect_level(map_rect) > 0:
        header += f"\nParent Cell Identifier: {map_root.tree.hierarchy.get_map_rect_cell_identifier(map_root.tree.hierarchy.get_parent_rect(map_rect))}"
    return header

def confirm_prompt(message: str) -> bool:
    answer: dict = prompt([
        {
            'type': 'confirm',
            'message': message,
            'name': 'confirm',
            'default': False,
        },
    ])
    return answer['confirm']


async def interact_with_cell(map_root: MapRoot, cell_identifier: str) -> tuple[Views, str]:
    header: str = format_human_readable_cell_description(map_root, cell_identifier)
    print_burgandy(header)
    answer: dict = prompt(get_cell_interface_questions_list("Select option", [m.value for m in CellOptions]))
    map_rect: MapRect = map_root.tree.get_sparse_node_from_cell_identifier(cell_identifier).rect
    if not answer or answer == BackNavigations.BACK:
        raise BackDirectiveException()
    elif answer[PromptOptions.USER_OPTION] == CellOptions.VIEW_PROMPT:
        view_prompt(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.EDIT_PROMPT:
        edit_prompt(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.VIEW_CHILD_MATRIX:
        view_matrix(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.GENERATE_CHILD_MATRIX:
        await generate_child_matrix(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.GENERATE_RECURSIVE_MATRICES:
        await generate_recursive_matrices(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.VIEW_CONTEXT_CHAIN:
        view_context_chain(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.GOTO_CELL:
        return prompt(get_cell_identifier_questions_list(map_root))[PromptOptions.USER_OPTION]
    elif answer[PromptOptions.USER_OPTION] == BackNavigations.BACK:
        raise BackDirectiveException()
    elif answer[PromptOptions.USER_OPTION] == CellOptions.VIEW_PROJECTED_IMAGE:
        view_projected_image(map_root, cell_identifier)
    elif answer[PromptOptions.USER_OPTION] == CellOptions.VIEW_LEVEL_COUNTS:
        pydoc.pager(pprint.pformat(map_root.tree.get_level_counts(map_rect)))
    elif answer[PromptOptions.USER_OPTION] == CellOptions.DELETE_SUBTREE:
        if map_root.data.readonly:
            print_burgandy("Map is readonly.  Cannot delete subtree.")
            return
        if confirm_prompt("Are you sure you want to delete the subtree?"):
            map_root.tree.delete_subtree(map_rect)
    else:
        raise ValueError(f"Invalid cell option: {answer[PromptOptions.USER_OPTION]}")


#class MapOptions(StrEnum):
#    GOTO_ROOT_CELL = "Goto Root Cell"
#    GOTO_CELL = "Goto Cell"
#    SELECT_TILE_ASSETS = "Select asset"
#    LIST_TILE_ASSETS = "List asset tiles"

def format_human_readable_map_description(map_root: MapRoot) -> str:
    header: str = f"Map: {map_root.data.name}\n{'-' * len('Map: ' + map_root.data.name)}"
    header += f"\nDraw Diameter: {map_root.data.draw_diameter}"
    header += f"\nWidth: {map_root.data.width} Height: {map_root.data.height}"
    header += f"\nReadonly: {map_root.data.readonly}"
    return header

def interact_with_map_root(map_root: MapRoot) -> str:
    header: str = format_human_readable_map_description(map_root)
    print_burgandy(header)

    answer: dict = prompt(get_map_options_questions_list([m.value for m in MapOptions]))
    if not answer or answer[PromptOptions.USER_OPTION] == BackNavigations.BACK:
        raise BackDirectiveException()
    elif answer[PromptOptions.USER_OPTION] == MapOptions.GOTO_ROOT_CELL:
        return ROOT_CELL_IDENTIFIER
    elif answer[PromptOptions.USER_OPTION] == MapOptions.GOTO_CELL:
        cell_answer: dict = prompt(get_cell_identifier_questions_list(map_root))
        if not cell_answer or cell_answer[PromptOptions.USER_OPTION] == BackNavigations.BACK:
            raise BackDirectiveException()
        return cell_answer[PromptOptions.USER_OPTION]
    elif answer[PromptOptions.USER_OPTION] == MapOptions.READONLY:
        map_root.set_readonly_state(
                prompt(
                        get_toggle_readonly_questions_list(map_root.data.readonly)
                )["readonly"]
        )
        return
    elif answer[PromptOptions.USER_OPTION] == MapOptions.LIST_TILE_ASSETS:
        pydoc.pager(get_gid_description_context_string(map_root.get_tiled_map()))
    elif answer[PromptOptions.USER_OPTION] == MapOptions.VIEW_IMAGE:
        view_projected_image(map_root, ROOT_CELL_IDENTIFIER)
    else:
        raise ValueError(f"Invalid map option: {answer[PromptOptions.USER_OPTION]}")

def view_matrix(map_root: MapRoot, cell: str):
    matrix_str: str
    sparse_node = map_root.tree.get_sparse_node_from_cell_identifier(cell)
    if not sparse_node.has_data():
        matrix_str = "No data."
    else:
        assert isinstance(sparse_node.data, BaseModel)
        data = sparse_node.data.model_copy()
        if isinstance(data, DescriptionMatrixData):
            hierarchy: MapHierarchy = map_root.tree.hierarchy
            rect_child_matrix: np.array = hierarchy.get_rect_child_matrix(sparse_node.rect)
            arr = np.array(data.tiles)
            assert rect_child_matrix.shape == arr.shape
            for y,x in product(range(arr.shape[0]), range(arr.shape[1])):
                cell_identifier = hierarchy.get_map_rect_cell_identifier(rect_child_matrix[y,x])
                arr[y,x] = f"{cell_identifier}: {str(arr[y,x])}"
            matrix_str = str(Table(arr.tolist(), 20, True))
        else:
            matrix_str = str(Table(np.array(sparse_node.data.tiles).astype(str).tolist(), 20, True))
    pydoc.pager(matrix_str)


SUPPORTED_VIEW_CHAIN: list[Views] = [Views.LANDING, Views.PROJECT, Views.MAP, Views.CELL]


class CLIController():
    def __init__(self, project_locator: WorldBuilderProjectDirectory):
        self._project_locator = project_locator
        self._view_chain: list[Views] = [Views.LANDING]
        self._project: WorldBuilderProject = None
        self._map_root: MapRoot = None
        self._cell: str = None

    @property
    def project(self) -> WorldBuilderProject:
        return self._project

    @property
    def map_root(self) -> MapRoot:
        return self._map_root

    async def run(self, exit_view: Optional[Views] = None):
        self._view_chain = [Views.LANDING]
        should_exit: bool = False
        while not should_exit:
            try:
                await self._interact()
            except BackDirectiveException:
                #if self._cell == ROOT_CELL_IDENTIFIER:
                #    self.pop_view()
                #    self.pop_view()
                #else:
                self.pop_view()
            except ExitDirectiveException:
                should_exit = True
            if exit_view is not None and self._view_chain[-1] == exit_view:
                should_exit = True

    def append_view(self, view: Views):
        # Check if the view is a valid next view
        if len(self._view_chain) + 1 > len(SUPPORTED_VIEW_CHAIN):
            raise ValueError(f"Invalid view chain: {self._view_chain}")
        for index, view in enumerate(self._view_chain + [view]):
            if isinstance(SUPPORTED_VIEW_CHAIN[index], list):
                if view not in SUPPORTED_VIEW_CHAIN[index]:
                    raise ValueError(f"Invalid view chain: {self._view_chain}")
            else:
                if view != SUPPORTED_VIEW_CHAIN[index]:
                    raise ValueError(f"Invalid view chain: {self._view_chain}")
        self._view_chain.append(view)

    def pop_view(self):
        if self._view_chain[-1] == Views.LANDING:
            return
        elif self._view_chain[-1] == Views.PROJECT:
            self._project = None
        elif self._view_chain[-1] == Views.MAP:
            self._map_root = None
        elif self._view_chain[-1] == Views.CELL:
            self._cell = None
        self._view_chain.pop()

    async def _interact(self):
        if self._view_chain[-1] == Views.LANDING:
            self.append_view(Views.PROJECT)
            self._project = select_project(self._project_locator)
        elif self._view_chain[-1] == Views.PROJECT:
            self._map_root = select_map_root(self._project)
            if self._map_root is not None:
                self.append_view(Views.MAP)
        elif self._view_chain[-1] == Views.MAP:
            cell_identifier: str = interact_with_map_root(self._map_root)
            if cell_identifier is not None:
                self._cell = cell_identifier
                self.append_view(Views.CELL)
        elif self._view_chain[-1] == Views.CELL:
            assert self._cell is not None
            cell: str = await interact_with_cell(self._map_root, self._cell)
            if cell is not None:
                self._cell = cell


"""
Open
List
List with data
Display
Up
"""

async def start():
    controller = CLIController(WorldBuilderProjectDirectory())
    await controller.run()

def start_interactive_console():
    vars = globals()
    vars.update(locals())

    readline.set_completer(rlcompleter.Completer(vars).complete)
    if sys.platform == 'darwin':
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(vars).interact()


async def start_interpreter():
    global project
    global map_root

    controller = CLIController(WorldBuilderProjectDirectory())
    await controller.run(exit_view=Views.MAP)
    while controller.project is not None:    
        project = controller.project
        map_root = controller.map_root
        start_interactive_console()
        await controller.run(exit_view=Views.MAP)
    


def main():
    use_interpreter_mode: bool = any([arg in {"-i", "--interpreter"} for arg in sys.argv])
    if use_interpreter_mode:
        asyncio.run(start_interpreter())
    else:
        asyncio.run(start())


if __name__ == "__main__":
    print(sys.argv)



    main()
