from enum import StrEnum
import sys
from typing import Callable

from gstk.creation.api import CreationProject, get_creation_project, ProjectProperties, new_creation_project

# START Monkey patch.
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
# END Monkey patch.


from world_builder.project import WorldBuilderProject, WorldBuilderProjectLocator


from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import print_function

class PromptOptions(StrEnum):
    USER_OPTION = "user_option"


class ModeOptions(StrEnum):
    NEW_PROJECT = "New project"
    OPEN_PROJECT = "Open project"
    LIST_PROJECTS = "List projects"


def get_validate_new_project_id_fn(project_locator: WorldBuilderProjectLocator):
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


mode_question = [
    {
        'type': 'list',
        'name': 'user_option',
        'message': 'Welcome to WorldBuilder',
        'choices': [mode.value for mode in ModeOptions]
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


new_questions = [

]


open_questions = [

]


def print_project_id_list(project_locator: WorldBuilderProjectLocator):
    print("Projects:")
    project_ids: list[str] = project_locator.list_project_ids()
    if not project_ids:
        print("No projects found.")
    for project_id in project_ids:
        print(project_id)


def main():
    project_locator = WorldBuilderProjectLocator()
    answer: dict = prompt(mode_question)
    while answer[PromptOptions.USER_OPTION] not in [ModeOptions.NEW_PROJECT, ModeOptions.OPEN_PROJECT]:
        if answer[PromptOptions.USER_OPTION] == ModeOptions.LIST_PROJECTS:
            print_project_id_list(project_locator)
        answer = prompt(mode_question)

    project: WorldBuilderProject
    if answer[PromptOptions.USER_OPTION] == ModeOptions.NEW_PROJECT:
        answer_new: dict = prompt(get_new_project_questions_list(get_validate_new_project_id_fn(project_locator)))
        project_properties: ProjectProperties = ProjectProperties(
            id=answer_new['project_id'],
            name=answer_new['project_name'],
            description=answer_new['project_description']
        )
        project = new_creation_project(project_properties.id, project_properties, resource_locator=project_locator, project_class=WorldBuilderProject)
    elif answer[PromptOptions.USER_OPTION] == ModeOptions.OPEN_PROJECT:
        answer_open: dict = prompt(get_open_project_questions_list(project_locator.list_project_ids()))
        project_id: str = answer_open[PromptOptions.USER_OPTION]
        project = get_creation_project(project_id, resource_locator=project_locator, project_class=WorldBuilderProject)

    if project:
        print(f"Project {project.project_node.id} opened.")

if __name__ == "__main__":
    main()