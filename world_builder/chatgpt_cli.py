import asyncio
from enum import Enum
import os
import re
import sys
import time
from typing import Any, Optional, Callable


from gstk.llmlib.async_openai import get_chat_completion_response
from gstk.models.chatgpt import Message, Role

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.patch_stdout import patch_stdout

from world_builder.config import DEFAULT_MODEL
from world_builder.models import ContextBuffer

# /DIRECTIVE [id [id]]
DIRECTIVE_RE: re.Pattern = re.compile(r'\s*\/([a-zA-Z\-]+)\s*(\d+)?(\s+\d+)?')

def get_prompt():
    return [
        ('bg:cornsilk fg:maroon', ' Human:'),
        ('', ' '),
    ]


def choose_openai_model() -> str:
    return radiolist_dialog(
        title='Choosing a ChatGPT model',
        text='Which ChatGPT model would you like ?',
        values=[
            ('gpt-3.5-turbo', 'gpt-3.5-turbo'),
            ('gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k'),
            ('gpt-4', 'gpt-4'),
            ('gpt-4-1106-preview', 'gpt-4-1106-preview (turbo)'),
        ]
    ).run()

def get_role_html(role: Role, prefix: str = "") -> str:
    if role == Role.SYSTEM:
        return f'<style fg="ansiwhite" bg="ansigreen">{prefix}System:</style> '
    elif role == Role.USER:
        return f'<style fg="maroon" bg="cornsilk">{prefix}User:</style> '
    elif role == Role.ASSISTANT:
        return f'<style fg="ansiwhite" bg="ansiblue">{prefix}Assistant:</style> '
    elif role == Role.FUNCTION:
        return f'<style fg="ansiwhite" bg="ansiyellow">{prefix}Function:</style> '

def parse_directive(text: str) -> Optional[tuple[str, Optional[int], Optional[int]]]:
    match: re.Match = DIRECTIVE_RE.match(text)
    if match:
        groups: list[str] = match.groups()
        return groups[0], int(groups[1]) if groups[1] else None, int(groups[2]) if groups[2] else None
    return (None, None, None)

class DirectiveEnum(Enum):
    MOVE = [('move', 'm'), 'Move a message from one position to another.']
    DELETE = [('delete', 'd'), 'Delete a message.']
    SAVE = [('save', 's'), 'Save the current context.']
    SAVE_EXIT = [('save-exit',), 'Save the current context and exit.']
    NO_SAVE_EXIT = [('no-save-exit',), 'Exit without saving.']
    LIST = [('list', 'l'), 'List all messages.']

def process_directive(context_buffer: ContextBuffer, directive: str, id_1: Optional[int] = None, id_2: Optional[int] = None):
    if directive in DirectiveEnum.MOVE.value[0] and id_1 is not None and id_2 is not None:
        context_buffer.move(id_1, id_2)
        print_formatted_text(f'Moved {id_1} to {id_2}.')
        print_message_list(context_buffer.list())
    elif directive in DirectiveEnum.DELETE.value[0] and id_1 is not None and id_2 is None:
        context_buffer.delete(id_1)
        print_formatted_text('Deleted.')
        print_message_list(context_buffer.list())
    elif directive in DirectiveEnum.NO_SAVE_EXIT.value[0]:
        raise KeyboardInterrupt
    elif directive in DirectiveEnum.SAVE_EXIT.value[0]:
        context_buffer.save()
        print_formatted_text('Saved.')
        raise KeyboardInterrupt
    elif directive in DirectiveEnum.SAVE.value[0]:
        context_buffer.save()
        print_formatted_text('Saved.')
    elif directive in DirectiveEnum.LIST.value[0]:
        print_message_list(context_buffer.list())
    else:
        print_formatted_text(f'Unknown directive: {directive}')    

def print_message(message: Message, prefix: str = ""):
    print_formatted_text(HTML(get_role_html(message.role, prefix=f'{prefix}: ')))
    if message.content:
        print_formatted_text(message.content)

def print_message_list(messages: list[Message]):
    if not messages:
        return
    for index, message in enumerate(messages):
        print_message(message, prefix=str(index).zfill(len(str(len(messages)))))
        print()

async def run_chat_completion(context_buffer: ContextBuffer):
    print("ChatGPT CLI")
    # Print directive enum names and values.
    print_formatted_text("Directives:")
    for directive in DirectiveEnum:
        print_formatted_text(f'{" | ".join(directive.value[0])} - {directive.value[1]}')
    print_formatted_text('\nDirectives start with a / followed by a command and optionally an id (or two ids for move).')
    prompt_history = FileHistory('.prompt_history')
    session = PromptSession(history=prompt_history)

    print()
    print_message_list(context_buffer.list())

    with patch_stdout():
        while True:
            print_message(Message(role=Role.USER, content=""), prefix=str(len(context_buffer)).zfill(len(str(len(context_buffer)))))
            user_string: str = await session.prompt_async(
                [],
                cursor=CursorShape.BLOCK,
                auto_suggest=AutoSuggestFromHistory(),
                multiline=True,
            )
            if not user_string or not user_string.strip():
                continue
            directive, id_1, id_2 = parse_directive(user_string)
            if directive:
                process_directive(context_buffer, directive, id_1, id_2)
            else:
                message: Message = Message(role=Role.USER, content=user_string)
                context_buffer.add(message)
                print_message(Message(role=Role.ASSISTANT, content=""), prefix=str(len(context_buffer)).zfill(len(str(len(context_buffer)))))
                chat_completion = await get_chat_completion_response(
                    messages=context_buffer.list(),
                    chat_gpt_model=DEFAULT_MODEL,
                )
                reply = chat_completion.choices[0].message.content
                print(reply)
                time.sleep(2)
                context_buffer.add(Message(role=Role.ASSISTANT, content=reply))

if __name__ == '__main__':
    try:
        model = choose_openai_model()
        if model:
            asyncio.run(run_chat_completion(model))
    except KeyboardInterrupt:
        print_formatted_text('GoodBye!')