import json
import os
from pathlib import Path
import sys
from typing import Optional

from platformdirs import user_config_dir
from pydantic import BaseModel

from world_builder.config import APPNAME, SETTINGS_FILENAME


class Settings(BaseModel):
    default_project: Optional[str] = None
    default_map: Optional[str] = None

def get_settings_path() -> Path:
    if sys.platform == "darwin":
        # use .config on macOS
        return Path(os.path.expanduser("~/.config")) / APPNAME / SETTINGS_FILENAME
    else:
        return Path(user_config_dir(APPNAME)) / SETTINGS_FILENAME


def get_settings() -> Optional[Settings]:
    settings_path: Path = get_settings_path()
    if not settings_path.exists():
        return None
    with settings_path.open("r") as settings_file:
        content = settings_file.read()
        if not content.strip():
            return None
        return Settings(**json.loads(content))


def set_settings(settings: Settings):
    settings_path: Path = get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w") as settings_file:
        json.dump(settings.model_dump(), settings_file, indent=4)