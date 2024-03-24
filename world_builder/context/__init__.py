import os
import importlib
from pathlib import Path

# Define the directory name that is a sibling to this __init__.py file
ENGINES_DIRECTORY: str = "engines"
engine_directory = (Path(__file__).resolve()).parent / ENGINES_DIRECTORY

# use pathlib Path iterdir to iterate engine_directory and get non __init__.py python files
module_files = [f for f in engine_directory.iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

# Import all modules found
for module_file in module_files:
    module_name = module_file.stem
    module_path = f"world_builder.context.{engine_directory.name}.{module_name}"
    print(f"Importing {module_path}")
    importlib.import_module(module_path)
