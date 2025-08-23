#!/usr/bin/env python3
"""Generate SQLModel classes from PostgreSQL database schema.

Usage:
    python generate_sqlmodels.py [OUTPUT_FILE]

Examples:
    python generate_sqlmodels.py                    # Output to stdout
    python generate_sqlmodels.py app/models/db.py   # Output to file

Available sqlacodegen options (add to command as needed):
    --tables TABLE_NAME       Generate only specific tables
    --schema SCHEMA          Use specific schema (default: public)
    --noindexes             Skip index definitions
    --noconstraints         Skip constraint definitions
    --nocomments            Skip column comments
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import click
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""
    host: str = Field(...)
    user: str = Field(...)
    password: str = Field(...)
    port: int = Field(5432)
    name: str = Field("postgres")
    
    @property
    def connection_string(self) -> str:
        """PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


def load_database_config() -> DatabaseConfig:
    """Load database configuration from environment."""
    backend_dir = Path(__file__).parent.parent
    env_file = backend_dir / '.env'
    
    if env_file.exists():
        load_dotenv(env_file)
    
    try:
        return DatabaseConfig(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=int(os.getenv('DB_PORT', 5432)),
            name=os.getenv('DB_NAME', 'postgres')
        )
    except (ValidationError, TypeError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)


def run_sqlacodegen(connection_string: str, output_file: Optional[Path] = None) -> None:
    """Execute sqlacodegen to generate SQLModel classes."""
    cmd = ['sqlacodegen', '--generator', 'sqlmodels', connection_string]
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--outfile', str(output_file)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout and not output_file:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: sqlacodegen not installed. Run: pip install sqlacodegen[sqlmodels]", 
              file=sys.stderr)
        sys.exit(1)


@click.command()
@click.argument('output', type=click.Path(path_type=Path), required=False)
def main(output: Optional[Path]):
    """Generate SQLModel classes from PostgreSQL database."""
    config = load_database_config()
    run_sqlacodegen(config.connection_string, output)


if __name__ == "__main__":
    main()