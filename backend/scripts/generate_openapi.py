#!/usr/bin/env python3
"""Generate OpenAPI schema from FastAPI application."""

import json
import sys
from pathlib import Path
import click

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@click.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default="json",
    help="Output format for the OpenAPI schema",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path. If not specified, outputs to stdout",
)
@click.option(
    "--indent",
    "-i",
    type=int,
    default=2,
    help="Indentation for JSON output (default: 2)",
)
def generate_openapi(format: str, output: str, indent: int):
    """Generate OpenAPI schema from FastAPI application.

    Examples:
        # Generate JSON to stdout
        python scripts/generate_openapi.py

        # Generate YAML to file
        python scripts/generate_openapi.py --format yaml --output openapi.yaml

        # Generate JSON with 4-space indentation
        python scripts/generate_openapi.py --indent 4
    """
    # Get OpenAPI schema from FastAPI app
    openapi_schema = app.openapi()

    if format.lower() == "json":
        # Format as JSON
        output_content = json.dumps(openapi_schema, indent=indent, ensure_ascii=False)
    elif format.lower() == "yaml":
        # Format as YAML
        try:
            import yaml
        except ImportError:
            click.echo(
                "Error: PyYAML is not installed. Install it with: uv add pyyaml",
                err=True
            )
            sys.exit(1)

        # Configure YAML to output in a readable format
        output_content = yaml.dump(
            openapi_schema,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120
        )

    # Write to file or stdout
    if output:
        output_path = Path(output)
        output_path.write_text(output_content, encoding="utf-8")
        click.echo(f"OpenAPI schema written to: {output_path}")
    else:
        click.echo(output_content)


if __name__ == "__main__":
    generate_openapi()
