from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
def adapters() -> None:
    """List registered memory adapters."""
    from lens.adapters.registry import list_adapters

    adapter_map = list_adapters()

    table = Table(title="Registered Adapters")
    table.add_column("Name", style="bold cyan")
    table.add_column("Class", style="dim")
    table.add_column("Module", style="dim")

    for name, cls in sorted(adapter_map.items()):
        table.add_row(name, cls.__name__, cls.__module__)

    console.print(table)


@click.command()
def metrics() -> None:
    """List registered scoring metrics."""
    from lens.scorer.registry import list_metrics

    metric_map = list_metrics()

    table = Table(title="Registered Metrics")
    table.add_column("Name", style="bold cyan")
    table.add_column("Tier", style="bold")
    table.add_column("Description")

    for name, cls in sorted(metric_map.items()):
        instance = cls()
        table.add_row(name, str(instance.tier), instance.description)

    console.print(table)
