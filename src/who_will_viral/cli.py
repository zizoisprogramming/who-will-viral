"""Console script for who_will_viral."""

import typer
from rich.console import Console

from who_will_viral import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
	"""Console script for who_will_viral."""
	console.print('Replace this message by putting your code into who_will_viral.cli.main')
	console.print('See Typer documentation at https://typer.tiangolo.com/')
	utils.do_something_useful()


if __name__ == '__main__':
	app()
