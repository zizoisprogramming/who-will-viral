# who-will-viral

![PyPI version](https://img.shields.io/pypi/v/who_will_viral.svg)

This project predicts if the youtube video is going to trend or not based on some features like description, number of likes, number of comments, etc.. .

* [GitHub](https://github.com/zizoisprogramming/who_will_viral/) | [PyPI](https://pypi.org/project/who_will_viral/) | [Documentation](https://zizoisprogramming.github.io/who_will_viral/)
* Created by [team_15](https://audrey.feldroy.com/) | GitHub [@zizoisprogramming](https://github.com/zizoisprogramming) | PyPI [@zizoisprogramming](https://pypi.org/user/zizoisprogramming/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://zizoisprogramming.github.io/who_will_viral/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/who_will_viral.git
cd who_will_viral

# Install all dependencies (including dev)
poetry install
```

Run tests:

```bash
poetry run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

who-will-viral was created in 2026 by team_15.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
