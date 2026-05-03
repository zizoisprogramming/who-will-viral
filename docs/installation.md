# Installation

## From PyPI

Install the latest stable release:

```bash
pip install who_will_viral
```

Or with Poetry:

```bash
poetry add who_will_viral
```

## From Source

Clone the repository and install with Poetry:

```bash
git clone https://github.com/zizoisprogramming/who_will_viral.git
cd who_will_viral
poetry install
```

Or download the source tarball:

```bash
curl -OJL https://github.com/zizoisprogramming/who_will_viral/tarball/main
cd who_will_viral
poetry install
```

## Requirements

- **Python**: 3.12 or higher
- **Package Manager**: Poetry (recommended)
- **API Key**: YouTube API key for data acquisition

## Verify Installation

After installation, verify everything works:

```bash
poetry run who_will_viral --help
```

You should see the CLI help output with available commands.

## Development Setup

For development and contributing:

```bash
git clone https://github.com/zizoisprogramming/who_will_viral.git
cd who_will_viral
poetry install
```

Then run tests to verify:

```bash
poetry run pytest
```
