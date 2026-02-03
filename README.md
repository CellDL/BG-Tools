# Generate CellML from BG-RDF models

> [!WARNING]
> These tools are actively being developed and nothing should be considered stable until version `1.0` has been released.

## Installation

### As a package

```sh
uv add https://github.com/CellDL/BG-Tools/releases/download/v0.9.0/bg2cellml-0.9.0-py3-none-any.whl
```

### Development

```sh
$ git clone https://github.com/CellDL/BG-Tools
$ cd BG-Tools
$ git submodule update --init --recursive
$ uv sync --all-groups
```

## Usage

See `cli/bgtool/__main__.py` for details.

## Example models:

* [An RC electrical circuit](https://github.com/CellDL/BG-RDF/blob/main/examples/example_RC.ttl) 
* [A simple biochemical reaction](https://github.com/CellDL/BG-RDF/blob/main/examples/example_B1.ttl)
* [All examples](https://github.com/CellDL/BG-RDF/blob/main/examples)

### Examples in a Jupyter notebook:

1. Clone the repository and its sub-modules as above.
2. Install with `uv sync --group notebook`
3. Start JupyterLab with `uv run jupyter lab`
4. Open `./notebooks/bg2cellml.ipynb` in Jupyter.
