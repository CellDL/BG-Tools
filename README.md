# Generate CellML from BG-RDF models

> [!WARNING]
> These tools are actively being developed and nothing should be considered stable until version `1.0` has been released.

## Installation

### As a package

```sh
uv add https://github.com/CellDL/BG-Tools/releases/download/v0.7.2/bg2cellml-0.7.2-py3-none-any.whl
```

### Development

```sh
$ git https://github.com/CellDL/BG-Tools
$ cd BG-Tools
$ git submodule update --init --recursive
$ uv sync
```

## Usage

```Python
from bg2cellml.bondgraph import BondgraphModelSource
from bg2cellml.cellml import CellMLModel

# NB: a source file might contain several models

for model in BondgraphModelSource(bondgraph_rdf_source).models:
    cellml = CellMLModel(model).to_xml()
```

## Example models:

* [An RC electrical circuit](https://github.com/CellDL/BG-RDF/blob/main/examples/example_RC.ttl) 
* [A simple biochemical reaction](https://github.com/CellDL/BG-RDF/blob/main/examples/example_B1.ttl)
