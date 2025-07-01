# Generate CellML from BG-RDF models

## Installation

```sh
$ git https://github.com/CellDL/BG-Tools
$ cd BG-Tools
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
