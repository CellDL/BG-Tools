# Generate CellML from BG-RDF models

## Installation

```sh
$ uv install
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

* [An RC electrical circuit](/examples/example_RC.ttl) 
* [A simple biochemical reaction](/examples/example_B1.ttl) -- [FAIRDO BG example 3.4](https://models.physiomeproject.org/e/b53/FAIRDO%20BG%20example%203.4.cellml/view) on PMR
