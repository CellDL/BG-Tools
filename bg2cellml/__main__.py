#===============================================================================
#
#  CellDL and bondgraph tools
#
#  Copyright (c) 2020 - 2025 David Brooks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#===============================================================================

from pathlib import Path

import structlog

#===============================================================================

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ]
)

logger = structlog.get_logger()

#===============================================================================

try:
    from bg2cellml.bondgraph import BondgraphModel, BondgraphModelSource
except Exception as error:
    logger.exception(error, exc_info=True)
    exit(1)

from bg2cellml.cellml import CellMLModel

from graph2celldl import Graph2CellDL

#===============================================================================

#===============================================================================

BGF_STYLESHEET = """
    .celldl-Component {
        stroke: none;
    }
    rect.celldl-Component {
        stroke-width: 2px;
    }

    text {
        font-size: 30px;
    }

    .bgf-HydraulicCapacitor rect.celldl-Component,
    .bgf-ElasticVessel rect.celldl-Component {
        fill: #DBEDD0;
        stroke: #FB0006;
    }
    .bgf-HydraulicCapacitor text,
    .bgf-ElasticVessel text,
    .bgf-OneNode text {
        fill: #15A43F;
    }

    .bgf-HydraulicResistor rect.celldl-Component {
        fill: #FED254 ;
        stroke: #15A43F;
    }
    .bgf-HydraulicResistor text {
        fill: #000;
    }

    .bgf-ZeroNode rect.celldl-Component {
        fill: #F8DECC;
        stroke: #FB0006;
    }
    .bgf-ZeroNode text {
        fill: #FB0006;
    }

    .bgf-OneNode rect.celldl-Component {
        fill: #DBEDD0;
        stroke: #15A43F;
    }
"""

#===============================================================================

def model2cellml(model: BondgraphModel, cellml_file: str):
#=========================================================
    cellml = CellMLModel(model).to_xml()
    with open(cellml_file, 'w') as fp:
        fp.write(cellml)
        print(f'Created {cellml_file}')

def model2celldl(model: BondgraphModel, celldl_file: str):
#=========================================================
    G = model.network_graph
    G.graph['k'] = 0.1                   # Parameter for spring layout
    celldl_graph = Graph2CellDL(G, layout_method='spring',
        stylesheet=BGF_STYLESHEET, node_size=(150, 100), connection_stroke_width=6)
    celldl_graph.save_diagram(celldl_file)
    print(f'Created {celldl_file}')

def bg2cellml(bondgraph_rdf_source: str):
#========================================
    source = Path(bondgraph_rdf_source)
    if not source.exists():
        raise IOError(f'Missing BG-RDF source file: {bondgraph_rdf_source}')
    for model in BondgraphModelSource(bondgraph_rdf_source).models:
        model2cellml(model, f'{source.stem}.cellml')
        model2celldl(model, f'{source.stem}.celldl.svg')

#===============================================================================

def main():
    #bg2cellml('../examples/example_RCR.ttl')
    #bg2cellml('../examples/example_A1.ttl')
    #bg2cellml('../examples/example_A2.ttl')
    bg2cellml('../BVC-model/bondgraph.ttl')

#===============================================================================

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        logger.exception(error, exc_info=True)
        exit(1)

#===============================================================================
#===============================================================================
