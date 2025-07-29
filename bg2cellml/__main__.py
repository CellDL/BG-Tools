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

from collections import defaultdict
from pathlib import Path
from typing import Optional

#===============================================================================

import libopencor as loc
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

    .bgf-HydraulicStorage rect.celldl-Component,
    .bgf-ElasticVessel rect.celldl-Component {
        fill: #DBEDD0;
        stroke: #FB0006;
    }
    .bgf-HydraulicStorage text,
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

def string_to_list(string: str) -> list[int]:
#============================================
    return [ord(x) for x in string]

def model2cellml(model: BondgraphModel, cellml_file: str, save_if_errors: bool=False):
#=====================================================================================
    cellml = CellMLModel(model).to_xml()
    file = loc.File(cellml_file, False)
    file.contents = string_to_list(cellml)
    has_issues = False
    if file.has_issues:
        for issue in file.issues:
            print(issue.description)
        print(f'{file.issue_count} CellML validation issues...')
        has_issues = True
    else:
        simulation = loc.SedDocument(file)
        if simulation.has_issues:
            for issue in simulation.issues:
                print(issue.description)
            print(f'{simulation.issue_count} issues creating simulation from CellML...')
            has_issues = True
        else:
            simulation.simulations[0].output_end_time = 0.1
            simulation.simulations[0].number_of_steps = 10

            instance = simulation.instantiate(True)
            instance.run()
            if instance.has_issues:
                for issue in instance.issues:
                    print(issue.description)
                print(f'{instance.issue_count} issues running simulation created from CellML...')
                has_issues = True

    if has_issues and not save_if_errors:
        print('No CellML generated')
    else:
        with open(cellml_file, 'w') as fp:
            fp.write(cellml)
            print(f'Generated {cellml_file}')

def model2celldl(model: BondgraphModel, celldl_file: str):
#=========================================================
    G = model.network_graph

    # Merge together multiple nodes for an element
    element_nodes = defaultdict(list)
    for node, element in G.nodes(data='element'):
        if element is not None:
            element_nodes[element.uri].append(node)     # type: ignore
    for element, nodes in element_nodes.items():
        if len(nodes) > 1:
            base_node = nodes[0]
            for node in nodes[1:]:
                for edge_domain in G.in_edges(node, data='domain'):
                    G.add_edge(edge_domain[0], base_node, domain=edge_domain[2])
                for edge_domain in G.out_edges(node, data='domain'):
                    G.add_edge(base_node, edge_domain[1], domain=edge_domain[2])
                G.remove_node(node)

    G.graph['k'] = 0.1                   # Parameter for spring layout
    celldl_graph = Graph2CellDL(G, layout_method='spring',
        stylesheet=BGF_STYLESHEET, node_size=(150, 100), connection_stroke_width=6)
    celldl_graph.save_diagram(celldl_file)
    print(f'Created {celldl_file}')

def bg2cellml(bondgraph_rdf_source: str, output_rdf: Optional[str]=None, save_cellml_if_errors: bool=False):
#===========================================================================================================
    source = Path(bondgraph_rdf_source)
    if not source.exists():
        raise IOError(f'Missing BG-RDF source file: {bondgraph_rdf_source}')
    for model in BondgraphModelSource(bondgraph_rdf_source, output_rdf=output_rdf).models:
        model2cellml(model, f'{source.stem}.cellml', save_cellml_if_errors)
        model2celldl(model, f'{source.stem}.celldl.svg')

#===============================================================================

def main():
    bg2cellml('../examples/example_RC.ttl', save_cellml_if_errors=True)
    #bg2cellml('../examples/example_RCR.ttl')
    #bg2cellml('../examples/example_A1.ttl')
    #bg2cellml('../examples/example_A2.ttl')
    #bg2cellml('../BVC-model/bvc.ttl', save_cellml_if_errors=True)
    bg2cellml('../Blood-volume-control/model/bvc.ttl', save_cellml_if_errors=True, output_rdf='bvc_saved.ttl')

#===============================================================================

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        logger.exception(error, exc_info=True)
        exit(1)

#===============================================================================
#===============================================================================
