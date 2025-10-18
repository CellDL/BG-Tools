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

#===============================================================================

import libopencor as loc

#===============================================================================

from bg2cellml.version import __version__

#===============================================================================

from bg2cellml.bondgraph import BondgraphModel, BondgraphModelSource
from bg2cellml.cellml import CellMLModel
from bg2cellml.utils import log, pretty_log

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

def model2cellml(model: BondgraphModel, cellml_file: Path, save_if_errors: bool=False):
#======================================================================================
    cellml = CellMLModel(model).to_xml()
    file = loc.File(str(cellml_file), False)
    file.contents = string_to_list(cellml)
    has_issues = False
    if file.has_issues:
        for issue in file.issues:
            log.warning(issue.description)
        log.warning(f'{file.issue_count} CellML validation issues...')
        has_issues = True
    else:
        simulation = loc.SedDocument(file)
        if simulation.has_issues:
            for issue in simulation.issues:
                log.warning(issue.description)
            log.warning(f'{simulation.issue_count} issues creating simulation from CellML...')
            has_issues = True
        else:
            simulation.simulations[0].output_end_time = 0.1
            simulation.simulations[0].number_of_steps = 10

            instance = simulation.instantiate()
            instance.run()
            if instance.has_issues:
                for issue in instance.issues:
                    log.warning(issue.description)
                log.warning(f'{instance.issue_count} issues running simulation created from CellML...')
                has_issues = True

    if has_issues and not save_if_errors:
        log.warning('No CellML generated')
    else:
        with open(cellml_file, 'w') as fp:
            fp.write(cellml)
            log.info(f'Generated {pretty_log(cellml_file)}')

def model2celldl(model: BondgraphModel, celldl_file: Path):
#==========================================================
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
    log.info(f'Created {pretty_log(celldl_file)}')

def bg2cellml(bondgraph_rdf_source: str, output_path: Path, save_rdf: bool=False, save_if_errors: bool=False, debug: bool=False):
#================================================================================================================================
    source = Path(bondgraph_rdf_source)
    if not source.exists():
        raise IOError(f'Missing BG-RDF source file: {bondgraph_rdf_source}')
    output_rdf = (output_path / f'{source.stem}.ttl') if save_rdf else None
    for model in BondgraphModelSource(bondgraph_rdf_source, output_rdf=output_rdf, debug=debug).models:
        model2cellml(model, output_path / f'{source.stem}.cellml', save_if_errors)
        model2celldl(model, output_path / f'{source.stem}.celldl.svg')

#===============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='BG-RDF to CellML and CellDL')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--debug', action='store_true', help='Show generated equations for model')
    parser.add_argument('--save-errors', action='store_true', help='Output CellML even if it has errors')
    parser.add_argument('--save-rdf', action='store_true', help='Optionally save intermediate RDF graph')
    parser.add_argument('--output', metavar='OUTPUT_DIR', required=True, help='Directory where generated files are saved')
    parser.add_argument('bg_rdf', metavar='BG-RDF', help='Input BG-RDF source file')

    args = parser.parse_args()

    bg2cellml(args.bg_rdf, Path(args.output), save_rdf=args.save_rdf, save_if_errors=args.save_errors, debug=args.debug)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
#===============================================================================
