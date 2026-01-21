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

from typing import Optional, Sequence, TYPE_CHECKING

#===============================================================================

import networkx as nx

#===============================================================================

from ..cellml import CellMLModel
from ..mathml import Equation
from ..rdf import ResultRow, RdfGraph, NamedNode, literal_as_string
from ..rdf import isBlankNode, isNamedNode, namedNode, Triple
from ..utils import Issue

from .framework_support import TRANSFORM_JUNCTION, TRANSFORM_PORT_IDS
from .model_support import BondgraphBond, BondgraphElement, BondgraphJunction
from .model_support import make_element_port_uri, make_symbolic_name
from .namespaces import BGF, NAMESPACES, get_curie
from .utils import Labelled, optional_integer, pretty_name, pretty_uri

if TYPE_CHECKING:
    from .framework import BondgraphFramework

#===============================================================================
#===============================================================================

MODEL_ELEMENTS = """
    SELECT DISTINCT ?uri ?type ?label ?domain ?symbol ?species ?location
    WHERE {
        <%MODEL%> bgf:hasBondElement ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri bgf:hasDomain ?domain }
        OPTIONAL { ?uri bgf:hasSymbol ?symbol }
        OPTIONAL { ?uri bgf:hasSpecies ?species }
        OPTIONAL { ?uri bgf:hasLocation ?location }
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri ?type"""

MODEL_JUNCTIONS = """
    SELECT DISTINCT ?uri ?type ?label ?value ?symbol ?species ?location
    WHERE {
        <%MODEL%> bgf:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri bgf:hasValue ?value }
        OPTIONAL { ?uri bgf:hasSymbol ?symbol }
        OPTIONAL { ?uri bgf:hasSpecies ?species }
        OPTIONAL { ?uri bgf:hasLocation ?location }
        OPTIONAL { ?uri rdfs:label ?label }
    }"""

MODEL_BONDS = """
    SELECT DISTINCT ?powerBond ?source ?target ?label ?bondCount
    WHERE {
        <%MODEL%> bgf:hasPowerBond ?powerBond .
        OPTIONAL { ?powerBond bgf:hasSource ?source }
        OPTIONAL { ?powerBond bgf:hasTarget ?target }
        OPTIONAL { ?powerBond rdfs:label ?label }
        OPTIONAL { ?powerBond bgf:bondCount ?bondCount }
    }"""

#===============================================================================
#===============================================================================

BONDGRAPH_MODEL = """
    SELECT DISTINCT ?uri ?label
    WHERE {
        ?uri a bgf:BondgraphModel .
        OPTIONAL { ?uri rdfs:label ?label }
    }"""

#===============================================================================

BONDGRAPH_BONDS = """
    SELECT DISTINCT ?bond ?source ?target ?sourceElement ?targetElement
    WHERE {
        {
            { ?bond
                bgf:hasSource ?source ;
                bgf:hasTarget ?target .
            }
      UNION { ?bond
                bgf:hasSource ?source ;
                bgf:hasTarget ?target .
                ?target
                    bgf:element ?targetElement ;
                    bgf:port ?targetPort .
            }
      UNION { ?bond
                bgf:hasTarget ?target ;
                bgf:hasSource ?source .
                ?source
                    bgf:element ?sourceElement ;
                    bgf:port ?sourcePort .
            }
      UNION { ?bond
                bgf:hasSource ?source ;
                bgf:hasTarget ?target .
                ?source
                    bgf:element ?sourceElement ;
                    bgf:port ?sourcePort .
               ?target
                    bgf:element ?targetElement ;
                    bgf:port ?targetPort .
            }
       }
    }"""

#===============================================================================
#===============================================================================

class BondgraphModel(Labelled):   ## Component ??
    def __init__(self, framework: 'BondgraphFramework', base_iri: str, rdf_source: str, debug=False):
        self.__issues: list[Issue] = []
        self.__framework = framework
        self.__rdf_graph = RdfGraph(NAMESPACES)
        self.__debug = debug
        self.__elements: list[BondgraphElement] = []
        self.__junctions: list[BondgraphJunction] = []
        self.__bonds: list[BondgraphBond] = []
        self.__graph = nx.DiGraph()
        (model_uri, label) = self.__load_rdf(base_iri, rdf_source)
        if model_uri is not None:
            super().__init__(namedNode(model_uri), label)   # pyright: ignore[reportArgumentType]
            self.__initialise()

    def report_issue(self, reason: str):
    #===================================
        self.__issues.append(Issue(reason))

    def __load_rdf(self, base_iri: str, rdf_source: str) -> tuple[Optional[str], Optional[str]]:
    #===========================================================================================
        self.__rdf_graph.load(base_iri, rdf_source)
        models: dict[str, Optional[str]] = {}
        for row in self.__rdf_graph.query(BONDGRAPH_MODEL):
            # ?uri ?label
            models[row['uri'].value] = label.value if (label := row.get('label')) is not None else None # pyright: ignore[reportOptionalMemberAccess]
        if len(models) == 0:
            self.report_issue('No BondgraphModels defined in RDF source')
            return (None, None)
        elif len(models) > 1:
            self.report_issue("Multiple BondgraphModels defined in RDF source -- a model's URI must be given")
            return (None, None)
        return list(models.items())[0]

    def __initialise(self):
    #======================
        self.__generate_bonds()
        last_element_uri: Optional[str] = None
        last_element_name: Optional[str] = None
        element = None
        symbol = None
        for row in self.__rdf_graph.query(MODEL_ELEMENTS.replace('%MODEL%', self.uri)):
            # ?uri ?type ?domain ?symbol ?species ?location ?label ORDER BY ?uri ?type
            if row['type'].value.startswith(NAMESPACES['bgf']):                             # pyright: ignore[reportOptionalMemberAccess]
                if row['uri'].value != last_element_uri:                                    # pyright: ignore[reportOptionalMemberAccess]
                    element = None
                    last_element_uri = row['uri'].value
                    symbol = make_symbolic_name(row)
                    if symbol:
                        last_element_name = pretty_name(symbol, last_element_uri)
                    else:
                        last_element_name = pretty_name('', last_element_uri)
                element_type: NamedNode = row['type']                                       # pyright: ignore[reportAssignmentType]
                template = self.__framework.element_template(element_type, row.get('domain'))   # pyright: ignore[reportArgumentType]
                if template is None:
                    self.report_issue(f'BondElement {last_element_name} has an unknown BG-RDF template: {get_curie(element_type)}')
                    continue
                else:
                    if element is None:
                        element = BondgraphElement.for_model(self, row['uri'], template,    # pyright: ignore[reportArgumentType]
                                                             row.get('domain'),             # pyright: ignore[reportArgumentType]
                                                             symbol, literal_as_string(row.get('label')))   # pyright: ignore[reportArgumentType]
                        self.__elements.append(element)
                    else:
                        self.report_issue(f'BondElement {last_element_name} has multiple BG-RDF templates')   # pyright: ignore[reportArgumentType]
                        continue

        if len(self.__elements) == 0:
            self.report_issue(f'Model {(pretty_uri(self.uri))} has no elements...')
            return

        for row in self.__rdf_graph.query(MODEL_JUNCTIONS.replace('%MODEL%', self.uri)):
            # ?uri ?type ?value ?symbol ?species ?location ?label
            if row['type'].value.startswith(NAMESPACES['bgf']):                             # pyright: ignore[reportOptionalMemberAccess]
                symbol = make_symbolic_name(row)
                self.__junctions.append(
                    BondgraphJunction(self, row['uri'], row['type'], row.get('value'),      # pyright: ignore[reportArgumentType]
                                      symbol, literal_as_string(row.get('label'))))         # pyright: ignore[reportArgumentType]
        for row in self.__rdf_graph.query(MODEL_BONDS.replace('%MODEL%', self.uri)):
            # ?powerBond ?source ?target ?label ?bondCount
            bond_uri: NamedNode = row['powerBond']                                      # pyright: ignore[reportAssignmentType]
            if row['source'] is None or row['target'] is None:
                self.report_issue(f'Bond {pretty_uri(bond_uri)} is missing source and/or target node')
                continue
            self.__bonds.append(
                BondgraphBond(self, bond_uri, row.get('source'), row.get('target'), row.get('label'), optional_integer(row.get('bondCount'))))    # pyright: ignore[reportArgumentType]

        self.__make_bond_network()
        self.__check_and_assign_domains_to_bond_network()
        self.__assign_junction_domain_and_variables()
        self.__assign_element_variables_and_equations()

        self.__equations: list[Equation] = []
        for element in self.__elements:
            if (cr := element.constitutive_relation) is not None:
                self.__equations.extend(cr.equations)
                for eq in cr.equations:
                    eq.provenance = 'cr'
            self.__equations.extend(element.equations)
            for eq in element.equations:
                eq.provenance = 'be'
        for junction in self.__junctions:
            equations = junction.build_equations(self.__graph)
            self.__equations.extend(equations)
            for eq in equations:
                eq.provenance = 'js'

        if self.__debug:
            print('Elements:')
            for element in self.__elements:
                print(' ', pretty_uri(element.uri))
                if (cr := element.constitutive_relation) is not None:
                    for eq in cr.equations:
                        print('   CR:', eq)
                for eq in element.equations:
                    print('   EQ:', eq)
            print('Junctions:')
            for junction in self.__junctions:
                print(' ', pretty_uri(junction.uri))
                equations = junction.equations
                for eq in equations:
                    print('   ', eq)

    def __generate_bonds(self):
    #==========================
        for row in self.__rdf_graph.query(BONDGRAPH_BONDS):
            # ?bond ?source ?target ?sourceElement ?targetElement
            if isNamedNode(row['source']):
                source = row['source']
            elif isBlankNode(row['source']) and isNamedNode(row.get('sourceElement')):
                source = row['sourceElement']
            else:
                source = None
            if isNamedNode(row['target']):
                target = row['target']
            elif isBlankNode(row['target']) and isNamedNode(row.get('targetElement')):
                target = row['targetElement']
            else:
                target = None
            if ((Triple(None, BGF.hasBondElement, source) in self.__rdf_graph
              or Triple(None, BGF.hasJunctionStructure, source) in self.__rdf_graph)
            and (Triple(None, BGF.hasBondElement, target) in self.__rdf_graph
              or Triple(None, BGF.hasJunctionStructure, target) in self.__rdf_graph)):
                self.__rdf_graph.add(Triple(namedNode(self.uri), BGF.hasPowerBond, row['bond']))

    @property
    def elements(self):
        return self.__elements

    @property
    def framework(self) -> 'BondgraphFramework':
        return self.__framework

    @property
    def has_issues(self) -> bool:
    #============================
        return len(self.__issues) > 0

    @property
    def issues(self) -> list[Issue]:
    #===============================
        return self.__issues

    @property
    def junctions(self):
        return self.__junctions

    @property
    def equations(self):
        return self.__equations

    @property
    def network_graph(self) -> nx.DiGraph:
        return self.__graph.copy()

    @property
    def rdf_graph(self) -> RdfGraph:
        return self.__rdf_graph

    def __assign_element_variables_and_equations(self):
    #==================================================
        for element in self.__elements:
            element.assign_variables(self.__graph)
        for element in self.__elements:
            element.build_expressions(self.__graph)

    def __assign_junction_domain_and_variables(self):
    #================================================
        for junction in self.__junctions:
            if junction.type != TRANSFORM_JUNCTION:
                junction.assign_node_variables(self.__graph)
        for junction in self.__junctions:
            if junction.type == TRANSFORM_JUNCTION:
                junction.assign_transform_variables(self.__graph)

    # Assign junction domains from elements and check consistency
    def __check_and_assign_domains_to_bond_network(self):
    #====================================================
        seen_nodes = set()
        undirected_graph = self.__graph.to_undirected(as_view=True)

        def check_node(node, domain):
            if node not in seen_nodes:
                seen_nodes.add(node)
                if 'domain' not in self.__graph.nodes[node]:
                    self.__graph.nodes[node]['domain'] = domain
                    for neighbour in undirected_graph.neighbors(node):
                        check_node(neighbour, domain)
                elif domain != (node_domain := self.__graph.nodes[node]['domain']):
                    self.report_issue(f'Node {node} with domain {node_domain} incompatible with {domain}')
                    return

        for element in self.__elements:
            for port_uri in element.power_ports.keys():
                check_node(port_uri, element.domain)

    # Construct network graph of PowerBonds
    def __make_bond_network(self):
    #=============================
        for element in self.__elements:
            for port_uri, port in element.power_ports.items():
                self.__graph.add_node(port_uri, uri=port_uri,
                    type=get_curie(element.type),
                    power_port=port, port_type=element.element_class,
                    element=element, label=element.symbol)
        for junction in self.__junctions:
            if junction.type == TRANSFORM_JUNCTION:
                # A Transform Node has two implicit ports, with ids `0` and `1`
                for port_id in TRANSFORM_PORT_IDS:
                    port_uri = make_element_port_uri(junction.uri, port_id)
                    self.__graph.add_node(port_uri, uri=port_uri,
                        type=get_curie(junction.type), junction=junction, label=junction.symbol)
            else:
                self.__graph.add_node(junction.uri, uri=junction.uri,
                    type=get_curie(junction.type), junction=junction, label=junction.symbol)
        for bond in self.__bonds:
            source = bond.source_id
            target = bond.target_id
            if source not in self.__graph and target not in self.__graph:
                self.report_issue(f'No element or junction for source {pretty_uri(source)} and target {pretty_uri(target)} of bond {pretty_uri(bond.uri)}')
                continue
            elif source not in self.__graph:
                self.report_issue(f'No element or junction for source {pretty_uri(source)} of bond {pretty_uri(bond.uri)}')
                continue
            elif target not in self.__graph:
                self.report_issue(f'No element or junction for target {pretty_uri(target)} of bond {pretty_uri(bond.uri)}')
                continue

            self.__graph.add_edge(source, target, bond_count=bond.bond_count)

    def sparql_query(self, query: str) -> Sequence[ResultRow]:
    #========================================================
        return self.__rdf_graph.query(query)

#===============================================================================

    def make_cellml_model(self) -> CellMLModel:
    #==========================================
        return CellMLModel(self)

#===============================================================================
#===============================================================================
