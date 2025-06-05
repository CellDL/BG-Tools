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
from typing import Optional

from pprint import pprint

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import Labelled, NamespaceMap
from ..units import Value

from .framework import BondgraphFramework as FRAMEWORK, Domain, Variable
from .framework import ONENODE_JUNCTION, TRANSFORM_JUNCTION, ZERONODE_JUNCTION

from .namespaces import NAMESPACES

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = f"""
    SELECT DISTINCT ?symbol ?value
    WHERE {{
        %ELEMENT_URI% bgf:hasVariable [
            bgf:hasSymbol ?symbol ;
            bgf:hasValue ?value
        ] .
    }}"""

#===============================================================================
#===============================================================================

MODEL_ELEMENTS = f"""
    SELECT DISTINCT ?uri ?type ?label
    WHERE {{
        %MODEL% bg:hasBondElement ?uri .
        ?uri a ?type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        FILTER (?type IN ({', '.join(FRAMEWORK.element_classes())}))
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphElement(Labelled):
    def __init__(self, uri: str, template: str, label: Optional[str]):
        super().__init__(uri, label)
        # print(uri, [str(v) for v in params.values()], [str(v) for v in states.values()])
        bond_element = FRAMEWORK.element(template)
        if bond_element is None:
            raise ValueError(f'Unknown BondElement {template} for node {uri}')
        self.__constitutive_relation = bond_element.constitutive_relation.copy()
        self.__domain = bond_element.domain
        self.__type = bond_element.uri
        self.__ports = {
            (self.uri if port.port_id is None else f'{self.uri}_{port.port_id}'): port
                for port in bond_element.ports }
        self.__variables = bond_element.variables.copy()
        # Assign variable names, substituting them into the constitutive relation
        self.__assign_variable_names()

    @classmethod
    def for_model(cls, model: 'BondgraphModel', *args):
        self = cls(*args)
        for row in model.source.sparql_query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri)):
            if row[0] not in self.__variables:
                raise ValueError(f'Element {self.uri} has unknown symbol {row[0]} for {self.__template.uri}')   # type: ignore
            self.__variables[row[0]].set_value(Value(row[1]))
        return self

    @property
    def domain(self):
        return self.__domain

    @property
    def num_ports(self):
        return len(self.__ports)

    @property
    def ports(self):
        return self.__ports

    @property
    def constitutive_relation(self):
        return self.__constitutive_relation

    @property
    def type(self):
        return self.__type

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def assign_domains(self, bond_graph: nx.DiGraph):
    #================================================
        seen_nodes = set()
        undirected_graph = bond_graph.to_undirected(as_view=True)

        def check_node(node, domain):
            if node not in seen_nodes:
                seen_nodes.add(node)
                if 'domain' not in bond_graph.nodes[node]:
                    bond_graph.nodes[node]['domain'] = domain
                    for neighbour in undirected_graph.neighbors(node):
                        check_node(neighbour, domain)
                elif domain != (node_domain := bond_graph.nodes[node]['domain']):
                    raise ValueError(f'Node {node} with domain {node_domain} incompatible with {domain}')
                ##TRANSFORM_JUNCTION_TYPE

        check_node(self.uri, self.domain)

    def __assign_variable_names(self):
    #=================================
        def substitute(symbol: str, prefix: str):
            self.__constitutive_relation.substitute(symbol, f'{prefix}_{symbol}')
        for port_name, port in self.__ports.items():
            substitute(port.flow.symbol, port_name)
            substitute(port.potential.symbol, port_name)
        for symbol in self.__variables.keys():
            substitute(symbol, self.uri)

#===============================================================================
#===============================================================================

MODEL_BONDS = f"""
    SELECT DISTINCT ?uri ?source ?target ?label
    WHERE {{
        %MODEL% bg:hasPowerBond ?uri .
        ?uri bgf:hasSource ?source .
        ?uri bgf:hasTarget ?target .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }}"""

MODEL_BOND_PORTS = f"""
    SELECT DISTINCT ?element ?port
    WHERE {{
        %MODEL% bg:hasPowerBond %BOND% .
        %BOND% %BOND_RELN% [
            bgf:element ?element ;
            bgf:port ?port
        ]
    }}"""



#===============================================================================

class BondgraphBond(Labelled):
    def __init__(self, model: 'BondgraphModel', uri: str,
                        source: str|rdflib.BNode, target: str|rdflib.BNode, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__model = model
        self.__source = self.__get_port(source, 'bgf:hasSource')
        self.__target = self.__get_port(target, 'bgf:hasTarget')

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

    def __get_port(self, port: str|rdflib.BNode, reln: str) -> str:
    #==============================================================
        if isinstance(port, rdflib.BNode):
            for row in self.__model.source.sparql_query(
                MODEL_BOND_PORTS.replace('%MODEL%', self.__model.uri)
                                .replace('%BOND%', self.uri)
                                .replace('%BOND_RELN%', reln)):
                return f'{row[0]}_{row[1]}'
        return port

#===============================================================================
#===============================================================================

MODEL_JUNCTIONS = f"""
    SELECT DISTINCT ?uri ?type ?label ?flow ?potential
    WHERE {{
        %MODEL% bg:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        OPTIONAL {{
            ?uri bgf:hasFlow [
                bgf:hasValue ?flow
            ]
        }}
        OPTIONAL {{
            ?uri bgf:hasPotential [
                bgf:hasValue ?potential
            ]
        }}
        FILTER (?type IN ({', '.join(FRAMEWORK.junction_classes())}))
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphJunction(Labelled):
    def __init__(self, uri, type: str, label: Optional[str], flow: Optional[rdflib.Literal], potential: Optional[rdflib.Literal]):
        super().__init__(uri, label)
        self.__type = type
        self.__junction = FRAMEWORK.junction(type)
        if self.__junction is None:
            raise ValueError(f'Unknown BondElement {type} for node {uri}')
        self.__num_ports = self.__junction.num_ports
        self.__port_ids: dict[str, str] = {}
        self.__domain = None
        self.__variables: dict[str, Variable] = {}
        if flow is not None:
            if type != ONENODE_JUNCTION:
                raise ValueError(f'{self.uri}: can only set Flow for one-nodes')
            self.__flow_value = Value(flow)
        else:
            self.__flow_value = None
        if potential is not None:
            if type != ZERONODE_JUNCTION:
                raise ValueError(f'{self.uri}: can only set Potential for zero-nodes')
            self.__potential_value = Value(potential)
        else:
            self.__potential_value = None

    @property
    def num_ports(self):
        return self.__num_ports

    @property
    def type(self):
        return self.__type

    @property
    def variables(self):
        return self.__variables

    def assign_ports_and_domain(self, bond_graph: nx.DiGraph):
    #=========================================================
        node_id = self.uri
        attributes = bond_graph.nodes[node_id]
        if bond_graph.degree[node_id] > 1:   # type: ignore
            n = 0
            for edge in bond_graph.in_edges(node_id):
                port_id = f'{node_id}_{n}'
                self.__port_ids[port_id] = '+'
                bond_graph.add_node(port_id, **attributes)
                bond_graph.add_edge(edge[0], port_id)
                n += 1
            for edge in bond_graph.out_edges(node_id):
                port_id = f'{node_id}_{n}'
                self.__port_ids[port_id] = '-'
                bond_graph.add_node(port_id, **attributes)
                bond_graph.add_edge(port_id, edge[1])
                n += 1
            bond_graph.remove_node(node_id)
        elif len(bond_graph.in_edges(node_id)):
            self.__port_ids[node_id] = '+'
        elif len(bond_graph.out_edges(node_id)):
            self.__port_ids[node_id] = '-'
        self.__set_domain(attributes['domain'])

    def assign_relations(self, bond_graph: nx.DiGraph):
    #==================================================
        undirected_graph = bond_graph.to_undirected(as_view=True)
        for node_id in self.__port_ids.keys():
            adjacent_node = list(undirected_graph[node_id])[0]
            adjacent_port = undirected_graph.nodes[adjacent_node]['port']

    def __set_domain(self, domain: Domain):
    #======================================
        self.__domain = domain


#===============================================================================
#===============================================================================

"""
:u_0
    a bgf:ZeroNode ;    ## 0-nodes can have a potential value; 1-nodes can have a flow value
    bgf:hasVariable [  ## has to be compatible with assigned domain....
        bgf:hasValue "11 J/coulomb"^^cdt:ucum
    ] .

The terminals of a JS network are the BEs it connects to and these
determine possible potential (u) and flow (v) symbols for JS nodes.

For each JS subgraph/network (reactions will divide JS network):
    Build flow and potential matrices to determine their equations.
        This will include transform nodes (Tf and Gy).


Each BE gets specific symbols for its parameter, state, and powerport
variables (and constants, when the same symbol has different values).
    ==> constants' registry (node, symbol, value)


:R_C_R_circuit
[[':C_1', 'bgf:ElectricalCapacitor'],
 [':R_0', 'bgf:ElectricalResistor'],
 [':R_1', 'bgf:ElectricalResistor']]
[[':u_0', 'bgf:ZeroNode', 'bgf:Electrical'],
 [':u_1', 'bgf:ZeroNode', 'bgf:Electrical'],
 [':v_0', 'bgf:OneNode', 'bgf:Electrical']]


[[':u_0.v_0', ':u_0', ':v_0'],
 [':u_1.C_1', ':u_1', ':C_1'],
 [':u_1.R_1', ':u_1', ':R_1'],
 [':v_0.R_0', ':v_0', ':R_0'],
 [':v_0.u_1', ':v_0', ':u_1']]
"""

#===============================================================================
#===============================================================================

BONDGRAPH_MODELS = f"""
    SELECT DISTINCT ?uri ?label
    WHERE {{
        ?uri a bg:BondGraph .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphModel(Labelled):
    def __init__(self, source: 'BondgraphModelSource', uri: str, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__source = source
        self.__elements = [BondgraphElement.for_model(self, *row)
                                for row in source.sparql_query(MODEL_ELEMENTS.replace('%MODEL%', uri))]
        self.__junctions = [BondgraphJunction(*row)
                                for row in source.sparql_query(MODEL_JUNCTIONS.replace('%MODEL%', uri))]
        self.__bonds = [BondgraphBond(self, *row)
                            for row in source.sparql_query(MODEL_BONDS.replace('%MODEL%', uri))]
        self.__make_bond_network()
        self.__assign_domains()
        self.__assign_junction_ports()
        self.__assign_junction_relations()

        # Check domain consistency and identify gyrators
        self.__check_domains()


    @property
    def source(self):
        return self.__source

    # Assign junction domains from elements and check consistency
    def __assign_domains(self):
    #=========================
        for element in self.__elements:
            element.assign_domains(self.__graph)

    # Add individual ports to junction nodes (only for 0 and 1 nodes)
    def __assign_junction_ports(self):
    #=================================
        for junction in self.__junctions:
            junction.assign_ports_and_domain(self.__graph)
        for node, degree in self.__graph.degree:
            if degree != 1:
                raise ValueError(f'{self.uri} is not properly connected (node {node})')

    def __assign_junction_relations(self):
    #=====================================
        for junction in self.__junctions:
            junction.assign_relations(self.__graph)

    # Construct network graph of PowerBonds
    def __make_bond_network(self):
    #=============================
        self.__graph = nx.DiGraph()
        for element in self.__elements:
            for port_id, port in element.ports.items():
                self.__graph.add_node(port_id, type=element.type, port=port)
        for junction in self.__junctions:
            self.__graph.add_node(junction.uri, type=junction.type, node=junction)
        for bond in self.__bonds:
            if (bond_source := bond.source) not in self.__graph:
                raise ValueError(f'No element or junction for source {bond_source} of bond {bond.uri}')
            if (bond_target := bond.target) not in self.__graph:
                raise ValueError(f'No element or junction for target {bond_target} of bond {bond.uri}')
            self.__graph.add_edge(bond_source, bond_target)

#===============================================================================
#===============================================================================

class BondgraphModelSource:
    def __init__(self, bondgraph_path: str):
        self.__namespace_map = NamespaceMap(NAMESPACES)
        self.__namespace_map.add_namespace('', f'{Path(bondgraph_path).resolve().as_uri()}#')
        self.__sparql_prefixes = self.__namespace_map.sparql_prefixes()
        self.__rdf = rdflib.Graph()
        self.__rdf.parse(bondgraph_path, format='turtle')
        self.__models = [BondgraphModel(self, *row) for row in self.sparql_query(BONDGRAPH_MODELS)]

    def sparql_query(self, query: str) -> list[list]:
    #================================================
        query_result = self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')
        if query_result is not None:
            return [[self.__namespace_map.simplify(term) for term in row]   # type: ignore
                                                                for row in query_result]
        return []

#===============================================================================
#===============================================================================
