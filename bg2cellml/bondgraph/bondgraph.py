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

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import Labelled, NamespaceMap
from ..mathml import equal, MathML, equal_variables, var_symbol
from ..units import Value

from .framework import BondgraphFramework as FRAMEWORK, Domain, PowerPort, Variable
from .framework import ONENODE_JUNCTION, TRANSFORM_JUNCTION, ZERONODE_JUNCTION

from .namespaces import NAMESPACES

#===============================================================================
#===============================================================================

def make_element_port_id(element_id: str, port_id: str) -> str:
#==============================================================
    return element_id if port_id in [None, ''] else f'{element_id}_{port_id}'

#===============================================================================

ELEMENT_VARIABLES = f"""
    SELECT DISTINCT ?name ?value ?symbol
    WHERE {{
        %ELEMENT_URI% bgf:variableValue ?variable .
        ?variable
            bgf:varName ?name ;
            bgf:hasValue ?value .
        OPTIONAL {{ ?variable bgf:hasSymbol ?symbol }}
    }}"""

#===============================================================================

MODEL_ELEMENTS = f"""
    SELECT DISTINCT ?uri ?element_type ?label ?domain
    WHERE {{
        %MODEL% bg:hasBondElement ?uri .
        ?uri a ?element_type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        OPTIONAL {{ ?uri bgf:hasDomain ?domain }}
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphElement(Labelled):
    def __init__(self, uri: str, element_type: str, label: Optional[str], domain_uri: Optional[str]):
        super().__init__(uri, label)
        element_template = FRAMEWORK.element_template(element_type, domain_uri)
        if element_template is None:
            raise ValueError(f'Cannot find BondElement with type/domain of `{element_type}/{domain_uri}` for node {uri}')
        elif element_template.domain is None:
            raise ValueError(f'No modelling domain for node {uri} with template {element_type}/{domain_uri}')
        elif domain_uri is not None and element_template.domain.uri != domain_uri:
            raise ValueError(f'Domain mismatch for node {uri} with template {element_type}/{domain_uri}')
        elif element_template.constitutive_relation is None:
            raise ValueError(f'Template {element_template.uri} for node {uri} has no constitutive relation')
        self.__constitutive_relation = element_template.constitutive_relation.copy()
        self.__domain = element_template.domain
        self.__type = element_template.uri
        self.__ports = { make_element_port_id(self.uri, port_id): port.copy(self.uri)
                            for port_id, port in element_template.ports.items() }
        self.__variables = {name: variable.copy(self.uri)
                                for name, variable in element_template.variables.items()}
        for port in self.__ports.values():
            self.__variables[port.flow.name] = port.flow.variable
            self.__variables[port.potential.name] = port.potential.variable

    @classmethod
    def for_model(cls, model: 'BondgraphModel', *args):
        self = cls(*args)
        for row in model.source.sparql_query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri)):
            if row[0] not in self.__variables:
                raise ValueError(f'Element {self.uri} has unknown name {row[0]} for {self.__type}')     # type: ignore
            self.__variables[row[0]].set_value(Value.from_literal(row[1]))
            if row[2] is not None:
                self.__variables[row[0]].set_symbol(row[2])
        return self

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def ports(self) -> dict[str, PowerPort]:
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

    # Substitute variable symbols into the constitutive relation
    def substitute_variable_names(self):
    #=====================================
        for name, variable in self.__variables.items():
            self.__constitutive_relation.substitute(name, variable.symbol)

#===============================================================================
#===============================================================================

MODEL_BONDS = f"""
    SELECT DISTINCT ?uri ?source ?target ?label
    WHERE {{
        %MODEL% bg:hasPowerBond ?uri .
        OPTIONAL {{ ?uri bgf:hasSource ?source }}
        OPTIONAL {{ ?uri bgf:hasTarget ?target }}
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
        self.__source_id = self.__get_port(source, 'bgf:hasSource')
        self.__target_id = self.__get_port(target, 'bgf:hasTarget')

    @property
    def source_id(self) -> str:
        return self.__source_id

    @property
    def target_id(self) -> str:
        return self.__target_id

    def __get_port(self, port: str|rdflib.BNode, reln: str) -> str:
    #==============================================================
        if isinstance(port, rdflib.BNode):
            for row in self.__model.source.sparql_query(
                MODEL_BOND_PORTS.replace('%MODEL%', self.__model.uri)
                                .replace('%BOND%', self.uri)
                                .replace('%BOND_RELN%', reln)):
                return make_element_port_id(*row)
        return make_element_port_id(port, '')

#===============================================================================
#===============================================================================

MODEL_JUNCTIONS = f"""
    SELECT DISTINCT ?uri ?type ?label ?value
    WHERE {{
        %MODEL% bg:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        OPTIONAL {{ ?uri bgf:hasValue ?value }}
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphJunction(Labelled):
    def __init__(self, uri, type: str, label: Optional[str], value: Optional[rdflib.Literal]):
        super().__init__(uri, label)
        self.__type = type
        self.__junction = FRAMEWORK.junction(type)
        if self.__junction is None:
            raise ValueError(f'Unknown Junction {type} for node {uri}')
        self.__constitutive_relation = None
        self.__domain = None
        self.__value = value
        self.__variables: list[Variable] = []

    @property
    def constitutive_relation(self) -> MathML:
    #=========================================
        return self.__constitutive_relation         # type: ignore

    @property
    def type(self) -> str:
        return self.__type

    @property
    def variables(self) -> list[Variable]:
        return self.__variables

    def assign_domain(self, bond_graph: nx.DiGraph):
    #===============================================
        node_id = self.uri
        attributes = bond_graph.nodes[node_id]
        self.__domain = attributes['domain']
        if self.__type == ONENODE_JUNCTION:
            self.__variables = [Variable(node_id, node_id, self.__domain.flow.units, self.__value)]
        elif self.__type == ZERONODE_JUNCTION:
            self.__variables = [Variable(node_id, node_id, self.__domain.potential.units, self.__value)]
        elif self.__type == TRANSFORM_JUNCTION:
            raise ValueError(f'Transform Nodes ({self.uri}) are not yet supported')
            ## each port needs a domain, if gyrator different domains...

    def assign_relation(self, bond_graph: nx.DiGraph):
    #=================================================
        node_id = self.uri
        if bond_graph.degree[node_id] > 1:   # type: ignore
            # we are connected to several nodes
            if self.__type == ONENODE_JUNCTION:
                # Sum of potentials connected to junction is 0
                inputs = [self.__potential_symbol(bond_graph.nodes[edge[0]])
                            for edge in bond_graph.in_edges(node_id)]
                outputs = [self.__potential_symbol(bond_graph.nodes[edge[1]])
                            for edge in bond_graph.out_edges(node_id)]
                equal_value = [node_dict['port'].flow.variable.symbol
                                for edge in bond_graph.in_edges(node_id)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[0]])]
                equal_value.extend([node_dict['port'].flow.variable.symbol
                                for edge in bond_graph.out_edges(node_id)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[1]])])
            elif self.__type == ZERONODE_JUNCTION:
                # Sum of flows connected to junction is 0
                inputs = [self.__flow_symbol(bond_graph.nodes[edge[0]])
                            for edge in bond_graph.in_edges(node_id)]
                outputs = [self.__flow_symbol(bond_graph.nodes[edge[1]])
                            for edge in bond_graph.out_edges(node_id)]
                equal_value = [node_dict['port'].potential.variable.symbol
                                for edge in bond_graph.in_edges(node_id)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[0]])]
                equal_value.extend([node_dict['port'].potential.variable.symbol
                                for edge in bond_graph.out_edges(node_id)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[1]])])
            else:
                raise ValueError(f'Unexpected bond graph node for {self.uri}: {self.__type}')
            equal_value = '\n'.join([equal(var_symbol(self.__variables[0].symbol), var_symbol(symbol))
                                                                                    for symbol in equal_value])
            self.__constitutive_relation = MathML.from_string(f'''<math xmlns="http://www.w3.org/1998/Math/MathML">
                {equal_variables(inputs, outputs)}
                {equal_value}
</math>''')

    def __flow_symbol(self, node_dict: dict) -> str:
    #===============================================
        if 'port' in node_dict:                         # A BondElement's port
            port: PowerPort = node_dict['port']
            return port.flow.variable.symbol
        elif 'junction' in node_dict:
            junction: BondgraphJunction = node_dict['junction']
            if junction.type == ONENODE_JUNCTION:
                return junction.variables[0].symbol     # type: ignore
            elif junction.type == ZERONODE_JUNCTION:
                raise ValueError(f'Adjacent Zero Nodes, {self.uri} and {junction.uri}, must be merged')
            elif junction.type == TRANSFORM_JUNCTION:
                raise ValueError('Transform Nodes are not yet supported')
        raise ValueError(f'Unexpected bond graph node connected to {self.uri}: {node_dict}')

    def __potential_symbol(self, node_dict: dict) -> str:
    #====================================================
        if 'port' in node_dict:                         # A BondElement's port
            port: PowerPort = node_dict['port']
            return port.potential.variable.symbol
        elif 'junction' in node_dict:
            junction: BondgraphJunction = node_dict['junction']
            if junction.type == ZERONODE_JUNCTION:
                return junction.variables[0].symbol     # type: ignore
            elif junction.type == ONENODE_JUNCTION:
                raise ValueError(f'Adjacent One Nodes, {self.uri} and {junction.uri}, must be merged')
            elif junction.type == TRANSFORM_JUNCTION:
                raise ValueError('Transform Nodes are not yet supported')
        raise ValueError(f'Unexpected bond graph node for {self.uri}: {node_dict}')

#===============================================================================
#===============================================================================

BONDGRAPH_MODELS = f"""
    SELECT DISTINCT ?uri ?label
    WHERE {{
        ?uri a bgf:BondgraphModel .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphModel(Labelled):
    def __init__(self, source: 'BondgraphModelSource', uri: str, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__source = source
        self.__elements = [BondgraphElement.for_model(self, *row)
                                for row in source.sparql_query(MODEL_ELEMENTS.replace('%MODEL%', uri))]
        for element in self.__elements:
            element.substitute_variable_names()
        self.__junctions = [BondgraphJunction(*row)
                                for row in source.sparql_query(MODEL_JUNCTIONS.replace('%MODEL%', uri))]
        self.__bonds = [BondgraphBond(self, *row)
                            for row in source.sparql_query(MODEL_BONDS.replace('%MODEL%', uri))]
        self.__make_bond_network()
        self.__check_and_assign_domains_to_bond_network()
        self.__assign_junction_domains_and_relations()

    @property
    def elements(self):
        return self.__elements

    @property
    def junctions(self):
        return self.__junctions

    @property
    def source(self):
        return self.__source

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
                    raise ValueError(f'Node {node} with domain {node_domain} incompatible with {domain}')

        for element in self.__elements:
            for port_id in element.ports.keys():
                check_node(port_id, element.domain)

    def __assign_junction_domains_and_relations(self):
    #=================================================
        for junction in self.__junctions:
            junction.assign_domain(self.__graph)
        for junction in self.__junctions:
            junction.assign_relation(self.__graph)

    # Construct network graph of PowerBonds
    def __make_bond_network(self):
    #=============================
        self.__graph = nx.DiGraph()
        for element in self.__elements:
            for port_id, port in element.ports.items():
                self.__graph.add_node(port_id, type=element.type, port=port)
        for junction in self.__junctions:
            self.__graph.add_node(junction.uri, type=junction.type, junction=junction)
        for bond in self.__bonds:
            if (source := bond.source_id) not in self.__graph:
                raise ValueError(f'No element or junction for source {source} of bond {bond.uri}')
            if (target := bond.target_id) not in self.__graph:
                raise ValueError(f'No element or junction for target {target} of bond {bond.uri}')
            self.__graph.add_edge(source, target)

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

    @property
    def models(self):
        return self.__models

    def sparql_query(self, query: str) -> list[list]:
    #================================================
        query_result = self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')
        if query_result is not None:
            return [[self.__namespace_map.simplify(term) for term in row]   # type: ignore
                                                                for row in query_result]
        return []

#===============================================================================
#===============================================================================
