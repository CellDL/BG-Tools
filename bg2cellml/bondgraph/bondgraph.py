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

import networkx as nx

#===============================================================================

from ..rdf import BNode, Literal, ResultRow, RDFGraph, URIRef
from ..mathml import equal, MathML, equal_variables, var_symbol
from ..units import Value

from .framework import BondgraphFramework as FRAMEWORK, Domain, PowerPort, Variable
from .framework import ONENODE_JUNCTION, TRANSFORM_JUNCTION, ZERONODE_JUNCTION
from .namespaces import NAMESPACES
from .utils import Labelled

#===============================================================================
#===============================================================================

def make_element_port_id(element_uri: URIRef, port_id: str) -> URIRef:
#=====================================================================
    return element_uri if port_id in [None, ''] else element_uri + f'_{port_id}'

#===============================================================================

class ModelElement(Labelled):
    def __init__(self,  model: 'BondgraphModel', uri: URIRef, label: Optional[str]):
        super().__init__(uri, label)
        self.__model = model

    @property
    def model(self):
        return self.__model

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = """
    SELECT DISTINCT ?name ?value ?symbol
    WHERE {
        <%ELEMENT_URI%> bgf:variableValue ?variable .
        ?variable
            bgf:varName ?name ;
            bgf:hasValue ?value .
        OPTIONAL { ?variable bgf:hasSymbol ?symbol }
    }"""

ELEMENT_STATE_VALUE = """
    SELECT DISTINCT ?quantityValue
    WHERE {
        <%ELEMENT_URI%> bgf:quantityValue ?quantityValue .
    }"""

#===============================================================================

type VariableValue = tuple[Literal|URIRef, Optional[str]]

#===============================================================================

class BondgraphElement(ModelElement):
    def __init__(self,  model: 'BondgraphModel', uri: URIRef, element_type: URIRef,
                        label: Optional[str], domain_uri: Optional[URIRef],
                        intrinsic_value: Optional[Value] = None,
                        variable_values: Optional[dict[str, VariableValue]]=None):
        super().__init__(model, uri, label)
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
        self.__ports = { make_element_port_id(self.uri, port_id): port.copy(self.uri.fragment)
                            for port_id, port in element_template.ports.items() }
        self.__variables = {name: variable.copy(self.uri.fragment)
                                for name, variable in element_template.variables.items()}
        if (intrinsic_var := element_template.intrinsic_variable) is not None:
            self.__variables[intrinsic_var.name] = intrinsic_var.copy(self.uri.fragment, True)
            self.__intrinsic_variable = self.__variables[intrinsic_var.name]
            if intrinsic_value is not None:
                self.__intrinsic_variable.set_value(intrinsic_value)
        else:
            self.__intrinsic_variable = None

        # Defer assigning until we have the full bondgraph
        self.__variable_values = variable_values if variable_values is not None else {}

        for port in self.__ports.values():
            self.__variables[port.flow.name] = port.flow.variable
            self.__variables[port.potential.name] = port.potential.variable

    @classmethod
    def for_model(cls, model: 'BondgraphModel', uri: URIRef, element_type: URIRef,
                       label: Optional[str], domain_uri: Optional[URIRef]):
        variable_values: dict[str, VariableValue] = {str(row[0]): (row[1], row[2])  # type: ignore
            for row in model.sparql_query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', uri))
        }
        intrinsic_value: Optional[Value] = None
        for row in model.sparql_query(ELEMENT_STATE_VALUE.replace('%ELEMENT_URI%', uri)):
            intrinsic_value = Value.from_literal(row[0])                            # type: ignore
            break
        return cls(model, uri, element_type, label, domain_uri,
                    intrinsic_value=intrinsic_value, variable_values=variable_values)

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def ports(self) -> dict[URIRef, PowerPort]:
        return self.__ports

    @property
    def constitutive_relation(self):
        return self.__constitutive_relation

    @property
    def type(self) -> URIRef:
        return self.__type

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    # Substitute variable symbols into the constitutive relation
    def assign_variables(self, bond_graph: nx.DiGraph):
    #==================================================
        for var_name, value in self.__variable_values.items():
            if (variable := self.__variables.get(var_name)) is None:
                raise ValueError(f'Element {self.uri} has unknown name {var_name} for {self.__type}')
            if value[1] is not None:
                variable.set_symbol(value[1])
            if isinstance(value[0], Literal):
                variable.set_value(Value.from_literal(value[0]))
            elif isinstance(value[0], URIRef):
                if value[0] not in bond_graph:
                    raise ValueError(f'Value for {self.uri} refers to unknown element: {value[0]}')
                elif (element := bond_graph.nodes[value[0]].get('element')) is None:
                    raise ValueError(f'Value for {self.uri} is not a bond element: {value[0]}')
                elif element.__intrinsic_variable is None:
                    raise ValueError(f'Value for {self.uri} is an element with no intrinsic variable: {value[0]}')
                elif variable.units != element.__intrinsic_variable.units:
                    raise ValueError(f'Units incompatible for {self.uri} value: {value[0]}')
                else:
                    variable.set_symbol(element.__intrinsic_variable.symbol)
        for name, variable in self.__variables.items():
            self.__constitutive_relation.substitute(name, variable.symbol)

#===============================================================================
#===============================================================================

MODEL_BOND_PORTS = """
    SELECT DISTINCT ?element ?port
    WHERE {
        { <%MODEL%> bgf:hasPowerBond <%BOND%> }
  UNION { <%MODEL%> bgf:hasInterfaceBond <%BOND%> }
        <%BOND%> %BOND_RELN% [
            bgf:element ?element ;
            bgf:port ?port
        ]
    }"""

#===============================================================================

class BondgraphBond(ModelElement):
    def __init__(self, model: 'BondgraphModel', uri: URIRef,
                        source: URIRef|BNode, target: URIRef|BNode,
                        interface_bond: bool=False, label: Optional[str]=None):
        super().__init__(model, uri, label)
        self.__source_id = self.__get_port(source, 'bgf:hasSource')
        self.__target_id = self.__get_port(target, 'bgf:hasTarget')
        self.__interface_bond = interface_bond

    @property
    def interface_bond(self):
        return self.__interface_bond

    @property
    def source_id(self) -> Optional[URIRef]:
        return self.__source_id

    @property
    def target_id(self) -> Optional[URIRef]:
        return self.__target_id

    def __get_port(self, port: URIRef|BNode, reln: str) -> Optional[URIRef]:
    #=======================================================================
        if isinstance(port, BNode):
            for row in self.model.sparql_query(
                MODEL_BOND_PORTS.replace('%MODEL%', self.model.uri)
                                .replace('%BOND%', self.uri)
                                .replace('%BOND_RELN%', reln)):
                return make_element_port_id(row[0], str(row[1]))    # type: ignore
        else:
            return make_element_port_id(port, '')

#===============================================================================
#===============================================================================

class BondgraphJunction(ModelElement):
    def __init__(self, model: 'BondgraphModel', uri: URIRef, type: URIRef, label: Optional[str], value: Optional[Literal]):
        super().__init__(model, uri, label)
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
    def type(self) -> URIRef:
        return self.__type

    @property
    def variables(self) -> list[Variable]:
        return self.__variables

    def assign_domain(self, bond_graph: nx.DiGraph):
    #===============================================
        node_uri = self.uri
        attributes = bond_graph.nodes[node_uri]
        if (domain := attributes.get('domain')) is None:
            raise ValueError(f'Cannot find domain for junction {self.uri}. Are there bonds to it?')
        self.__domain = domain
        if self.__type == ONENODE_JUNCTION:
            self.__variables = [Variable(node_uri, node_uri.fragment, self.__domain.flow.units, self.__value)]
        elif self.__type == ZERONODE_JUNCTION:
            self.__variables = [Variable(node_uri, node_uri.fragment, self.__domain.potential.units, self.__value)]
        elif self.__type == TRANSFORM_JUNCTION:
            raise ValueError(f'Transform Nodes ({self.uri}) are not yet supported')
            ## each port needs a domain, if gyrator different domains...

    def assign_relation(self, bond_graph: nx.DiGraph):
    #=================================================
        node_uri = self.uri
        if bond_graph.degree[node_uri] > 1:   # type: ignore
            # we are connected to several nodes
            if self.__type == ONENODE_JUNCTION:
                # Sum of potentials connected to junction is 0
                inputs = [self.__potential_symbol(bond_graph.nodes[edge[0]])
                            for edge in bond_graph.in_edges(node_uri)]
                outputs = [self.__potential_symbol(bond_graph.nodes[edge[1]])
                            for edge in bond_graph.out_edges(node_uri)]
                equal_value = [node_dict['port'].flow.variable.symbol
                                for edge in bond_graph.in_edges(node_uri)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[0]])]
                equal_value.extend([node_dict['port'].flow.variable.symbol
                                for edge in bond_graph.out_edges(node_uri)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[1]])])
            elif self.__type == ZERONODE_JUNCTION:
                # Sum of flows connected to junction is 0
                inputs = [self.__flow_symbol(bond_graph.nodes[edge[0]])
                            for edge in bond_graph.in_edges(node_uri)]
                outputs = [self.__flow_symbol(bond_graph.nodes[edge[1]])
                            for edge in bond_graph.out_edges(node_uri)]
                equal_value = [node_dict['port'].potential.variable.symbol
                                for edge in bond_graph.in_edges(node_uri)
                                    if 'port' in (node_dict := bond_graph.nodes[edge[0]])]
                equal_value.extend([node_dict['port'].potential.variable.symbol
                                for edge in bond_graph.out_edges(node_uri)
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

MODEL_ELEMENTS = """
    SELECT DISTINCT ?model ?uri ?type ?label ?domain
    WHERE {
        ?model bgf:hasBondElement ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri rdfs:label ?label }
        OPTIONAL { ?uri bgf:hasDomain ?domain }
    } ORDER BY ?model ?uri"""

MODEL_JUNCTIONS = """
    SELECT DISTINCT ?model ?uri ?type ?label ?value
    WHERE {
        ?model bgf:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri rdfs:label ?label }
        OPTIONAL { ?uri bgf:hasValue ?value }
    } ORDER BY ?model ?uri"""

MODEL_BONDS = """
    SELECT DISTINCT ?model ?powerBond ?interfaceBond ?source ?target ?label
    WHERE {
        {
            ?model bgf:hasPowerBond ?powerBond .
            OPTIONAL { ?powerBond bgf:hasSource ?source }
            OPTIONAL { ?powerBond bgf:hasTarget ?target }
            OPTIONAL { ?powerBond rdfs:label ?label }
        } UNION {
            ?model bgf:hasInterfaceBond ?interfaceBond .
            OPTIONAL { ?interfaceBond bgf:hasSource ?source }
            OPTIONAL { ?interfaceBond bgf:hasTarget ?target }
            OPTIONAL { ?interfaceBond rdfs:label ?label }
        }
    } ORDER BY ?model"""

BONDGRAPH_MODELS = """
    SELECT DISTINCT ?uri ?label
    WHERE {
        ?uri a bgf:BondgraphModel .
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

#===============================================================================

class BondgraphModel(Labelled):
    def __init__(self, rdf_graph: RDFGraph, uri: URIRef, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__rdf_graph = rdf_graph
        self.__elements = [BondgraphElement.for_model(self, row[1], row[2], row[3], row[4]) # type: ignore
                                for row in rdf_graph.query(MODEL_ELEMENTS)]
        self.__junctions = [BondgraphJunction(self, row[1], row[2], row[3], row[4])         # type: ignore
                                for row in rdf_graph.query(MODEL_JUNCTIONS)]
        self.__bonds = []
        for row in rdf_graph.query(MODEL_BONDS):
            uri = row[1] if row[1] is not None else row[2]      # type: ignore
            if row[1] is not None and row[2] is not None:
                raise ValueError(f'Bond {uri} cannot be both a power and an interface bond')
            elif row[3] is None or row[4] is None:
                raise ValueError(f'Bond {uri} is missing source and/or target node')
            self.__bonds.append(
                BondgraphBond(self, uri, row[3], row[4], row[2] is not None, row[5]))       # type: ignore
        self.__graph = nx.DiGraph()
        self.__make_bond_network()
        self.__assign_element_variables()
        self.__check_and_assign_domains_to_bond_network()
        self.__assign_junction_domains_and_relations()

    @property
    def elements(self):
        return self.__elements

    @property
    def junctions(self):
        return self.__junctions

    @property
    def network_graph(self) -> nx.DiGraph:
        return self.__graph.copy()

    def __assign_element_variables(self):
    #====================================
        for element in self.__elements:
            element.assign_variables(self.__graph)

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
        for element in self.__elements:
            for port_id, port in element.ports.items():
                self.__graph.add_node(port_id,
                    type=self.__rdf_graph.curie(element.type), port=port, element=element, label=element.uri.fragment)
        for junction in self.__junctions:
            self.__graph.add_node(junction.uri,
                type=self.__rdf_graph.curie(junction.type), junction=junction, label=junction.uri.fragment)
        for bond in self.__bonds:
            source = bond.source_id
            target = bond.target_id
            if source not in self.__graph and target not in self.__graph:
                raise ValueError(f'No element or junction for source and target of bond {bond.uri}')
            elif not bond.interface_bond:
                if source not in self.__graph or target not in self.__graph:
                    raise ValueError(f'No element or junction for source or target of bond {bond.uri}')
                self.__graph.add_edge(source, target)
            elif source in self.__graph and target in self.__graph:
                self.__graph.add_edge(source, target)

    def sparql_query(self, query: str) -> list[ResultRow]:
    #=====================================================
        return self.__rdf_graph.query(query)

#===============================================================================
#===============================================================================

BONDGRAPH_MODELS = """
    SELECT DISTINCT ?uri ?label
    WHERE {
        ?uri a bgf:BondgraphModel .
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

#===============================================================================

BONDGRAPH_MODEL_BLOCKS = """
    SELECT DISTINCT ?blockSource
    WHERE {
        ?uri
            a bgf:BondgraphModel ;
            bgf:hasBlock ?blockSource .
    }"""

#===============================================================================

class BondgraphModelSource:
    def __init__(self, source: str):
        self.__rdf_graph = RDFGraph(NAMESPACES)
        self.__source_path = Path(source).resolve()
        self.__loaded_sources: set[Path] = set()
        self.__load_rdf(self.__source_path)
        base_models: list[tuple[URIRef, Optional[Literal]]] = [(row[0], row[1])     # type: ignore
            for row in self.__rdf_graph.query(BONDGRAPH_MODELS)]
        if len(base_models) < 1:
            raise ValueError(f'No BondgraphModels in source {source}')
        self.__load_blocks(self.__source_path)
        FRAMEWORK.resolve_composites(base_models[0][0], self.__rdf_graph)
        FRAMEWORK.generate_bonds(base_models[0][0], self.__rdf_graph)
        self.__models = { uri: BondgraphModel(self.__rdf_graph, uri, label)
                                for (uri, label) in base_models }

    def __load_blocks(self, base_path: Path):
    #========================================
        for row in self.__rdf_graph.query(BONDGRAPH_MODEL_BLOCKS):
            path = base_path.parent.joinpath(str(row[0])).resolve()
            self.__load_source(path)

    def __load_rdf(self, source_path: Path):
    #=======================================
        self.__rdf_graph.parse(source_path.as_uri())
        self.__loaded_sources.add(source_path)

    def __load_source(self, source_path: Path):
    #==========================================
        if source_path not in self.__loaded_sources:
            self.__load_rdf(source_path)
            self.__load_blocks(source_path)

    @property
    def models(self):
        return list(self.__models.values())

#===============================================================================
#===============================================================================
