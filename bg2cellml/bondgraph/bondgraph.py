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
from typing import cast, Optional
from urllib.parse import urldefrag

#===============================================================================

import networkx as nx
import sympy

#===============================================================================

from ..rdf import BNode, Literal, ResultRow, ResultType, RDFGraph, URIRef
from ..mathml import Equation, MathML
from ..units import Value
from ..utils import bright, log, pretty_log, pretty_uri

from .framework import BondgraphFramework as FRAMEWORK, BondgraphElementTemplate, CompositeTemplate
from .framework import Domain, PowerPort, Variable
from .framework import ONENODE_JUNCTION, TRANSFORM_JUNCTION, ZERONODE_JUNCTION
from .framework import FLOW_SOURCE, POTENTIAL_SOURCE
from .framework import DISSIPATOR, KINETIC_STORE, QUANTITY_STORE
from .namespaces import BGF, NAMESPACES
from .utils import Labelled

#===============================================================================

def make_element_port_id(element_uri: URIRef, port_id: str) -> URIRef:
#=====================================================================
    return element_uri if port_id in [None, ''] else element_uri + f'_{port_id}'

def flow_expression(node_dict: dict) -> Optional[sympy.Expr]:
#============================================================
    if ('port' in node_dict
    and (expr := node_dict['element'].flow_expression) is not None):
        return expr
    return flow_symbol(node_dict)

def flow_symbol(node_dict: dict) -> Optional[sympy.Symbol]:
#==========================================================
    if 'port' in node_dict:                         # A BondElement's port
        return sympy.Symbol(node_dict['port'].flow.variable.symbol)
    elif 'junction' in node_dict:
        junction: BondgraphJunction = node_dict['junction']
        if junction.type == ONENODE_JUNCTION:
            return sympy.Symbol(junction.variables[0].symbol)
        elif junction.type == ZERONODE_JUNCTION:
            log.error(f'Adjacent Zero Nodes to junction {junction.uri} must be merged')
        elif junction.type == TRANSFORM_JUNCTION:
            log.error('Transform Nodes are not yet supported')
    log.error(f'Unexpected bond graph node, cannot get flow: {node_dict}')

def potential_expression(node_dict: dict) -> Optional[sympy.Expr]:
#=================================================================
    if ('port' in node_dict
    and (expr := node_dict['element'].potential_expression) is not None):
        return expr
    return potential_symbol(node_dict)

def potential_symbol(node_dict: dict) -> Optional[sympy.Symbol]:
#===============================================================
    if 'port' in node_dict:                         # A BondElement's port
        return sympy.Symbol(node_dict['port'].potential.variable.symbol)
    elif 'junction' in node_dict:
        junction: BondgraphJunction = node_dict['junction']
        if junction.type == ZERONODE_JUNCTION:
            return sympy.Symbol(junction.variables[0].symbol)
        elif junction.type == ONENODE_JUNCTION:
            log.error(f'Adjacent One Nodes to junction {junction.uri} must be merged')
        elif junction.type == TRANSFORM_JUNCTION:
            log.error('Transform Nodes are not yet supported')
    log.error(f'Unexpected bond graph node, cannot get potential: {node_dict}')

#===============================================================================

class ModelElement(Labelled):
    def __init__(self,  model: 'BondgraphModel', uri: URIRef, symbol: Optional[str], label: Optional[str]):
        super().__init__(uri, symbol, label)
        self.__model = model

    @property
    def model(self):
        return self.__model

#===============================================================================
#===============================================================================

ELEMENT_PARAMETER_VALUES = """
    SELECT DISTINCT ?name ?value ?symbol
    WHERE {
        <%ELEMENT%> bgf:parameterValue ?variable .
        ?variable
            bgf:varName ?name ;
            bgf:hasValue ?value .
        OPTIONAL { ?variable bgf:hasSymbol ?symbol }
    }"""

ELEMENT_STATE_VALUE = """
    SELECT DISTINCT ?value
    WHERE {
        <%ELEMENT%> bgf:hasValue ?value .
    }"""

ELEMENT_VARIABLE_VALUES = """
    SELECT DISTINCT ?name ?value ?symbol
    WHERE {
        <%ELEMENT%> bgf:variableValue ?variable .
        ?variable
            bgf:varName ?name ;
            bgf:hasValue ?value .
        OPTIONAL { ?variable bgf:hasSymbol ?symbol }
    }"""

#===============================================================================

type VariableValue = tuple[Literal|URIRef, Optional[str]]

#===============================================================================

class BondgraphElement(ModelElement):
    def __init__(self,  model: 'BondgraphModel', uri: URIRef, template: BondgraphElementTemplate,
                        parameter_values: Optional[dict[str, VariableValue]]=None,
                        variable_values: Optional[dict[str, VariableValue]]=None,
                        domain_uri: Optional[URIRef]=None, intrinsic_value: Optional[Value]=None,
                        symbol: Optional[str]=None, label: Optional[str]=None):
        super().__init__(model, uri, symbol, label)
        element_type = pretty_uri(template.uri)
        if isinstance(template, CompositeTemplate):
            element_template = template.template
            composite = True
        else:
            element_template = template
            composite = False
        if element_template.domain is None:
            raise ValueError(f'No modelling domain for element {uri} with template {element_type}/{domain_uri}')
        elif domain_uri is not None and element_template.domain.uri != domain_uri:
            raise ValueError(f'Domain mismatch for element {uri} with template {element_type}/{domain_uri}')

        self.__element_class = element_template.element_class
        if self.__element_class in [FLOW_SOURCE, POTENTIAL_SOURCE]:
            self.__constitutive_relation = None
        else:
            if element_template.constitutive_relation is None:
                raise ValueError(f'Template {element_template.uri} for element {uri} has no constitutive relation')
            self.__constitutive_relation = element_template.constitutive_relation.copy()

        self.__domain = element_template.domain
        self.__type = element_template.uri

        self.__ports: dict[URIRef, PowerPort] = {}
        for port_id, port in element_template.ports.items():
            self.__ports[make_element_port_id(self.uri, port_id)] = port.copy(suffix=self.symbol, domain=self.__domain)

        self.__flow_variable = None
        self.__potential_variable = None
        self.__flow_expression = None
        self.__potential_expression = None
        self.__equations = []

        self.__implied_junction = None
        if composite:
            if self.__element_class == QUANTITY_STORE:
                self.__implied_junction = ZERONODE_JUNCTION
            elif self.__element_class == DISSIPATOR:
                self.__implied_junction = ONENODE_JUNCTION
        elif self.__element_class == POTENTIAL_SOURCE:
            self.__implied_junction = ZERONODE_JUNCTION
        elif self.__element_class == FLOW_SOURCE:
            self.__implied_junction = ONENODE_JUNCTION

        self.__variables = {}
        self.__port_variable_names = set()
        for port in self.__ports.values():
            # A port, by definition, always has flow and potential??
            # Then for a composite storage element, element_flow = -sum(connected_flows)
            #                "   dissapator   "        potential = -sum(connected_potentials)
            #
            self.__variables[port.flow.name] = port.flow.variable
            self.__port_variable_names.add(port.flow.name)
            self.__variables[port.potential.name] = port.potential.variable
            self.__port_variable_names.add(port.potential.name)
        self.__variables.update({name: variable.copy(suffix=self.symbol, domain=self.__domain)
                                for name, variable in element_template.parameters.items()})
        self.__variables.update({name: variable.copy(suffix=self.symbol, domain=self.__domain)
                                for name, variable in element_template.variables.items()})

        # Defer assignment until we have the full bondgraph
        self.__variable_values = {}
        if parameter_values is None:
            if len(element_template.parameters):
                log.error(f'No parameters given for element {pretty_uri(uri)}')
        else:
            for name in element_template.parameters.keys():
                if name not in parameter_values:
                    log.error(f'Missing value for parameter {name} of element {pretty_uri(uri)}')
            self.__variable_values.update(parameter_values)
        if variable_values is not None:
            for name in variable_values.keys():
                if name not in name not in self.__variables:
                    log.error(f'Unknown variable {name} for element {pretty_uri(uri)}')
            self.__variable_values.update(variable_values)

        if (intrinsic_var := element_template.intrinsic_variable) is not None:
            self.__variables[intrinsic_var.name] = intrinsic_var.copy(suffix=self.symbol, domain=self.__domain)
            self.__intrinsic_variable = self.__variables[intrinsic_var.name]
            if self.__element_class == FLOW_SOURCE:
                port_var = self.__ports[self.uri].flow
                port_var.variable = self.__intrinsic_variable
            elif self.__element_class == POTENTIAL_SOURCE:
                port_var = self.__ports[self.uri].potential
                port_var.variable = self.__intrinsic_variable
            if intrinsic_value is not None:
                self.__intrinsic_variable.set_value(intrinsic_value)
        else:
            self.__intrinsic_variable = None

    @classmethod
    def for_model(cls, model: 'BondgraphModel', uri: URIRef, template: BondgraphElementTemplate,
    #===========================================================================================
                                    domain_uri: Optional[URIRef], symbol: Optional[str], label: Optional[str]):
        parameter_values: dict[str, VariableValue] = {str(row[0]): (row[1], row[2])  # type: ignore
            for row in model.sparql_query(ELEMENT_PARAMETER_VALUES.replace('%ELEMENT%', uri))
        }
        variable_values: dict[str, VariableValue] = {str(row[0]): (row[1], row[2])  # type: ignore
            for row in model.sparql_query(ELEMENT_VARIABLE_VALUES.replace('%ELEMENT%', uri))
        }
        intrinsic_value: Optional[Value] = None
        for row in model.sparql_query(ELEMENT_STATE_VALUE.replace('%ELEMENT%', uri)):
            intrinsic_value = Value.from_literal(row[0])                            # type: ignore
            break
        return cls(model, uri, template, domain_uri=domain_uri,
                    parameter_values=parameter_values, variable_values=variable_values,
                    intrinsic_value=intrinsic_value, symbol=symbol, label=label)

    @property
    def constitutive_relation(self) -> Optional[MathML]:
        return self.__constitutive_relation

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def element_class(self) -> URIRef:
        return self.__element_class

    @property
    def flow_expression(self) -> Optional[sympy.Basic]:
        return self.__flow_expression

    @property
    def equations(self) -> list[Equation]:
        return self.__equations

    @property
    def ports(self) -> dict[URIRef, PowerPort]:
        return self.__ports

    @property
    def potential_expression(self) -> Optional[sympy.Basic]:
        return self.__potential_expression

    @property
    def type(self) -> URIRef:
        return self.__type

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    # Substitute variable symbols into the constitutive relation
    def assign_variables(self, bond_graph: nx.DiGraph):
    #==================================================
        ## Remove variables associated with any unconnected ports
        unused_ports = []
        for port_id, port in self.__ports.items():
            if bond_graph.degree(port_id) == 0:
                unused_ports.append(port_id)
                if self.__element_class == DISSIPATOR:
                    del self.__variables[port.potential.name]
                    self.__port_variable_names.remove(port.potential.name)
        for port_id in unused_ports:
            del self.__ports[port_id]

        for var_name, value in self.__variable_values.items():
            if (variable := self.__variables.get(var_name)) is None:
                log.error(f'Element {pretty_uri(self.uri)} has unknown name {var_name} for {self.__type}')
                continue
            if value[1] is not None:
                variable.set_symbol(value[1])
            if isinstance(value[0], Literal):
                variable.set_value(Value.from_literal(value[0]))
            elif isinstance(value[0], URIRef):
                if value[0] not in bond_graph:
                    log.error(f'Value for {pretty_uri(self.uri)} refers to unknown element: {value[0]}')
                    continue
                elif (element := bond_graph.nodes[value[0]].get('element')) is None:
                    log.error(f'Value for {pretty_uri(self.uri)} is not a bond element: {value[0]}')
                    continue
                elif element.__intrinsic_variable is None:
                    log.error(f'Value for {pretty_uri(self.uri)} is an element with no intrinsic variable: {value[0]}')
                    continue
                elif variable.units != element.__intrinsic_variable.units:
                    log.error(f'Units incompatible for {pretty_uri(self.uri)} value: {value[0]}')
                    continue
                else:
                    self.__variables[var_name] = element.__intrinsic_variable

        if (variable := self.__variables.get(self.__domain.flow.symbol)) is not None:
            self.__flow_variable = sympy.Symbol(variable.symbol)
        if (variable := self.__variables.get(self.__domain.potential.symbol)) is not None:
            self.__potential_variable = sympy.Symbol(variable.symbol)
        if self.__constitutive_relation is not None:
            for name, variable in self.__variables.items():
                self.__constitutive_relation.substitute(name, variable.symbol,
                                                        missing_ok=(name in self.__port_variable_names))
            for eqn in self.__constitutive_relation.equations:
                if eqn.lhs == self.__flow_variable:
                    self.__flow_expression = eqn.rhs
                elif eqn.lhs == self.__potential_variable:
                    self.__potential_expression = eqn.rhs

    def build_expressions(self, bond_graph: nx.DiGraph):
    #===================================================
        for port_id, port in self.__ports.items():
            flow_expr = None
            potential_expr = None
            if self.__implied_junction == ZERONODE_JUNCTION:
                # Sum of flows connected to junction is 0
                inputs = [expr for node in bond_graph.predecessors(port_id)
                            if (expr := flow_expression(bond_graph.nodes[node])) is not None]
                outputs = [expr for node in bond_graph.successors(port_id)
                            if (expr := flow_expression(bond_graph.nodes[node])) is not None]
                flow_expr = sympy.Add(*inputs, sympy.Mul(-1, sympy.Add(*outputs)))   ## flow_expression
                if self.__flow_variable is not None:
                    self.__equations.append(Equation(self.__flow_variable, flow_expr))
                if len(self.__ports) > 1 and self.__element_class == QUANTITY_STORE:
                    # Two port capacitor...
                    flow = sympy.Symbol(port.flow.variable.symbol)
                    for node in bond_graph.predecessors(port_id):
                        if (symbol := flow_symbol(bond_graph.nodes[node])) is not None:
                            self.__equations.append(Equation(flow, symbol))
                    for node in bond_graph.successors(port_id):
                        if (symbol := flow_symbol(bond_graph.nodes[node])) is not None:
                            self.__equations.append(Equation(flow, symbol))
            elif self.__implied_junction == ONENODE_JUNCTION:
                # Sum of potentials connected to junction is 0
                inputs = [expr for node in bond_graph.predecessors(port_id)
                            if (expr := potential_expression(bond_graph.nodes[node])) is not None]
                outputs = [expr for node in bond_graph.successors(port_id)
                            if (expr := potential_expression(bond_graph.nodes[node])) is not None]
                potential_expr = sympy.Add(*inputs, sympy.Mul(-1, sympy.Add(*outputs)))
                if self.__potential_variable is not None:
                    self.__equations.append(Equation(self.__potential_variable, potential_expr))
                if len(self.__ports) > 1 and self.__element_class == DISSIPATOR:
                    # Reaction...
                    potential = sympy.Symbol(port.potential.variable.symbol)
                    for node in bond_graph.predecessors(port_id):
                        if (symbol := potential_symbol(bond_graph.nodes[node])) is not None:
                            self.__equations.append(Equation(potential, symbol))
                    for node in bond_graph.successors(port_id):
                        if (symbol := potential_symbol(bond_graph.nodes[node])) is not None:
                            self.__equations.append(Equation(potential, symbol))
            if self.__constitutive_relation is not None:
                for eqn in self.__constitutive_relation.equations:
                    if eqn.rhs == self.__flow_variable and flow_expr:
                        eqn.rhs = flow_expr
                    elif eqn.lhs == self.__potential_variable and potential_expr:
                        eqn.rhs = potential_expr

#===============================================================================
#===============================================================================

MODEL_BOND_PORTS = """
    SELECT DISTINCT ?element ?port
    WHERE {
        <%MODEL%> bgf:hasPowerBond <%BOND%> .
        <%BOND%> <%BOND_RELN%> [
            bgf:element ?element ;
            bgf:port ?port
        ]
    }"""

#===============================================================================

class BondgraphBond(ModelElement):
    def __init__(self, model: 'BondgraphModel', uri: URIRef,
                        source: URIRef|BNode, target: URIRef|BNode,
                        label: Optional[str]=None):
        super().__init__(model, uri, None, label)
        self.__source_id = self.__get_port(source, BGF.hasSource)
        self.__target_id = self.__get_port(target, BGF.hasTarget)

    @property
    def source_id(self) -> Optional[URIRef]:
        return self.__source_id

    @property
    def target_id(self) -> Optional[URIRef]:
        return self.__target_id

    def __get_port(self, port: URIRef|BNode, reln: URIRef) -> Optional[URIRef]:
    #==========================================================================
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
    def __init__(self, model: 'BondgraphModel', uri: URIRef, type: URIRef,
            label: Optional[str], value: Optional[Literal], element_port: Optional[PowerPort]):
        super().__init__(model, uri, None, label)
        self.__type = type
        self.__junction = FRAMEWORK.junction(type)
        if self.__junction is None:
            raise ValueError(f'Unknown Junction {type} for node {uri}')
        self.__constitutive_relation = None
        self.__domain = None
        self.__value = value
        self.__associated_element_port = element_port
        self.__variables: list[Variable] = []

    @property
    def constitutive_relation(self) -> Optional[MathML]:
        return self.__constitutive_relation

    @property
    def type(self) -> URIRef:
        return self.__type

    @property
    def variables(self) -> list[Variable]:
        return self.__variables

    def assign_domain_and_variables(self, bond_graph: nx.DiGraph):
    #=============================================================
        attributes = bond_graph.nodes[self.uri]
        if (domain := attributes.get('domain')) is None:
            log.error(f'Cannot find domain for junction {pretty_uri(self.uri)}. Are there bonds to it?')
            return
        self.__domain = domain
        if self.__type == ONENODE_JUNCTION:
            if self.__associated_element_port is not None:
                self.__variables = [self.__associated_element_port.flow.variable]
            else:
                self.__variables = [Variable(self.uri, self.symbol, self.__domain.flow.units, self.__value)]
        elif self.__type == ZERONODE_JUNCTION:
            if self.__associated_element_port is not None:
                self.__variables = [self.__associated_element_port.potential.variable]
            else:
                self.__variables = [Variable(self.uri, self.symbol, self.__domain.potential.units, self.__value)]
        elif self.__type == TRANSFORM_JUNCTION:
            log.error(f'Transform Nodes {pretty_uri(self.uri)} are not yet supported')
            ## each port needs a domain, if gyrator different domains...

    def equations(self, bond_graph: nx.DiGraph) -> list[Equation]:
    #=============================================================
        ## is this where we multiply by bondCount??
        equations: list[Equation] = []
        if bond_graph.degree[self.uri] > 1:   # type: ignore
            # we are connected to several nodes
            inputs = []
            outputs = []
            equal_value: list[str] = []
            if self.__type == ONENODE_JUNCTION:
                # Sum of potentials connected to junction is 0
                inputs = [symbol for node in bond_graph.predecessors(self.uri)
                            if (symbol := potential_symbol(bond_graph.nodes[node])) is not None]
                outputs = [symbol for node in bond_graph.successors(self.uri)
                            if (symbol := potential_symbol(bond_graph.nodes[node])) is not None]
                equal_value.extend([node_dict['port'].flow.variable.symbol
                                for node in bond_graph.predecessors(self.uri)
                                    if 'port' in (node_dict := bond_graph.nodes[node])
                                        and node_dict['element'].element_class != POTENTIAL_SOURCE])
                equal_value.extend([node_dict['port'].flow.variable.symbol
                                for node in bond_graph.successors(self.uri)
                                    if 'port' in (node_dict := bond_graph.nodes[node])
                                        and node_dict['element'].element_class != POTENTIAL_SOURCE])
            elif self.__type == ZERONODE_JUNCTION:
                # Sum of flows connected to junction is 0
                inputs = [symbol for node in bond_graph.predecessors(self.uri)
                            if (symbol := flow_symbol(bond_graph.nodes[node])) is not None]
                outputs = [symbol for node in bond_graph.successors(self.uri)
                            if (symbol := flow_symbol(bond_graph.nodes[node])) is not None]

                equal_value.extend([node_dict['port'].potential.variable.symbol
                                for node in bond_graph.predecessors(self.uri)
                                    if 'port' in (node_dict := bond_graph.nodes[node])
                                        and node_dict['element'].element_class != FLOW_SOURCE])
                equal_value.extend([node_dict['port'].potential.variable.symbol
                                for node in bond_graph.successors(self.uri)
                                    if 'port' in (node_dict := bond_graph.nodes[node])
                                        and node_dict['element'].element_class != FLOW_SOURCE])
            else:
                raise ValueError(f'Unexpected bond graph node for {self.uri}: {self.__type}')

            if len(equal_value):
                # The first junction variable represents the flow/potential of the node itself
                junction_symbol = sympy.Symbol(self.__variables[0].symbol)
                for value in equal_value:
                    # Filter out known equality between an implied junction and an element's port
                    if (symbol := sympy.Symbol(value)) != junction_symbol:
                        equations.append(Equation(junction_symbol, sympy.Symbol(value)))
            if len(inputs) or len(outputs):
                if len(outputs):
                    lhs = outputs.pop()
                    if len(outputs):
                        equations.append(Equation(lhs, sympy.Add(sympy.Mul(-1, sympy.Add(*outputs)), *inputs)))
                    else:
                        equations.append(Equation(lhs, sympy.Add(*inputs)))
                elif len(inputs) > 1:
                    lhs = inputs.pop()
                    equations.append(Equation(lhs, sympy.Mul(-1, sympy.Add(*inputs))))
        return equations

#===============================================================================
#===============================================================================

MODEL_ELEMENTS = """
    SELECT DISTINCT ?uri ?type ?domain ?symbol ?label
    WHERE {
        <%MODEL%> bgf:hasBondElement ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri bgf:hasDomain ?domain }
        OPTIONAL { ?uri bgf:hasSymbol ?symbol }
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri ?type"""

MODEL_JUNCTIONS = """
    SELECT DISTINCT ?uri ?type ?label ?value ?elementPort
    WHERE {
        <%MODEL%> bgf:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL { ?uri rdfs:label ?label }
        OPTIONAL { ?uri bgf:hasValue ?value }
        OPTIONAL { ?uri bgf:hasElementPort ?elementPort }
    }"""

MODEL_BONDS = """
    SELECT DISTINCT ?powerBond ?source ?target ?label
    WHERE {
        <%MODEL%> bgf:hasPowerBond ?powerBond .
        OPTIONAL { ?powerBond bgf:hasSource ?source }
        OPTIONAL { ?powerBond bgf:hasTarget ?target }
        OPTIONAL { ?powerBond rdfs:label ?label }
    }"""

#===============================================================================

class BondgraphModel(Labelled):
    def __init__(self, rdf_graph: RDFGraph, uri: URIRef, label: Optional[str]=None, debug=False):
        super().__init__(uri, label)
        self.__rdf_graph = rdf_graph
        self.__elements = []
        last_element_uri = None
        element = None
        for row in rdf_graph.query(MODEL_ELEMENTS.replace('%MODEL%', uri)):
            if row[0] != last_element_uri:
                if last_element_uri is not None and element is None:
                    log.error(f'BondElement {pretty_uri(last_element_uri)} has no BG-RDF class')
                element = None
                last_element_uri = row[0]
            template = FRAMEWORK.element_template(row[1], row[2])   # pyright: ignore[reportArgumentType]
            if template is not None:
                if element is None:
                    try:
                        element = BondgraphElement.for_model(self, row[0], template, row[2], row[3], row[4]) # type: ignore
                        self.__elements.append(element)
                    except ValueError as e:
                        log.error(str(e))
                else:
                    log.error(f'BondElement {pretty_uri(row[0])} has more than one BG-RDF class')
        if last_element_uri is not None and element is None:
            log.error(f'BondElement {pretty_uri(last_element_uri)} has no BG-RDF class')
        if len(self.__elements) == 0:
            log.error(f'Model {(pretty_uri(uri))} has no elements...')
        element_ports = {}
        for element in self.__elements:
            element_ports.update(element.ports)
        self.__junctions = [
            BondgraphJunction(self, row[0], row[1], row[2], row[3], element_ports.get(row[4]))  # type: ignore
                for row in rdf_graph.query(MODEL_JUNCTIONS.replace('%MODEL%', uri))]
        self.__bonds = []
        for row in rdf_graph.query(MODEL_BONDS.replace('%MODEL%', uri)):
            bond_uri: URIRef = row[0]                                                            # type: ignore
            if row[1] is None or row[2] is None:
                log.error(f'Bond {pretty_uri(bond_uri)} is missing source and/or target node')
                continue
            self.__bonds.append(
                BondgraphBond(self, bond_uri, row[1], row[2], row[3]))                           # type: ignore

        self.__graph = nx.DiGraph()
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
            equations = junction.equations(self.__graph)
            self.__equations.extend(equations)
            for eq in equations:
                eq.provenance = 'js'

        if debug:
            print(f'{bright()}Elements:')
            for element in self.__elements:
                print(' ', pretty_uri(element.uri))
                if (cr := element.constitutive_relation) is not None:
                    for eq in cr.equations:
                        print('   ', eq)
                for eq in element.equations:
                    print('   ', eq)
            print('Junctions:')
            for junction in self.__junctions:
                print(' ', pretty_uri(junction.uri))
                equations = junction.equations(self.__graph)
                for eq in equations:
                    print('   ', eq)

    @property
    def elements(self):
        return self.__elements

    @property
    def junctions(self):
        return self.__junctions

    @property
    def equations(self):
        return self.__equations

    @property
    def network_graph(self) -> nx.DiGraph:
        return self.__graph.copy()

    def __assign_element_variables_and_equations(self):
    #==================================================
        for element in self.__elements:
            element.assign_variables(self.__graph)
        for element in self.__elements:
            element.build_expressions(self.__graph)

    def __assign_junction_domain_and_variables(self):
    #================================================
        for junction in self.__junctions:
            junction.assign_domain_and_variables(self.__graph)

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
                    log.error(f'Node {node} with domain {node_domain} incompatible with {domain}')

        for element in self.__elements:
            for port_id in element.ports.keys():
                check_node(port_id, element.domain)

    # Construct network graph of PowerBonds
    def __make_bond_network(self):
    #=============================
        for element in self.__elements:
            for port_id, port in element.ports.items():
                self.__graph.add_node(port_id,
                    type=self.__rdf_graph.curie(element.type), port=port, element=element, label=element.symbol)
        for junction in self.__junctions:
            self.__graph.add_node(junction.uri,
                type=self.__rdf_graph.curie(junction.type), junction=junction, label=junction.symbol)
        for bond in self.__bonds:
            source = bond.source_id
            target = bond.target_id
            if source not in self.__graph and target not in self.__graph:
                log.error(f'No element or junction for source {pretty_uri(source)} and target {pretty_uri(target)} of bond {pretty_uri(bond.uri)}')
                continue
            elif source not in self.__graph:
                log.error(f'No element or junction for source {pretty_uri(source)} of bond {pretty_uri(bond.uri)}')
                continue
            elif target not in self.__graph:
                log.error(f'No element or junction for target {pretty_uri(target)} of bond {pretty_uri(bond.uri)}')
                continue
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
    SELECT DISTINCT ?blockUrl
    WHERE {
        <%MODEL%>
            a bgf:BondgraphModel ;
            bgf:hasBlock ?blockUrl .
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

BONDGRAPH_MODEL_TEMPLATES = """
    SELECT DISTINCT ?template
    WHERE {
        <%MODEL%>
            a bgf:BondgraphModel ;
            bgf:usesTemplate ?template .
    }"""

#===============================================================================

class BondgraphModelSource:
    def __init__(self, source: str, output_rdf: Optional[Path]=None, debug=False):
        self.__rdf_graph = RDFGraph(NAMESPACES)
        self.__source_path = Path(source).resolve()
        source_url = self.__source_path.as_uri()
        self.__loaded_sources: set[str] = set()
        self.__load_rdf(source_url)
        base_models = []
        for row in self.__rdf_graph.query(BONDGRAPH_MODELS):
            uri = cast(URIRef, row[0])
            base_models.append((uri, row[1]))
            self.__load_blocks(uri, source_url)
            self.__generate_bonds(uri)

        if len(base_models) < 1:
            log.error(f'No BondgraphModels in source {source}')

        if output_rdf is not None:
            with open(output_rdf, 'w') as fp:
                fp.write(self.__rdf_graph.serialise())
            log.info(f'Expanded model saved as {pretty_log(output_rdf)}')

        self.__models: dict[URIRef, BondgraphModel] = {}
        for (uri, label) in base_models:
            for row in self.__rdf_graph.query(BONDGRAPH_MODEL_TEMPLATES.replace('%MODEL%', uri)):
                self.__add_template(row[0])
            self.__models[uri] = BondgraphModel(self.__rdf_graph, uri, label, debug=debug)

    @property
    def models(self):
        return list(self.__models.values())

    def __add_template(self, path: ResultType):
    #==========================================
        if isinstance(path, URIRef):
            FRAMEWORK.add_template(path)
        elif isinstance(path, Literal):
            FRAMEWORK.add_template(self.__source_path
                                        .parent
                                        .joinpath(str(path))
                                        .resolve())

    def __generate_bonds(self, model_uri: URIRef):
    #=============================================
        for row in self.__rdf_graph.query(BONDGRAPH_BONDS):
            if isinstance(row[1], URIRef):
                source = row[1]
            elif isinstance(row[1], BNode) and isinstance(row[3], URIRef):
                source = row[3]
            else:
                source = None
            if isinstance(row[2], URIRef):
                target = row[2]
            elif isinstance(row[2], BNode) and isinstance(row[4], URIRef):
                target = row[4]
            else:
                target = None
            if (((None, BGF.hasBondElement, source) in self.__rdf_graph
              or (None, BGF.hasJunctionStructure, source) in self.__rdf_graph)
            and ((None, BGF.hasBondElement, target) in self.__rdf_graph
              or (None, BGF.hasJunctionStructure, target) in self.__rdf_graph)):
                self.__rdf_graph.add((model_uri, BGF.hasPowerBond, row[0]))

    def __load_blocks(self, model_uri: URIRef, base_path: str):
    #==========================================================
        ## need to make sure blocks are only loaded once. c.f templates
        for row in self.__rdf_graph.query(BONDGRAPH_MODEL_BLOCKS.replace('%MODEL%', str(model_uri))):
            self.__load_rdf(urldefrag(str(row[0])).url)

    def __load_rdf(self, source_path: str):
    #======================================
        if source_path not in self.__loaded_sources:
            self.__loaded_sources.add(source_path)
            graph = RDFGraph(NAMESPACES)
            graph.parse(source_path)
            for row in graph.query(BONDGRAPH_MODELS):
                if isinstance(row[0], URIRef):
                    #FRAMEWORK.resolve_composites(row[0], graph)
                    self.__load_blocks(row[0], source_path)
            self.__rdf_graph.merge(graph)

#===============================================================================
#===============================================================================
