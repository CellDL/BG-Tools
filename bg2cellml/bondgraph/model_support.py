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

import re
from typing import Optional, TYPE_CHECKING

#===============================================================================

import networkx as nx
import sympy

#===============================================================================

from ..mathml import Equation, MathML
from ..rdf import BlankNode, Literal, NamedNode
from ..rdf import isBlankNode, isLiteral, isNamedNode, literal_as_string
from ..units import Value
from ..utils import Issue

from .framework_support import BondgraphElementTemplate, CompositeTemplate
from .framework_support import Domain, NamedPortVariable, PowerPort, Variable
from .framework_support import ONENODE_JUNCTION, TRANSFORM_JUNCTION, ZERONODE_JUNCTION
from .framework_support import FLOW_SOURCE, POTENTIAL_SOURCE
from .framework_support import DISSIPATOR, FLOW_STORE, QUANTITY_STORE, REACTION
from .framework_support import GYRATOR_EQUATIONS, TRANSFORMER_EQUATIONS
from .framework_support import TRANSFORM_FLOW_NAME, TRANSFORM_PORT_IDS, TRANSFORM_POTENTIAL_NAME, TRANSFORM_RATIO_NAME
from .namespaces import BGF
from .utils import Labelled, pretty_name

if TYPE_CHECKING:
    from .model import BondgraphModel

#===============================================================================

def make_element_port_uri(element_uri: str, port_id: str) -> str:
#================================================================
    return element_uri if port_id in [None, ''] else f'{element_uri}_{port_id}'

def flow_expression(node_dict: dict, model: 'BondgraphModel') -> Optional[sympy.Expr]:
#=====================================================================================
    if ('power_port' in node_dict and 'element' in node_dict
    and (expr := node_dict['element'].flow_expression) is not None):
        return expr
    return flow_symbol(node_dict, model)

def flow_symbol(node_dict: dict, model: 'BondgraphModel') -> Optional[sympy.Symbol]:
#===================================================================================
    if 'power_port' in node_dict:       # A BondElement or TransformNode's power port
        return sympy.Symbol(node_dict['power_port'].flow.variable.symbol)
    elif 'junction' in node_dict:
        junction: BondgraphJunction = node_dict['junction']
        if junction.type == ONENODE_JUNCTION:
            return sympy.Symbol(junction.variables[''].symbol)
        elif junction.type == ZERONODE_JUNCTION:
            model.report_issue(f'Adjacent Zero Nodes to junction {junction.uri} must be merged')
            return
    model.report_issue(f'Unexpected bond graph node, cannot get flow: {node_dict}')

def potential_expression(node_dict: dict, model: 'BondgraphModel') -> Optional[sympy.Expr]:
#==========================================================================================
    if ('power_port' in node_dict and 'element' in node_dict
    and (expr := node_dict['element'].potential_expression) is not None):
        return expr
    return potential_symbol(node_dict, model)

def potential_symbol(node_dict: dict, model: 'BondgraphModel') -> Optional[sympy.Symbol]:
#========================================================================================
    if 'power_port' in node_dict:       # A BondElement or TransformNode's power port
        return sympy.Symbol(node_dict['power_port'].potential.variable.symbol)
    elif 'junction' in node_dict:
        junction: BondgraphJunction = node_dict['junction']
        if junction.type == ZERONODE_JUNCTION:
            return sympy.Symbol(junction.variables[''].symbol)
        elif junction.type == ONENODE_JUNCTION:
            model.report_issue(f'Adjacent One Nodes to junction {junction.uri} must be merged')
            return
    model.report_issue(f'Unexpected bond graph node, cannot get potential: {node_dict}')

def clean_latex(latex: str) -> str:
#==================================
    latex = re.sub(r'\\[^{]*{([^}]*)}', '_\\1', latex.strip())
    latex = latex.replace('^', '_')
    latex = re.sub(r'[^a-zA-Z_0-9]', '', latex)
    return re.sub(r'^_*', '', latex)

def make_symbolic_name(result_row: dict) -> Optional[str]:
#=========================================================
    symbol = literal_as_string(result_row['symbol'])
    if symbol is not None:
        species = literal_as_string(result_row['species'])
        location = literal_as_string(result_row['location'])
        if species is not None:
            symbol += f'_{clean_latex(species)}'
        if location is not None:
            symbol += f'_{clean_latex(location)}'
    return symbol

#===============================================================================

class ModelElement(Labelled):
    def __init__(self,  model: 'BondgraphModel', uri: NamedNode, symbol: Optional[str]=None, label: Optional[str]=None):
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

class VariableValue:
    def __init__(self, value: Literal|NamedNode, symbol: Optional[Literal]):
        self.__value = value
        self.__symbol = symbol.value if symbol is not None else None

    @property
    def value(self):
        return self.__value

    @property
    def symbol(self):
        return self.__symbol

#===============================================================================

class BondgraphElement(ModelElement):
    def __init__(self,  model: 'BondgraphModel', uri: NamedNode, template: BondgraphElementTemplate,
                        parameter_values: Optional[dict[str, VariableValue]]=None,
                        variable_values: Optional[dict[str, VariableValue]]=None,
                        domain_uri: Optional[NamedNode]=None, value: Optional[Value|MathML]=None,
                        symbol: Optional[str]=None, label: Optional[str]=None):
        super().__init__(model, uri, symbol=symbol, label=label)
        element_name = pretty_name(self.symbol, template.uri)
        if isinstance(template, CompositeTemplate):
            element_template = template.template
            composite = True
        else:
            element_template = template
            composite = False
        if element_template.domain is None:
            model.report_issue(f'No modelling domain for element {uri} with template {element_name}/{domain_uri}')
        elif domain_uri is not None and element_template.domain.uri != domain_uri:
            model.report_issue(f'Domain mismatch for element {uri} with template {element_name}/{domain_uri}')

        self.__domain = element_template.domain
        self.__element_class = element_template.element_class
        self.__type = element_template.uri

        if self.__element_class in [FLOW_SOURCE, POTENTIAL_SOURCE]:
            if isinstance(value, MathML):
                self.__constitutive_relation = value
            else:
                self.__constitutive_relation = None
        else:
            if element_template.constitutive_relation is None:
                model.report_issue(f'Template {element_template.uri} for element {uri} has no constitutive relation')
            else:
                self.__constitutive_relation = element_template.constitutive_relation.copy()

        self.__power_ports: dict[str, PowerPort] = {}
        for port_id, port in element_template.power_ports.items():
            self.__power_ports[make_element_port_uri(self.uri, port_id)] = port.copy(suffix=self.symbol, domain=self.__domain)

        self.__flow: Optional[Variable] = None
        self.__potential: Optional[Variable] = None
        self.__flow_symbol = None
        self.__potential_symbol = None
        self.__flow_expression = None
        self.__potential_expression = None
        self.__equations = []

        self.__implied_junction = None
        if composite:
            if self.__element_class == QUANTITY_STORE:
                self.__implied_junction = ZERONODE_JUNCTION
            elif self.__element_class == FLOW_STORE:
                self.__implied_junction = ZERONODE_JUNCTION
            elif self.__element_class in [DISSIPATOR, REACTION]:
                self.__implied_junction = ONENODE_JUNCTION
        elif self.__element_class == POTENTIAL_SOURCE:
            self.__implied_junction = ZERONODE_JUNCTION
        elif self.__element_class == FLOW_SOURCE:
            self.__implied_junction = ONENODE_JUNCTION

        self.__variables: dict[str, Variable] = {}
        self.__port_variable_names = set()
        for port in self.__power_ports.values():
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
        self.__variable_values: dict[str, VariableValue] = {}
        if parameter_values is None:
            if len(element_template.parameters):
                model.report_issue(f'No parameters given for element {pretty_name(self.symbol, uri)}')
        else:
            for name in element_template.parameters.keys():
                if name not in parameter_values:
                    model.report_issue(f'Missing value for parameter {name} of element {pretty_name(self.symbol, uri)}')
                    continue
            self.__variable_values.update(parameter_values)
        if variable_values is not None:
            for name in variable_values.keys():
                if name not in name not in self.__variables:
                    model.report_issue(f'Unknown variable {name} for element {pretty_name(self.symbol, uri)}')
                    continue
            self.__variable_values.update(variable_values)

        if (intrinsic_var := element_template.intrinsic_variable) is not None:
            self.__variables[intrinsic_var.name] = intrinsic_var.copy(suffix=self.symbol, domain=self.__domain)
            self.__intrinsic_variable = self.__variables[intrinsic_var.name]
            if self.__element_class == FLOW_SOURCE:
                port_var = self.__power_ports[self.uri].flow
                port_var.variable = self.__intrinsic_variable
            elif self.__element_class == POTENTIAL_SOURCE:
                port_var = self.__power_ports[self.uri].potential
                port_var.variable = self.__intrinsic_variable
            if value is not None and isinstance(value, Value):
                self.__intrinsic_variable.set_value(value)
        else:
            self.__intrinsic_variable = None

        if (variable := self.__variables.get(self.__domain.flow.symbol)) is not None:
            self.__flow = variable
            self.__flow_symbol = sympy.Symbol(variable.symbol)
        if (variable := self.__variables.get(self.__domain.potential.symbol)) is not None:
            self.__potential = variable
            self.__potential_symbol = sympy.Symbol(variable.symbol)

    @classmethod
    def for_model(cls, model: 'BondgraphModel', uri: NamedNode, template: BondgraphElementTemplate,
    #==============================================================================================
                    domain_uri: Optional[NamedNode], symbol: Optional[str], label: Optional[str]):
        parameter_values: dict[str, VariableValue] = {row['name'].value:
                                                        VariableValue(row['value'], row.get('symbol'))  # pyright: ignore[reportArgumentType]
            for row in model.sparql_query(ELEMENT_PARAMETER_VALUES.replace('%ELEMENT%', uri.value))
            # ?name ?value ?symbol
        }
        variable_values: dict[str, VariableValue] = {row['name'].value:
                                                        VariableValue(row['value'], row.get('symbol'))  # pyright: ignore[reportArgumentType]
            for row in model.sparql_query(ELEMENT_VARIABLE_VALUES.replace('%ELEMENT%', uri.value))
            # ?name ?value ?symbol
        }
        value: Optional[Value|MathML] = None
        for row in model.sparql_query(ELEMENT_STATE_VALUE.replace('%ELEMENT%', uri.value)):
            # ?value
            if isLiteral(row['value']):
                if row['value'].datatype.value == BGF.mathml.value:   # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                    value = MathML.from_string(row['value'].value)    # pyright: ignore[reportOptionalMemberAccess]
                else:
                    value = Value.from_literal(row['value'])          # pyright: ignore[reportArgumentType]
                break
        return cls(model, uri, template, domain_uri=domain_uri,
                    parameter_values=parameter_values, variable_values=variable_values,
                    value=value, symbol=symbol, label=label)

    @property
    def constitutive_relation(self) -> Optional[MathML]:
        return self.__constitutive_relation

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def element_class(self) -> str:
        return self.__element_class

    @property
    def flow(self) -> Optional[Variable]:
        return self.__flow

    @property
    def flow_expression(self) -> Optional[sympy.Basic]:
        return self.__flow_expression

    @property
    def equations(self) -> list[Equation]:
        return self.__equations

    @property
    def implied_junction(self) -> Optional[str]:
        return self.__implied_junction

    @property
    def power_ports(self) -> dict[str, PowerPort]:
        return self.__power_ports

    @property
    def potential(self) -> Optional[Variable]:
        return self.__potential

    @property
    def potential_expression(self) -> Optional[sympy.Basic]:
        return self.__potential_expression

    @property
    def type(self) -> str:
        return self.__type

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    # Substitute variable symbols into the constitutive relation
    def assign_variables(self, bond_graph: nx.DiGraph):
    #==================================================
        ## Remove variables associated with any unconnected ports
        unused_ports = []
        for port_uri, port in self.__power_ports.items():
            if bond_graph.degree(port_uri) == 0:     # pyright: ignore[reportCallIssue]
                unused_ports.append(port_uri)
                if self.__element_class in [DISSIPATOR, REACTION]:
                    del self.__variables[port.potential.name]
                    self.__port_variable_names.remove(port.potential.name)
        for port_uri in unused_ports:
            del self.__power_ports[port_uri]

        for var_name, value in self.__variable_values.items():
            if (variable := self.__variables.get(var_name)) is None:
                self.__model.report_issue(f'Element {pretty_name(self.symbol, self.uri)} has unknown name {var_name} for {self.__type}')
                continue
            if value.symbol is not None:
                variable.set_symbol(value.symbol)   ## need symbol of value[1]'s element...
            if isLiteral(value.value):
                variable.set_value(Value.from_literal(value.value))  # pyright: ignore[reportArgumentType]
            elif isNamedNode(value.value):
                if value.value.value not in bond_graph:
                    self.__model.report_issue(f'Value for {pretty_name(self.symbol, self.uri)} refers to unknown element: {value.value}')
                    continue
                elif (element := bond_graph.nodes[value.value.value].get('element')) is None:
                    self.__model.report_issue(f'Value for {pretty_name(self.symbol, self.uri)} is not a bond element: {value.value}')
                    continue
                elif element.__intrinsic_variable is None:
                    self.__model.report_issue(f'Value for {pretty_name(self.symbol, self.uri)} is an element with no intrinsic variable: {value.value}')
                    continue
                elif variable.units != element.__intrinsic_variable.units:
                    self.__model.report_issue(f'Units incompatible for {pretty_name(self.symbol, self.uri)} value: {value.value}')
                    continue
                else:
                    self.__variables[var_name] = element.__intrinsic_variable

        if self.__constitutive_relation is not None:
            for name, variable in self.__variables.items():
                self.__constitutive_relation.substitute(name, variable.symbol,
                                                        missing_ok=(name in self.__port_variable_names))
            for eqn in self.__constitutive_relation.equations:
                if eqn.lhs == self.__flow_symbol:
                    self.__flow_expression = eqn.rhs
                elif eqn.lhs == self.__potential_symbol:
                    self.__potential_expression = eqn.rhs

    def build_expressions(self, bond_graph: nx.DiGraph):
    #===================================================
        for port_id, port in self.__power_ports.items():
            bond_inputs = []
            bond_outputs = []
            port_inputs = []
            port_outputs = []

            def update_state_equalities(node, input) -> Optional[sympy.Symbol]:
                node_dict = bond_graph.nodes[node]
                edge = (node, port_id) if input else (port_id, node)
                bond_count = bond_graph.edges[edge].get('bond_count', 1)
                forward_dirn = (port.direction
                            and (input and port.direction.value == BGF.InwardPort.value
                                 or not input and port.direction.value == BGF.OutwardPort.value))
                if self.__implied_junction == ONENODE_JUNCTION:
                    if (expr := potential_expression(node_dict, self.model)) is not None:
                        if bond_count != 1:
                            expr = sympy.Mul(bond_count, expr)
                        if input: bond_inputs.append(expr)
                        else: bond_outputs.append(expr)
                    if len(self.__power_ports) > 1 and self.__element_class == REACTION:
                        if (symbol := potential_symbol(node_dict, self.model)) is not None:
                            if bond_count != 1:
                                symbol = sympy.Mul(bond_count, symbol)
                            if forward_dirn: port_inputs.append(symbol)
                            else: port_outputs.append(symbol)

                elif self.__implied_junction == ZERONODE_JUNCTION:
                    if (expr := flow_expression(bond_graph.nodes[node], self.model)) is not None:
                        if bond_count != 1:
                            expr = sympy.Mul(bond_count, expr)
                        if input: bond_inputs.append(expr)
                        else: bond_outputs.append(expr)
                    if len(self.__power_ports) > 1 and self.__element_class == QUANTITY_STORE:
                        if (symbol := flow_symbol(node_dict, self.model)) is not None:
                            if bond_count != 1:
                                symbol = sympy.Mul(bond_count, symbol)
                            if forward_dirn: port_inputs.append(symbol)
                            else: port_outputs.append(symbol)

            for node in bond_graph.predecessors(port_id):
                update_state_equalities(node, True)
            for node in bond_graph.successors(port_id):
                update_state_equalities(node, False)

            bond_expr = sympy.Add(*bond_inputs, sympy.Mul(-1, sympy.Add(*bond_outputs)))
            port_expr = None
            if len(port_inputs) or len(port_outputs):
                if len(port_outputs):
                    if len(port_inputs):
                        port_expr = sympy.Add(*port_inputs, sympy.Mul(-1, sympy.Add(*port_outputs)))
                    else:
                        port_expr = sympy.Mul(-1, sympy.Add(*port_inputs))
                elif len(port_inputs):
                    port_expr = sympy.Add(*port_inputs)

            if self.__implied_junction == ONENODE_JUNCTION:
                if self.__potential_symbol is not None:
                    self.__equations.append(Equation(self.__potential_symbol, bond_expr))
                if port_expr is not None:
                    potential = sympy.Symbol(port.potential.variable.symbol)
                    self.__equations.append(Equation(potential, port_expr))
            elif self.__implied_junction == ZERONODE_JUNCTION:
                if self.__flow_symbol is not None:
                    self.__equations.append(Equation(self.__flow_symbol, bond_expr))
                if port_expr is not None:
                    flow = sympy.Symbol(port.flow.variable.symbol)
                    self.__equations.append(Equation(flow, port_expr))

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
    def __init__(self, model: 'BondgraphModel', uri: NamedNode,
                        source: NamedNode|BlankNode, target: NamedNode|BlankNode,
                        count: Optional[Literal]=None, label: Optional[str]=None):
        super().__init__(model, uri, label=label)
        self.__source_id = self.__get_port_uri(source, BGF.hasSource)
        self.__target_id = self.__get_port_uri(target, BGF.hasTarget)
        ## Check source and target units match...
        self.__bond_count = int(count.value) if count is not None else 1

    @property
    def bond_count(self) -> int:
        return self.__bond_count

    @property
    def source_id(self) -> Optional[str]:
        return self.__source_id

    @property
    def target_id(self) -> Optional[str]:
        return self.__target_id

    def __get_port_uri(self, port_uri: NamedNode|BlankNode, reln: NamedNode) -> Optional[str]:
    #=========================================================================================
        if isBlankNode(port_uri):
            for row in self.model.sparql_query(
                # ?element ?port
                MODEL_BOND_PORTS.replace('%MODEL%', self.model.uri)
                                .replace('%BOND%', self.uri)
                                .replace('%BOND_RELN%', reln.value)):
                return make_element_port_uri(row['element'].value, row['port'].value)  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
        else:
            return port_uri.value

#===============================================================================
#===============================================================================

class BondgraphJunction(ModelElement):
    def __init__(self, model: 'BondgraphModel', uri: NamedNode, type: NamedNode,
            value: Optional[Literal], symbol: Optional[str], label: Optional[str]):
        super().__init__(model, uri, symbol=symbol, label=label)
        self.__type = type.value
        self.__junction = model.framework.junction(self.__type)
        if self.__junction is None:
            model.report_issue(f'Unknown Junction {self.__type} for node {uri}')
        self.__transform_relation: Optional[MathML] = None
        self.__value = value
        self.__variables: dict[str, Variable] = {}
        self.__equations: list[Equation] = []

    @property
    def equations(self) -> list[Equation]:
        return self.__equations

    @property
    def type(self) -> str:
        return self.__type

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def __get_domain(self, attributes: dict) -> Optional[Domain]:
    #============================================================
        if (domain := attributes.get('domain')) is None:
            self.__model.report_issue(f'Cannot find domain for junction {pretty_name(self.symbol, self.uri)}. Are there bonds to it?')
            return
        return domain

    def assign_node_variables(self, bond_graph: nx.DiGraph):
    #=======================================================
        assert self.__type != TRANSFORM_JUNCTION
        if (domain := self.__get_domain(bond_graph.nodes[self.uri])) is not None:
            if self.__type == ONENODE_JUNCTION:
                self.__variables[''] = Variable(self.uri, self.symbol, units=domain.flow.units, value=self.__value)
            elif self.__type == ZERONODE_JUNCTION:
                self.__variables[''] = Variable(self.uri, self.symbol, units=domain.potential.units, value=self.__value)

    def assign_transform_variables(self, bond_graph: nx.DiGraph):
    #============================================================
        assert self.__type == TRANSFORM_JUNCTION
        domains = []
        graph = bond_graph.to_undirected(as_view=True)
        self.__variables[TRANSFORM_RATIO_NAME] = Variable(self.uri, self.symbol, value=self.__value)
        for port_id in TRANSFORM_PORT_IDS:
            port_uri = make_element_port_uri(self.uri, port_id)
            if (domain := self.__get_domain(bond_graph.nodes[port_uri])) is None:
                return
            domains.append(domain)
            neighbours = list(graph.neighbors(port_uri))
            power_port = None
            flow_name = f'{TRANSFORM_FLOW_NAME}_{self.symbol}_{port_id}'
            flow_var_name = f'{TRANSFORM_FLOW_NAME}_{port_id}'
            potential_name = f'{TRANSFORM_POTENTIAL_NAME}_{self.symbol}_{port_id}'
            potential_var_name = f'{TRANSFORM_POTENTIAL_NAME}_{port_id}'
            if len(neighbours):
                neighbour = graph.nodes[neighbours[0]]
                junction_type = None
                variable = None
                if 'element' in neighbour:
                    # Should be a composite element
                    bond_element: BondgraphElement = neighbour['element']
                    junction_type = bond_element.implied_junction
                    if junction_type == ONENODE_JUNCTION:
                        variable = bond_element.flow
                    elif junction_type == ZERONODE_JUNCTION:
                        variable = bond_element.potential
                elif 'junction' in neighbour:
                    junction: BondgraphJunction = neighbour['junction']
                    junction_type = junction.type
                    variable = junction.variables['']
                if variable is not None:
                    if junction_type == ONENODE_JUNCTION:
                        self.__variables[flow_var_name] = variable
                        potential = Variable(self.uri, potential_name, units=domain.potential.units)
                        self.__variables[potential_var_name] = potential
                        power_port = PowerPort(port_uri, NamedPortVariable(flow_name, variable),
                                                         NamedPortVariable(potential_name, potential))
                    elif junction_type == ZERONODE_JUNCTION:
                        flow = Variable(self.uri, flow_name, units=domain.flow.units)
                        self.__variables[flow_var_name] = flow
                        self.__variables[potential_var_name] = variable
                        power_port = PowerPort(port_uri, NamedPortVariable(flow_name, flow),
                                                         NamedPortVariable(potential_name, variable))
            if power_port is not None:
                bond_graph.nodes[port_uri]['power_port'] = power_port
                bond_graph.nodes[port_uri]['port_type'] = self.type
        if domains[0] == domains[1]:
            self.__transform_relation = TRANSFORMER_EQUATIONS.copy()
        else:
            self.__transform_relation = GYRATOR_EQUATIONS.copy()

    def build_equations(self, bond_graph: nx.DiGraph):
    #=================================================
        ## is this where we multiply by bondCount??
        if self.__type == TRANSFORM_JUNCTION:
            assert self.__transform_relation is not None
            for name, variable in self.__variables.items():
                self.__transform_relation.substitute(name, variable.symbol)
            self.__equations = self.__transform_relation.equations
        elif bond_graph.degree[self.uri] > 1:       # pyright: ignore[reportIndexIssue]
            # we are connected to several nodes
            inputs = []
            outputs = []
            equation_lhs = []   # Use a list to pass value from `update_state_equalities` function
            equal_value: list[str] = []

            def update_state_equalities(node, input):
                node_dict = bond_graph.nodes[node]
                edge = (node, self.uri) if input else (self.uri, node)
                bond_count = bond_graph.edges[edge].get('bond_count', 1)
                if self.__type == ONENODE_JUNCTION:
                    if (symbol := potential_symbol(bond_graph.nodes[node], self.model)) is not None:
                        if bond_count != 1:
                            symbol = sympy.Mul(bond_count, symbol)
                        if input: inputs.append(symbol)
                        else: outputs.append(symbol)
                    if 'element' in node_dict and node_dict['element'].implied_junction is None:
                        equal_value.append(node_dict['power_port'].flow.variable.symbol)
                elif self.__type == ZERONODE_JUNCTION:
                    if (symbol := flow_symbol(bond_graph.nodes[node], self.model)) is not None:
                        if bond_count != 1:
                            symbol = sympy.Mul(bond_count, symbol)
                        if input: inputs.append(symbol)
                        else: outputs.append(symbol)
                    if 'element' in node_dict and node_dict['element'].implied_junction is None:
                        equal_value.append(node_dict['power_port'].potential.variable.symbol)
                if len(equation_lhs) == 0 and (port := node_dict.get('power_port')) is not None:
                    if self.__type == ONENODE_JUNCTION:
                        if node_dict.get('port_type') in [DISSIPATOR, REACTION]:
                            equation_lhs.append(sympy.Symbol(port.potential.variable.symbol))

            for node in bond_graph.predecessors(self.uri):
                update_state_equalities(node, True)
            for node in bond_graph.successors(self.uri):
                update_state_equalities(node, False)

            if len(inputs) or len(outputs):
                lhs = equation_lhs[0] if len(equation_lhs) else None
                if len(outputs):
                    if lhs in inputs:
                        assert lhs is not None
                        inputs.remove(lhs)
                        if len(inputs):
                            self.__equations.append(Equation(lhs, sympy.Add(*outputs, sympy.Mul(-1, sympy.Add(*inputs)))))
                        else:
                            self.__equations.append(Equation(lhs, sympy.Add(*outputs)))
                    else:
                        if lhs in outputs:
                            assert lhs is not None
                            outputs.remove(lhs)
                        else:
                            lhs = outputs.pop()
                        if len(outputs):
                            self.__equations.append(Equation(lhs, sympy.Add(*inputs, sympy.Mul(-1, sympy.Add(*outputs)))))
                        else:
                            self.__equations.append(Equation(lhs, sympy.Add(*inputs)))
                elif len(inputs) > 1:
                    if lhs in inputs:
                        assert lhs is not None
                        inputs.remove(lhs)
                    else:
                        lhs = inputs.pop()
                    self.__equations.append(Equation(lhs, sympy.Mul(-1, sympy.Add(*inputs))))

            if len(equal_value):
                # The first junction variable represents the flow/potential of the node itself
                junction_symbol = sympy.Symbol(self.__variables[''].symbol)
                for value in equal_value:
                    # Filter out known equality between an implied junction and an element's port
                    if (symbol := sympy.Symbol(value)) != junction_symbol:
                        self.__equations.append(Equation(junction_symbol, sympy.Symbol(value)))

        return self.__equations

#===============================================================================
#===============================================================================
