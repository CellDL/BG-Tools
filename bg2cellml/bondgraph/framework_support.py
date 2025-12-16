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

from dataclasses import dataclass
from typing import Optional, Self

#===============================================================================

from ..mathml import MathML
from ..rdf import literal_as_string, uri_fragment
from ..rdf import isLiteral, Literal, literal, NamedNode, namedNode, RdfGraph
from ..units import Units, Value
from ..utils import Issue

from .namespaces import BGF, CDT
from .utils import clean_name, Labelled, optional_integer

#===============================================================================
#===============================================================================

DISSIPATOR         = BGF.Dissipator.value
FLOW_SOURCE        = BGF.FlowSource.value
FLOW_STORE         = BGF.FlowStore.value
POTENTIAL_SOURCE   = BGF.PotentialSource.value
QUANTITY_STORE     = BGF.QuantityStore.value
REACTION           = BGF.Reaction.value

ONENODE_JUNCTION   = BGF.OneNode.value
TRANSFORM_JUNCTION = BGF.TransformNode.value  # Can act as a transformer of gyrator
ZERONODE_JUNCTION  = BGF.ZeroNode.value

#===============================================================================
#===============================================================================

# Variable of integration

VOI_SYMBOL = 't'
VOI_UCUMUNIT = literal('s', datatype=CDT.ucumunit)      # pyright: ignore[reportArgumentType]

#===============================================================================
#===============================================================================

TRANSFORM_RATIO_NAME     = 'RATIO'

TRANSFORM_FLOW_NAME      = 'v'
TRANSFORM_POTENTIAL_NAME = 'u'

TRANSFORM_PORT_IDS       = ['0', '1']

#===============================================================================

GYRATOR_EQUATIONS = MathML.from_string(f"""
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            </apply>
        </apply>
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            </apply>
        </apply>
    </math>
""")

#===============================================================================

TRANSFORMER_EQUATIONS = MathML.from_string(f"""
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            </apply>
        </apply>
        <apply>
            <eq/>
            <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <divide/>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
            </apply>
        </apply>
    </math>
""")

#===============================================================================
#===============================================================================

class Variable:
    def __init__(self, element_uri: Optional[str], name: str,
                        units: Optional[Literal|Units]=None,
                        value: Optional[Literal]=None):
        self.__element_uri = element_uri
        self.__name = clean_name(name)
        self.__symbol: Optional[str] = None
        if isLiteral(units):
            self.__units = Units.from_ucum(units)   # pyright: ignore[reportArgumentType]
        else:
            self.__units: Optional[Units] = units   # pyright: ignore[reportAttributeAccessIssue]
        if value is not None:
            self.__value = Value.from_literal(value)
            if self.__units is None and self.__value.units is not None:
                self.__units = self.__value.units
        else:
            self.__value = None
        if self.__units is None:
            raise Issue(f'Variable {name} for {element_uri} has no Units specified')
        if self.__value is not None:
            if self.__value.units is None:
                self.__value.set_units(self.__units)
            elif not self.__units.is_compatible_with(self.__value.units):
                raise Issue(f'Value for variable {name} has incompatible units ({self.__value.units} != {self.__units})')

    def __str__(self):
        return f'{self.symbol} ({self.__value if self.__value is not None else self.__units})'

    @property
    def element_uri(self):
        return self.__element_uri

    @property
    def name(self):
        return self.__name

    @property
    def symbol(self) -> str:
        return self.__symbol if self.__symbol is not None else self.__name

    @property
    def value(self):
        return self.__value

    @property
    def units(self) -> Units:
        return self.__units         # type: ignore

    def copy(self, suffix: Optional[str]=None, domain: Optional['Domain']=None) -> 'Variable':
    #=========================================================================================
        if suffix is None:
            name = self.__name
        else:
            suffix = clean_name(suffix)
            if domain is None:
                name = f'{self.__name}_{suffix}'
            else:
                domain_symbols = domain.intrinsic_symbols
                suffix_parts = suffix.split('_')
                if self.name in domain_symbols and self.name == suffix_parts[0]:
                    name = suffix
                else:
                    name = f'{self.__name}_{suffix}'
        copy = Variable(self.__element_uri, name, units=self.__units)
        copy.__value = self.__value.copy() if self.__value is not None else None
        return copy

    def set_symbol(self, symbol: str):
    #=================================
        self.__symbol = symbol

    def set_value(self, value: Value):
    #=================================
        self.__value = value
        if self.__value.units is None:
            self.__value.set_units(self.__units)                        # type: ignore
        elif not self.__units.is_compatible_with(self.__value.units):   # type: ignore
            raise Issue(
                f'Value for variable {self.__name} has incompatible units ({self.__units} != {self.__value.units})') # type: ignore

#===============================================================================

VOI_VARIABLE = Variable(None, VOI_SYMBOL, units=VOI_UCUMUNIT)

#===============================================================================
#===============================================================================

DOMAIN_CONSTANTS = """
    SELECT DISTINCT ?name ?value
    WHERE {
        <%DOMAIN_URI%>
            a bgf:PhysicalDomain ;
            bgf:hasConstant [
                bgf:varName ?name ;
                bgf:hasValue ?value
            ] .
    }"""

#===============================================================================

class Domain(Labelled):
    def __init__(self, uri: NamedNode, label: Optional[str],
                    flow_name: str, flow_units: Literal,
                    potential_name: str, potential_units: Literal,
                    quantity_name: str, quantity_units: Literal):
        super().__init__(uri, label)
        self.__flow = Variable(self.uri, flow_name, units=flow_units)
        self.__potential = Variable(self.uri, potential_name, units=potential_units)
        self.__quantity = Variable(self.uri, quantity_name, units=quantity_units)
        self.__intrinsic_symbols = [
            self.__flow.symbol,
            self.__potential.symbol,
            self.__quantity.symbol
        ]
        self.__constants: list[Variable] = []

    @classmethod
    def from_rdf_graph(cls, graph: RdfGraph,
                    uri: NamedNode, label: Optional[Literal],
                    flow_name: Literal, flow_units: Literal,
                    potential_name: Literal, potential_units: Literal,
                    quantity_name: Literal, quantity_units: Literal) -> Self:
        self = cls(uri, literal_as_string(label),
                        flow_name.value, flow_units,
                        potential_name.value, potential_units,
                        quantity_name.value, quantity_units)
        self.__add_constants(graph)
        return self

    def __eq__(self, other):
        return self.uri == other.uri

    @property
    def constants(self):
        return self.__constants

    @property
    def flow(self):
        return self.__flow

    @property
    def intrinsic_symbols(self):
        return self.__intrinsic_symbols

    @property
    def name(self) -> str:
        return self.symbol

    @property
    def potential(self):
        return self.__potential

    @property
    def quantity(self):
        return self.__quantity

    def __add_constants(self, graph: RdfGraph):
    #==========================================
        self.__constants.extend([Variable(self.uri, row['name'].value, value=row['value'])  # pyright: ignore[reportArgumentType]
                                for row in graph.query(
                                    # ?name ?value
                                    DOMAIN_CONSTANTS.replace('%DOMAIN_URI%', self.uri))])

#===============================================================================
#===============================================================================

@dataclass
class NamedPortVariable:
    name: str
    variable: Variable

    def __str__(self):
        return f'(name: {self.name}, variable: {self.variable})'

#===============================================================================

class PowerPort:
    def __init__(self, uri: str, flow: NamedPortVariable, potential: NamedPortVariable,
                       direction: Optional[NamedNode]=None):
        self.__uri = uri
        self.__flow = flow
        self.__potential = potential
        self.__direction = direction

    def __str__(self):
        return f'{uri_fragment(self.__uri)}, potential: {self.__potential}, flow: {self.__flow}'

    @property
    def direction(self) -> Optional[NamedNode]:
        return self.__direction

    @property
    def flow(self) -> NamedPortVariable:
        return self.__flow

    @property
    def potential(self) -> NamedPortVariable:
        return self.__potential

    def copy(self, suffix: Optional[str]=None, domain: Optional[Domain]=None) -> 'PowerPort':
    #========================================================================================
        return PowerPort(self.__uri,
            NamedPortVariable(name=self.__flow.name, variable=self.__flow.variable.copy(suffix=suffix, domain=domain)),
            NamedPortVariable(name=self.__potential.name, variable=self.__potential.variable.copy(suffix=suffix, domain=domain)),
            direction = self.__direction
        )

#===============================================================================
#===============================================================================

ELEMENT_PARAMETERS = """
    SELECT DISTINCT ?name ?units ?value
    WHERE {
        <%ELEMENT_URI%> bgf:hasParameter ?variable .
        ?variable bgf:varName ?name .
        OPTIONAL { ?variable bgf:hasUnits ?units }
        OPTIONAL { ?variable bgf:hasValue ?value }
    }"""

ELEMENT_VARIABLES = """
    SELECT DISTINCT ?name ?units ?value
    WHERE {
        <%ELEMENT_URI%> bgf:hasVariable ?variable .
        ?variable bgf:varName ?name .
        OPTIONAL { ?variable bgf:hasUnits ?units }
        OPTIONAL { ?variable bgf:hasValue ?value }
    }"""

#===============================================================================

ELEMENT_PORT_BONDS = """
    SELECT DISTINCT ?portId ?bondCount ?direction
    WHERE {
        {
            { <%ELEMENT_URI%> bgf:hasPort ?portId .
            }
        UNION {
            <%ELEMENT_URI%> bgf:hasPort ?port .
            ?port bgf:portId ?portId ;
            OPTIONAL { ?port bgf:bondCount ?bondCount }
            OPTIONAL { ?port bgf:direction ?direction }
            }
        }
    } ORDER BY ?portId"""

#===============================================================================

class ElementTemplate(Labelled):
    def __init__(self, uri: NamedNode, element_class: NamedNode,
                    label: Optional[str], domain: Domain, relation: str|Literal):
        super().__init__(uri, label)
        self.__element_class = element_class.value
        self.__domain = domain
        if self.__element_class in [FLOW_SOURCE, POTENTIAL_SOURCE]:
            self.__relation = None
        else:
            mathml = None
            if isLiteral(relation):
                if relation.datatype.value == BGF.mathml.value: # pyright: ignore[reportAttributeAccessIssue]
                    mathml = relation.value                     # pyright: ignore[reportAttributeAccessIssue]
                else:
                    # Do we insist on datatyping? Default to MathML ??
                    mathml = relation
            if mathml is None:
                raise Issue(f'BondElement {uri} has no constitutive relation')
            elif isLiteral(mathml):
                mathml = mathml.value                           # pyright: ignore[reportAttributeAccessIssue]
            try:
                self.__relation = MathML.from_string(mathml)    # pyright: ignore[reportArgumentType]
            except ValueError as error:
                raise Issue(f'{self.uri}: {error}')
        self.__power_ports: dict[str, PowerPort] = {}
        self.__parameters: dict[str, Variable] = {}
        self.__variables: dict[str, Variable] = {}
        self.__intrinsic_variable: Optional[Variable] = None

    @classmethod
    def from_rdf_graph(cls, graph: RdfGraph, uri: NamedNode, element_class: NamedNode,
                        label: Optional[Literal], domain: Domain, relation: Literal) -> Self:
        self = cls(uri, element_class, literal_as_string(label), domain, relation)
        self.__add_ports(graph)
        self.__add_variables(graph)
        self.__check_names()
        return self

    @property
    def constitutive_relation(self) -> Optional[MathML]:
        return self.__relation

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def element_class(self) -> str:
        return self.__element_class

    @property
    def intrinsic_variable(self) -> Optional[Variable]:
        return self.__intrinsic_variable

    @property
    def parameters(self) -> dict[str, Variable]:
        return self.__parameters

    @property
    def power_ports(self) -> dict[str, PowerPort]:
        return self.__power_ports

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def __add_ports(self, graph: RdfGraph):
    #======================================
        port_bonds: dict[str, int|None] = {}
        directions: dict[str, NamedNode|None] = {}
        for row in graph.query(
                        ELEMENT_PORT_BONDS.replace('%ELEMENT_URI%', self.uri)):
            # ?portId ?bondCount ?direction
            if isLiteral(row['portId']):
                port_bonds[row['portId'].value] = optional_integer(row.get('bondCount'), 1)  # pyright: ignore[reportArgumentType]
                directions[row['portId'].value] = row.get('direction')    # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
        if len(port_bonds):
            flow_suffixed = False ##(len(port_bonds) == 2)
            self.__power_ports = {}
            for id, _ in port_bonds.items():
                suffix = f'_{id}'
                flow_var = self.__port_name_variable(self.domain.flow, suffix if flow_suffixed else '')
                potential_var = self.__port_name_variable(self.domain.potential, suffix)
                self.__power_ports[id] = PowerPort(namedNode(f'{self.uri}{suffix}'),  # pyright: ignore[reportArgumentType]
                                                    flow_var, potential_var, direction=directions[id])
        else:
            self.__power_ports = {'': PowerPort(self.uri,
                                    self.__port_name_variable(self.domain.flow),
                                    self.__port_name_variable(self.domain.potential)
                                 )}

    def __port_name_variable(self, domain_variable: Variable, suffix: str='') -> NamedPortVariable:
    #==============================================================================================
        port_var_name = f'{domain_variable.name}{suffix}'
        return NamedPortVariable(name=port_var_name,
                                variable=Variable(self.uri, port_var_name, units=domain_variable.units))

    def __add_variables(self, graph: RdfGraph):
    #==========================================
        for row in graph.query(ELEMENT_PARAMETERS.replace('%ELEMENT_URI%', self.uri, True)):
            # ?name ?units ?value
            var_name = row['name'].value             # pyright: ignore[reportOptionalMemberAccess]
            if var_name in self.__domain.intrinsic_symbols:
                raise Issue(f'Cannot specify domain symbol {var_name} as a variable for {self.uri}')
            self.__parameters[var_name] = Variable(self.uri, row['name'].value, units=row.get('units'), value=row.get('value'))   # type: ignore
        for row in graph.query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri, True)):
            # ?name ?units ?value
            var_name = row['name'].value             # pyright: ignore[reportOptionalMemberAccess]
            if var_name in self.__domain.intrinsic_symbols:
                raise Issue(f'Cannot specify domain symbol {var_name} as a variable for {self.uri}')
            self.__variables[var_name] = Variable(self.uri, row['name'].value, units=row.get('units'), value=row.get('value'))   # type: ignore
        # A variable that is intrinsic to the element's class
        # Values of intrinsic variables are set by bgf:hasValue
        if self.__element_class == QUANTITY_STORE:
            self.__intrinsic_variable = self.__domain.quantity.copy()
        if self.__element_class == FLOW_STORE:
            self.__intrinsic_variable = self.__domain.flow.copy()
        elif self.__element_class == POTENTIAL_SOURCE:
            self.__intrinsic_variable = self.__domain.potential.copy()
        elif self.__element_class == FLOW_SOURCE:
            self.__intrinsic_variable = self.__domain.flow.copy()

    def __check_names(self):
    #=======================
        names = []
        def add_name(name: str, unique=True):
            if name not in names:
                names.append(name)
            elif unique:
                raise Issue(f'Duplicate name `{name}` for {self.uri}')
        for name in self.__parameters.keys():
            add_name(name)
        for name in self.__variables.keys():
            add_name(name)
        eqn_names = self.__relation.variables if self.__relation is not None else []
        if len(names) > len(eqn_names):
            raise Issue(f"{self.uri} has variables that are not in it's constitutive relation")
        for port in self.__power_ports.values():
            if port.flow is not None:
                add_name(port.flow.name, False)
            if port.potential is not None:
                add_name(port.potential.name, False)
        names.extend([c.name for c in self.__domain.constants])
        names.extend(self.__domain.intrinsic_symbols)
        names.append(VOI_VARIABLE.name)
        for name in eqn_names:
            if name not in names:
                raise Issue(f'Constitutive relation of {self.uri} has undeclared name {name}')

#===============================================================================
#===============================================================================

class JunctionStructure(Labelled):
    def __init__(self, uri: NamedNode, label: Optional[str]):
        super().__init__(uri, label)

#===============================================================================
#===============================================================================

class CompositeElement(Labelled):
    def __init__(self, uri: NamedNode, template: ElementTemplate, junction: JunctionStructure, label: Optional[str]):
        super().__init__(uri, label)
        self.__template = template
        self.__junction = junction

    @property
    def junction(self):
        return self.__junction

    @property
    def template(self):
        return self.__template

#===============================================================================

class CompositeTemplate(Labelled):
    def __init__(self, uri: NamedNode, template: ElementTemplate, label: Optional[str]):
        super().__init__(uri, label)
        self.__template = template

    @property
    def template(self):
        return self.__template

#===============================================================================
#===============================================================================

type BondgraphElementTemplate = ElementTemplate | CompositeTemplate

#===============================================================================
#===============================================================================
