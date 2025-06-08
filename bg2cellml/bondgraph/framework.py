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

from typing import NamedTuple, Optional, Self

#===============================================================================

import rdflib
from rdflib.namespace import XSD

#===============================================================================

from ..rdf import Labelled, NamespaceMap
from ..units import Units, Value

from ..mathml import MathML
from .namespaces import BGF, CDT, NAMESPACES

#===============================================================================

# Variable of integration

VOI_SYMBOL = 't'
VOI_UCUMUNIT = rdflib.Literal('s', datatype=CDT.ucumunit)

#===============================================================================

ONENODE_JUNCTION   = 'bgf:OneNode'
TRANSFORM_JUNCTION = 'bgf:TransformNode'    # Can act as a transformer of gyrator
ZERONODE_JUNCTION  = 'bgf:ZeroNode'

#===============================================================================

NAMESPACE_MAP = NamespaceMap(NAMESPACES)
SPARQL_PREFIXES = NAMESPACE_MAP.sparql_prefixes()

#===============================================================================

def sparql_query(rdf_graph: rdflib.Graph, query: str) -> list[list]:
#===================================================================
    query_result = rdf_graph.query(f'{SPARQL_PREFIXES}\n{query}')
    if query_result is not None:
        return [[NAMESPACE_MAP.simplify(term) for term in row]   # type: ignore
                                                    for row in query_result]
    return []

def optional_integer(value: Optional[rdflib.Literal], default: Optional[int]=None) -> Optional[int]:
#===================================================================================================
    if value is not None and value.datatype == XSD.integer:
        return int(value)
    return default

def clean_name(curie: str) -> str:
#=================================
    return curie.strip(':').replace(':', '_').replace('-', '_')

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = f"""
    SELECT DISTINCT ?symbol ?units ?value
    WHERE {{
        %ELEMENT_URI% bgf:hasVariable ?variable .
        ?variable bgf:hasSymbol ?symbol .
        OPTIONAL {{ ?variable bgf:hasUnits ?units }}
        OPTIONAL {{ ?variable bgf:hasValue ?value }}
    }}"""

#===============================================================================

class Variable:
    def __init__(self, element_uri: str, symbol: str, units: Optional[rdflib.Literal|Units], value: Optional[rdflib.Literal]):
        self.__element_uri = element_uri
        self.__symbol = clean_name(symbol)
        self.__units = Units.from_ucum(units) if isinstance(units, rdflib.Literal) else units
        if value is not None:
            self.__value = Value.from_literal(value)
            if self.__units is None and self.__value.units is not None:
                self.__units = self.__value.units
        else:
            self.__value = None
        if self.__units is None:
            raise ValueError(f'Variable {symbol} for {element_uri} has no Units specified')
        if self.__value is not None:
            if self.__value.units is None:
                self.__value.set_units(self.__units)
            elif not self.__units.is_compatible_with(self.__value.units):
                raise ValueError(f'Value for variable {symbol} has incompatible units ({self.__value.units} != {self.__units})')

    def __str__(self):
        return f'{self.__symbol} ({self.__value if self.__value is not None else ''}  {self.__units})'

    @property
    def element_uri(self):
        return self.__element_uri

    @property
    def symbol(self):
        return self.__symbol

    @property
    def value(self):
        return self.__value

    @property
    def units(self):
        return self.__units

    def copy(self, suffix: Optional[str]=None) -> 'Variable':
    #========================================================
        symbol = self.__symbol if suffix is None else f'{self.__symbol}_{clean_name(suffix)}'
        copy = Variable(self.__element_uri, symbol, self.__units, None)
        copy.__value = self.__value.copy() if self.__value is not None else None
        return copy

    def set_value(self, value: Value):
    #=================================
        self.__value = value
        if self.__value.units is None:
            self.__value.set_units(self.__units)                        # type: ignore
        elif not self.__units.is_compatible_with(self.__value.units):   # type: ignore
            raise ValueError(
                f'Value for variable {self.__symbol} has incompatible units ({self.__units} != {self.__value.units})') # type: ignore

#===============================================================================

VOI_VARIABLE = Variable('', VOI_SYMBOL, VOI_UCUMUNIT, None)

#===============================================================================
#===============================================================================

DOMAIN_QUERY = f"""
    SELECT DISTINCT ?domain ?label ?flowSymbol ?flowUnits ?potentialSymbol ?potentialUnits
    WHERE {{
        ?domain
            a bgf:Domain ;
            bgf:hasFlow [
                bgf:hasSymbol ?flowSymbol ;
                bgf:hasUnits ?flowUnits
            ] ;
            bgf:hasPotential [
                bgf:hasSymbol ?potentialSymbol ;
                bgf:hasUnits ?potentialUnits
            ] .
        OPTIONAL {{ ?domain rdfs:label ?label }}
    }} ORDER BY ?domain"""

DOMAIN_CONSTANTS = f"""
    SELECT DISTINCT ?symbol ?value
    WHERE {{
        %DOMAIN_URI%
            a bgf:Domain ;
            bgf:hasConstant [
                bgf:hasSymbol ?symbol ;
                bgf:hasValue ?value
            ] .
    }}"""

#===============================================================================

class Domain(Labelled):
    def __init__(self, uri: str, label: Optional[str],
                    flow_symbol: str, flow_units: rdflib.Literal,
                    potential_symbol: str, potential_units: rdflib.Literal):
        super().__init__(uri, label)
        self.__flow = Variable(self.uri, flow_symbol, flow_units, None)
        self.__potential = Variable(self.uri, potential_symbol, potential_units, None)
        self.__constants: list[Variable] = []

    @classmethod
    def from_framework(cls, framework: '_BondgraphFramework', *args) -> Self:
        self = cls(*args)
        self.__add_constants(framework)
        return self

    def __eq__(self, other):
        return self.uri == other.uri

    def __str__(self):
        return self.uri

    @property
    def constants(self):
        return self.__constants

    @property
    def flow(self):
        return self.__flow

    @property
    def potential(self):
        return self.__potential

    def __add_constants(self, framework: '_BondgraphFramework'):
    #===========================================================
        self.__constants.extend([Variable(self.uri, row[0], None, row[1])
                                for row in sparql_query(framework.knowledge,
                                                        DOMAIN_CONSTANTS.replace('%DOMAIN_URI%', self.uri))])

#===============================================================================
#===============================================================================

ELEMENT_PORT_IDS = f"""
    SELECT DISTINCT ?portId
    WHERE {{
        %ELEMENT_URI% bgf:hasPortId ?portId .
    }}
    ORDER BY ?port"""

#===============================================================================

class PortSymbolVariable(NamedTuple):
    symbol: str
    variable: Variable

    def __str__(self):
        return f'(symbol: {self.symbol}, variable: {self.variable})'

#===============================================================================

class PowerPort:
    def __init__(self, element: 'ElementTemplate', id: Optional[str]=None):
        self.__element = element
        self.__suffix = '' if id is None else f'_{id}'
        self.__flow = self.__symbol_variable(element.domain.flow)
        self.__potential = self.__symbol_variable(element.domain.potential)

    def __str__(self):
        return f'{self.__element.uri}{self.__suffix}, potential: {self.__potential}, flow: {self.__flow}'

    @property
    def element(self):
        return self.__element

    @property
    def flow(self) -> PortSymbolVariable:
        return self.__flow

    @property
    def potential(self) -> PortSymbolVariable:
        return self.__potential

    def copy(self, suffix: Optional[str]=None) -> 'PowerPort':
    #=========================================================
        copy = PowerPort(self.__element)
        copy.__suffix = self.__suffix
        copy.__flow = PortSymbolVariable(symbol=self.__flow.symbol,
                                         variable=self.__flow.variable.copy(suffix))
        copy.__potential = PortSymbolVariable(symbol=self.__potential.symbol,
                                              variable=self.__potential.variable.copy(suffix))
        return copy

    def __symbol_variable(self, domain_variable: Variable) -> PortSymbolVariable:
    #==============================================================================
        symbol = f'{domain_variable.symbol}{self.__suffix}'
        return PortSymbolVariable(symbol=symbol,
                                  variable=Variable(self.__element.uri, symbol, domain_variable.units, None))

#===============================================================================
#===============================================================================

ELEMENT_DEFINITIONS = f"""
    SELECT DISTINCT ?uri ?label ?domain ?relation
    WHERE {{
        ?uri
            a bgf:ElementTemplate ;
            rdfs:subClassOf bg:BondElement ;
            bgf:hasDomain ?domain ;
            bgf:constitutiveRelation ?relation .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }} ORDER BY ?uri"""

#===============================================================================

class ElementTemplate(Labelled):
    def __init__(self, uri: str, label: Optional[str], domain: Domain, relation: str|rdflib.Literal):
        super().__init__(uri, label)
        self.__domain = domain
        mathml = None
        if isinstance(relation, rdflib.Literal):
            if relation.datatype == BGF.mathml:
                mathml = str(relation)
        else:
            # Do we insist on datatyping? Default to MathML ??
            mathml = relation
        if mathml is None:
            raise ValueError(f'BondElement {uri} has no constitutive relation')
        try:
            self.__relation = MathML.from_string(mathml)
        except ValueError as error:
            raise ValueError(f'{self.uri}: {error}')
        self.__ports: dict[str, PowerPort] = {}
        self.__variables: dict[str, Variable] = {}

    @classmethod
    def from_framework(cls, framework: '_BondgraphFramework', uri, label, domain_uri, relation) -> Self:
        if (domain := framework.domain(domain_uri)) is None:
            raise ValueError(f'Unknown domain {domain_uri} for {uri} element')
        self = cls(uri, label, domain, relation)
        self.__add_ports(framework)
        self.__add_variables(framework)
        self.__check_symbols()
        return self

    @property
    def constitutive_relation(self) -> MathML:
        return self.__relation

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def ports(self) -> dict[str, PowerPort]:
        return self.__ports

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def __add_ports(self, framework: '_BondgraphFramework'):
    #=======================================================
        port_ids = [str(row[0]) for row in sparql_query(framework.knowledge,
                        ELEMENT_PORT_IDS.replace('%ELEMENT_URI%', self.uri))]
        if len(port_ids):
            self.__ports = {id: PowerPort(self, id) for id in port_ids}
        else:
            self.__ports = {'': PowerPort(self)}

    def __add_variables(self, framework: '_BondgraphFramework'):
    #===========================================================
        self.__variables = { row[0]: Variable(self.uri, *row)
                                for row in sparql_query(framework.knowledge,
                                                        ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri)) }

    def __check_symbols(self):
    #=========================
        symbols = []
        def add_symbol(symbol: str, unique=True):
            if symbol not in symbols:
                symbols.append(symbol)
            elif unique:
                raise ValueError(f'Duplicate symbol `{symbol}` for {self.uri}')
        for port in self.__ports.values():
            add_symbol(port.flow.symbol, False)
            add_symbol(port.potential.symbol, False)
        for symbol in self.__variables.keys():
            add_symbol(symbol)
        eqn_symbols = self.__relation.symbols
        if len(symbols) > len(eqn_symbols):
            raise ValueError(f"{self.uri} has variables that are not in it's constitutive relation")
        symbols.extend([c.symbol for c in self.__domain.constants])
        symbols.append(VOI_VARIABLE.symbol)
        for eqn_symbol in eqn_symbols:
            if eqn_symbol not in symbols:
                raise ValueError(f'Constitutive relation of {self.uri} has undeclared symbol {eqn_symbol}')

#===============================================================================
#===============================================================================

JUNCTION_STRUCTURES = f"""
    SELECT DISTINCT ?junction ?label ?numPorts
    WHERE {{
        ?junction rdfs:subClassOf bg:JunctionStructure .
        OPTIONAL {{ ?junction rdfs:label ?label }}
        OPTIONAL {{ ?junction bgf:numPorts ?numPorts }}
    }} ORDER BY ?junction"""

#===============================================================================

class JunctionStructure(Labelled):
    def __init__(self, uri: str, label: Optional[str], num_ports: Optional[rdflib.Literal]):
        super().__init__(uri, label)
        if uri in [ONENODE_JUNCTION, ZERONODE_JUNCTION]:
            self.__fixed_ports = None
        else:
            self.__fixed_ports = optional_integer(num_ports)

    @property
    def fixed_ports(self):
        return self.__fixed_ports

#===============================================================================
#===============================================================================

class _BondgraphFramework:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_BondgraphFramework, cls).__new__(cls)
        return cls._instance

    def __init__(self, bg_knowledge: list[str]):
        self.__knowledge = rdflib.Graph()
        for knowledge in bg_knowledge:
            self.__knowledge.parse(knowledge, format='turtle')
        self.__domains = {row[0]: Domain.from_framework(self, *row)
                                for row in sparql_query(self.__knowledge, DOMAIN_QUERY)}
        self.__elements = {row[0]: ElementTemplate.from_framework(self, *row)
                                for row in sparql_query(self.__knowledge, ELEMENT_DEFINITIONS)}
        self.__junctions = {row[0]: JunctionStructure(*row)
                                for row in sparql_query(self.__knowledge, JUNCTION_STRUCTURES)}


    @property
    def knowledge(self):
        return self.__knowledge

    def domain(self, uri: str) -> Optional[Domain]:
    #==============================================
        return self.__domains.get(uri)

    def element(self, uri: str) -> Optional[ElementTemplate]:
    #========================================================
        return self.__elements.get(uri)

    def element_classes(self) -> list[str]:
    #======================================
        return list(self.__elements.keys())

    def junction(self, uri: str) -> Optional[JunctionStructure]:
    #===========================================================
        return self.__junctions.get(uri)

    def junction_classes(self) -> list[str]:
    #=======================================
        return list(self.__junctions.keys())

#===============================================================================
#===============================================================================

BondgraphFramework = _BondgraphFramework([
    '../schema/ontology.ttl',
    '../schema/elements/general.ttl',
    '../schema/elements/biochemical.ttl',
    '../schema/elements/electrical.ttl'
])

#===============================================================================
#===============================================================================
