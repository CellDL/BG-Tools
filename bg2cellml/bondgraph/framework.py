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

from typing import Optional, Self

#===============================================================================

import rdflib
from rdflib.namespace import XSD

#===============================================================================

from ..rdf import Labelled, NamespaceMap
from ..units import Units, Value

from ..mathml import MathML
from .namespaces import BGF, CDT, NAMESPACES

#===============================================================================

TIME_SYMBOL = 't'
TIME_UCUMUNIT = rdflib.Literal('s', datatype=CDT.ucumunit)

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

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = f"""
    SELECT DISTINCT ?symbol ?units ?value
    WHERE {{
        %ELEMENT_URI% bgf:hasVariable ?variable .
        ?variable bgf:hasSymbol ?symbol .
        OPTIONAL {{ ?variable bgf:hasUnits ?units }}
        OPTIONAL {{ ?variable bgf:hasValue ?value }}
    }} ORDER BY ?variable"""

#===============================================================================

class TemplateVariable:
    def __init__(self, element_uri: str, symbol: str, units: Optional[rdflib.Literal], value: Optional[rdflib.Literal]):
        self.__element_uri = element_uri
        self.__symbol = symbol
        self.__units = Units.from_ucum(units) if units is not None else None
        if value is not None:
            self.__value = Value(value)
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

    def set_value(self, value: Value):
    #=================================
        self.__value = value
        if self.__value.units is None:
            self.__value.set_units(self.__units)                        # type: ignore
        elif not self.__units.is_compatible_with(self.__value.units):   # type: ignore
            raise ValueError(
                f'Value for variable {self.__symbol} has incompatible units ({self.__units} != {self.__value.units})') # type: ignore

#===============================================================================
#===============================================================================

DOMAIN_QUERY = f"""
    SELECT DISTINCT ?domain ?label ?flowUnits ?potentialUnits
    WHERE {{
        ?domain a bgf:Domain ;
            bgf:hasFlow [
                bgf:hasUnits ?flowUnits
            ] ;
            bgf:hasPotential [
                bgf:hasUnits ?potentialUnits
            ] .
        OPTIONAL {{ ?domain rdfs:label ?label }}
    }} ORDER BY ?domain"""

#===============================================================================

class Domain(Labelled):
    def __init__(self, uri: str, label: Optional[str], flow_units: rdflib.Literal, potential_units: rdflib.Literal):
        super().__init__(uri, label)
        self.__flow_units = Units.from_ucum(flow_units)
        self.__potential_units = Units.from_ucum(potential_units)

    @property
    def flow_units(self):
        return self.__flow_units

    @property
    def potential_units(self):
        return self.__potential_units

#===============================================================================
#===============================================================================

ELEMENT_PORTS = f"""
    SELECT DISTINCT ?portNumber ?domain ?flowSymbol ?potentialSymbol
    WHERE {{
        %ELEMENT_URI% bgf:hasPort ?port .
        ?port bgf:hasDomain ?domain ;
            bgf:hasFlow [
                bgf:hasSymbol ?flowSymbol
            ] ;
            bgf:hasPotential [
                bgf:hasSymbol ?potentialSymbol
            ] .
        OPTIONAL {{ ?port bgf:portNumber ?portNumber }}
    }}
    ORDER BY ?port"""

#===============================================================================

class PowerPort:
    def __init__(self, element_uri: str, number: Optional[rdflib.Literal], domain: str, flow: str, potential: str):
        self.__element_uri = element_uri
        self.__domain_uri = domain
        self.__flow_symbol = flow
        self.__potential_symbol = potential
        self.__port_number = optional_integer(number, 0)

    @property
    def element_uri(self):
        return self.__element_uri

    @property
    def domain_uri(self):
        return self.__domain_uri

    @property
    def flow_symbol(self):
        return self.__flow_symbol

    @property
    def port_number(self):
        return self.__port_number

    @property
    def potential_symbol(self):
        return self.__potential_symbol

#===============================================================================
#===============================================================================

ELEMENT_DEFINITIONS = f"""
    SELECT DISTINCT ?definition ?label ?relation
    WHERE {{
        ?definition a bgf:ElementTemplate ;
          rdfs:subClassOf bg:BondElement ;
          bgf:constitutiveRelation ?relation .
        OPTIONAL {{ ?definition rdfs:label ?label }}
    }} ORDER BY ?definition"""

#===============================================================================

class ElementTemplate(Labelled):
    def __init__(self, uri: str, label: Optional[str], relation: str|rdflib.Literal):
        super().__init__(uri, label)
        mathml = None
        if isinstance(relation, rdflib.Literal):
            if relation.datatype == BGF.mathml:
                mathml = str(relation)
        else:
            # Do we insist on datatyping? Default to MathML ??
            mathml = relation
        if mathml is None:
            raise ValueError(f'BondElement {uri} has no constitutive relation')
        self.__time_variable = TemplateVariable(self.uri, TIME_SYMBOL, TIME_UCUMUNIT, None)
        self.__relation = MathML.from_string(mathml)
        self.__ports = []
        self.__variables = []

    @classmethod
    def from_knowledge(cls, knowledge: rdflib.Graph, *args) -> Self:
        self = cls(*args)
        self.__add_ports(knowledge)
        self.__add_variables(knowledge)
        self.__check_symbols()
        return self

    @property
    def constitutive_relation(self) -> MathML:
        return self.__relation

    @property
    def ports(self):
        return self.__ports

    @property
    def variables(self):
        return self.__variables

    def __add_ports(self, knowledge: rdflib.Graph):
    #==============================================
        self.__ports.extend([PowerPort(self.uri, *row)
                                for row in sparql_query(knowledge,
                                                        ELEMENT_PORTS.replace('%ELEMENT_URI%', self.uri))])

    def __add_variables(self, knowledge: rdflib.Graph):
    #==================================================
        self.__variables.extend([TemplateVariable(self.uri, *row)
                                for row in sparql_query(knowledge,
                                                        ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri))])

    def __check_symbols(self):
    #=========================
        symbols = []
        def add_symbol(symbol: str, unique=True):
            if symbol not in symbols:
                symbols.append(symbol)
            elif unique:
                raise ValueError(f'Duplicate symbol `{symbol}` for {self.uri}')
        for port in self.__ports:
            add_symbol(port.flow_symbol, False)
            add_symbol(port.potential_symbol, False)
        for variable in self.__variables:
            add_symbol(variable.symbol)
        eqn_symbols = self.__relation.symbols
        print(symbols, eqn_symbols)   ## <<<<<<<<<<<<<<<<<<<<
        if len(symbols) > len(eqn_symbols):
            raise ValueError(f"{self.uri} has variables that are not in it's constitutive relation")
        for eqn_symbol in eqn_symbols:
            if eqn_symbol != self.__time_variable.symbol and eqn_symbol not in symbols:
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
        self.__num_ports = optional_integer(num_ports)

    @property
    def num_ports(self):
        return self.__num_ports

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
        self.__domains = {row[0]: Domain(*row)
                                for row in sparql_query(self.__knowledge, DOMAIN_QUERY)}
        self.__elements = {row[0]: ElementTemplate.from_knowledge(self.__knowledge, *row)
                                for row in sparql_query(self.__knowledge, ELEMENT_DEFINITIONS)}
        self.__junctions = {row[0]: JunctionStructure(*row)
                                for row in sparql_query(self.__knowledge, JUNCTION_STRUCTURES)}

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
    '../bg-rdf/schema/ontology.ttl',
    '../bg-rdf/schema/elements/general.ttl',
    '../bg-rdf/schema/elements/biochemical.ttl',
    '../bg-rdf/schema/elements/electrical.ttl'
])

#===============================================================================
#===============================================================================
