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

from typing import Optional

#===============================================================================

import rdflib
from rdflib.namespace import XSD

#===============================================================================

from ..rdf import NamespaceMap

from .namespaces import BGF, CDT, NAMESPACES

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

def optional_integer(value: Optional[rdflib.Literal]):
#=====================================================
    if value is not None and value.datatype == XSD.integer:
        return int(value.value)

#===============================================================================

class PowerPort:
    def __init__(self, element_uri: str, number: Optional[rdflib.Literal], domain: str, flow: str, potential: str):
        self.__element = element_uri
        self.__number = optional_integer(number)

#===============================================================================

class Variable:
    def __init__(self, element_uri: str, type: str, symbol: str, units: rdflib.Literal):
        self.__element = element_uri
        self.__type = type
        self.__symbol = symbol
        self.__units = units.value if units.datatype == CDT.ucumunit else None

#===============================================================================

ELEMENT_PORTS = f"""
    SELECT DISTINCT ?port ?portNumber ?domain ?flowSymbol ?potentialSymbol
    WHERE {{
        %ELEMENT_URI% bgf:hasPort ?port .
        ?port bgf:hasDomain ?domain ;
            bgf:hasFlow [
                a bgf:Variable ;
                bgf:hasSymbol ?flowSymbol
            ] ;
            bgf:hasPotential [
                a bgf:Variable ;
                bgf:hasSymbol ?potentialSymbol
            ] .
        OPTIONAL {{ ?port bgf:portNumber ?portNumber }}
    }} ORDER BY ?port"""

#===============================================================================

ELEMENT_VARIABLES = f"""
SELECT DISTINCT ?variable ?varType ?symbol ?units
WHERE {{
    %ELEMENT_URI% %VAR_RELN% ?variable .
    ?variable a ?varType ;
        bgf:hasSymbol ?symbol ;
        bgf:hasUnits ?units .
    FILTER (?varType IN (bgf:Variable, bgf:Constant))
}} ORDER BY ?variable"""

#===============================================================================

class BondElement:
    def __init__(self, uri: str, label: Optional[str], relation: str|rdflib.Literal):
        self.__uri = uri
        self.__label = label
        if isinstance(relation, str):
            # Do we insist on datatyping? Or default to MathML ??
            self.__relation = relation
        elif isinstance(relation, rdflib.Literal) and relation.datatype == BGF.mathml:
            self.__relation = relation.value
        else:
            # raise an error?
            self.__relation = ''
        self.__ports = []
        self.__parameters = []
        self.__states = []

    @property
    def label(self):
        return self.__label

    @property
    def uri(self):
        return self.__uri

    def add_variables(self, knowledge: rdflib.Graph):
    #================================================
        self.__add_ports(knowledge)
        self.__add_parameters_and_states(knowledge)

    def __add_ports(self, knowledge: rdflib.Graph):
    #==============================================
        self.__ports.extend([PowerPort(self.__uri, *row[1:])
                                for row in sparql_query(knowledge,
                                                        ELEMENT_PORTS.replace('%ELEMENT_URI%', self.__uri))])

    def __add_parameters_and_states(self, knowledge: rdflib.Graph):
    #==============================================================
        self.__parameters.extend([Variable(self.__uri, *row[1:])
                                for row in sparql_query(knowledge,
                                                        ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.__uri)
                                                                         .replace('%VAR_RELN%', 'bgf:hasParameter'))])
        self.__states.extend([Variable(self.__uri, *row[1:])
                                for row in sparql_query(knowledge,
                                                        ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.__uri)
                                                                         .replace('%VAR_RELN%', 'bgf:hasState'))])

#===============================================================================

class JunctionStructure:
    def __init__(self, uri: str, label: Optional[str], num_ports: Optional[rdflib.Literal]):
        self.__uri = uri
        self.__num_ports = optional_integer(num_ports)

    @property
    def num_ports(self):
        return self.__num_ports

    @property
    def uri(self):
        return self.__uri

#===============================================================================

ELEMENT_DEFINITIONS = f"""
    SELECT DISTINCT ?definition ?label ?relation
    WHERE {{
        ?definition a bgf:NodeDefinition ;
          rdfs:subClassOf bg:BondElement ;
          bgf:constitutiveRelation ?relation .
        OPTIONAL {{ ?definition rdfs:label ?label }}
    }} ORDER BY ?definition"""

JUNCTION_STRUCTURES = f"""
    SELECT DISTINCT ?junction ?label ?numPorts
    WHERE {{
        ?junction rdfs:subClassOf bg:JunctionStructure .
        OPTIONAL {{ ?junction rdfs:label ?label }}
        OPTIONAL {{ ?junction bgf:numPorts ?numPorts }}
    }} ORDER BY ?junction"""

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
        self.__elements: list[BondElement] = []
        for row in sparql_query(self.__knowledge, ELEMENT_DEFINITIONS):
            bond_element = BondElement(*row)
            bond_element.add_variables(self.__knowledge)
            self.__elements.append(bond_element)
        self.__junctions = [JunctionStructure(*row)
                                for row in sparql_query(self.__knowledge, JUNCTION_STRUCTURES)]

    def element_classes(self) -> list[str]:
    #======================================
        return [element.uri for element in self.__elements]

    def junction_classes(self) -> list[str]:
    #=======================================
        return [junction.uri for junction in self.__junctions]

#===============================================================================

BondgraphFramework = _BondgraphFramework([
    '../bg-rdf/schema/ontology.ttl',
    '../bg-rdf/schema/elements/general.ttl',
    '../bg-rdf/schema/elements/biochemical.ttl',
    '../bg-rdf/schema/elements/electrical.ttl'
])

#===============================================================================
