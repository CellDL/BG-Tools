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

import rdflib

#===============================================================================

from ..rdf import NamespaceMap
from .namespaces import NAMESPACES

#===============================================================================

NAMESPACE_MAP = NamespaceMap(NAMESPACES)
SPARQL_PREFIXES = NAMESPACE_MAP.sparql_prefixes()

#===============================================================================

ELEMENT_DEFINITIONS = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?class ?label ?relation
WHERE {{
    ?class a bgf:NodeDefinition ;
      rdfs:subClassOf bg:BondElement ;
      bgf:constitutiveRelation ?relation .
    OPTIONAL {{ ?class rdfs:label ?label }}
}} ORDER BY ?class"""


ELEMENT_PORTS = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?port ?portNumber ?domain ?flowSymbol ?potentialSymbol
WHERE {{
    %ELEMENT% bgf:hasPort ?port .
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


ELEMENT_VARIABLES = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?variable ?varType ?symbol ?units
WHERE {{
    %ELEMENT% %VAR_RELN% ?variable .
    ?variable a ?varType ;
        bgf:hasSymbol ?symbol ;
        bgf:hasUnits ?units .
    FILTER (?varType IN (bgf:Variable, bgf:Constant))
}} ORDER BY ?variable"""

#===============================================================================

JUNCTION_STRUCTURES = f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?class ?label ?numPorts
WHERE {{
    ?class rdfs:subClassOf bg:JunctionStructure .
    OPTIONAL {{ ?class rdfs:label ?label }}
    OPTIONAL {{ ?class bgf:numPorts ?numPorts }}
}} ORDER BY ?class"""

#===============================================================================

class NodeDefinition:
    pass

#===============================================================================

class BondgraphFramework:
    def __init__(self, bg_knowledge: list[str]):
        self.__knowledge = rdflib.Graph()
        for knowledge in bg_knowledge:
            self.__knowledge.parse(knowledge, format='turtle')

        self.__elements = []
        query_result = self.__knowledge.query(ELEMENT_DEFINITIONS)
        if query_result is not None:
            for row in query_result:
                self.__elements.append(tuple(NAMESPACE_MAP.simplify(term) for term in row)) # type: ignore

        self.__junctions = []
        query_result = self.__knowledge.query(JUNCTION_STRUCTURES)
        if query_result is not None:
            for row in query_result:
                self.__junctions.append(tuple(NAMESPACE_MAP.simplify(term) for term in row)) # type: ignore

        for element in self.__elements:
            print(element[0:2])
            print('Ports:')
            self.__test_query(ELEMENT_PORTS.replace('%ELEMENT%', element[0]))
            print('Params:')
            self.__test_query(ELEMENT_VARIABLES.replace('%ELEMENT%', element[0]).replace('%VAR_RELN%', 'bgf:hasParameter'))
            print('States:')
            self.__test_query(ELEMENT_VARIABLES.replace('%ELEMENT%', element[0]).replace('%VAR_RELN%', 'bgf:hasState'))

    def __test_query(self, query):
        result = self.__knowledge.query(query)
        if result is not None:
            for row in result:
                print([NAMESPACE_MAP.simplify(term) for term in row])    # type: ignore

#===============================================================================
