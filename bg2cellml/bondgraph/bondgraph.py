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
from typing import TYPE_CHECKING

from pprint import pprint

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import NamespaceMap
from .namespaces import NAMESPACES

if TYPE_CHECKING:
    from .framework import BondgraphFramework

#===============================================================================


#===============================================================================

BONDGRAPH_MODELS = f"""
SELECT DISTINCT ?model
WHERE {{
    ?model a bg:BondGraph .
}} ORDER BY ?model"""

#===============================================================================

MODEL_BONDS = f"""
SELECT DISTINCT ?bond ?source ?target
WHERE {{
    %MODEL% bg:hasPowerBond ?bond .
    ?bond bgf:hasSource ?source .
    ?bond bgf:hasTarget ?target .
}} ORDER BY ?bond ?source ?target"""

#===============================================================================

MODEL_ELEMENTS = f"""
SELECT DISTINCT ?element ?type
WHERE {{
    %MODEL% bg:hasBondElement ?element .
    ?element a ?type .
    OPTIONAL {{ ?element bgf:parameterValue ?param }}
    OPTIONAL {{ ?element bgf:stateValue ?state }}
    FILTER (?type IN (%ELEMENT_CLASSES%))
}} ORDER BY ?element"""

MODEL_JUNCTIONS = f"""
SELECT DISTINCT ?junction ?type ?domain
WHERE {{
    %MODEL% bg:hasJunctionStructure ?junction .
    ?junction a ?type .
    OPTIONAL {{ ?junction bgf:hasDomain ?domain }}
    FILTER (?type IN (%JUNCTION_CLASSES%))
}} ORDER BY ?junction"""

#===============================================================================

class BondgraphModelSet:
    def __init__(self, bondgraph_path: str, framework: 'BondgraphFramework'):
        self.__namespace_map = NamespaceMap(NAMESPACES)
        self.__namespace_map.add_namespace('', f'{Path(bondgraph_path).resolve().as_uri()}#')
        self.__sparql_prefixes = self.__namespace_map.sparql_prefixes()

        self.__rdf = rdflib.Graph()
        self.__rdf.parse(bondgraph_path, format='turtle')

        self.__graph = nx.DiGraph()

        self.__model_uris = [row[0] for row in self.__query(BONDGRAPH_MODELS)]

        for model in self.__model_uris:
            print(model)
            element_query = (MODEL_ELEMENTS.replace('%MODEL%', model)
                                           .replace('%ELEMENT_CLASSES%', ', '.join(framework.element_classes())))
            pprint(self.__query(element_query))

            junction_query = (MODEL_ELEMENTS.replace('%MODEL%', model)
                                            .replace('%JUNCTION_CLASSES%', ', '.join(framework.junction_classes())))
            pprint(self.__query(junction_query))

            pprint(self.__query(MODEL_BONDS.replace('%MODEL%', model)))

    def __query(self, query: str) -> list[list]:
        query_result = self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')
        if query_result is not None:
            return [[self.__namespace_map.simplify(term) for term in row]   # type: ignore
                                                                for row in query_result]
        return []

#===============================================================================
