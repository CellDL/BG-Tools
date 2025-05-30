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

from pprint import pprint

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import Identified, NamespaceMap

from .framework import BondgraphFramework as FRAMEWORK
from .namespaces import NAMESPACES

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
        FILTER (?type IN ({', '.join(FRAMEWORK.element_classes())}))
    }} ORDER BY ?element"""

#===============================================================================

MODEL_JUNCTIONS = f"""
    SELECT DISTINCT ?junction ?type ?domain
    WHERE {{
        %MODEL% bg:hasJunctionStructure ?junction .
        ?junction a ?type .
        OPTIONAL {{ ?junction bgf:hasDomain ?domain }}
        FILTER (?type IN ({', '.join(FRAMEWORK.junction_classes())}))
    }} ORDER BY ?junction"""

#===============================================================================
#===============================================================================

class BondgraphBond(Identified):
    def __init__(self, uri: str, source: str, target: str):
        super().__init__(uri)
        self.__source = source
        self.__target = target

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

#===============================================================================

class BondgraphNode(Identified):
    def __init__(self, uri: str, type: str, domain: Optional[str]=None):
        super().__init__(uri)
        self.__type = type
        self.__domain = domain

#===============================================================================

class BondgraphModel(Identified):
    def __init__(self, source: 'BondgraphModelSource', uri: str):
        super().__init__(uri)
        self.__elements = [BondgraphNode(*row)
                            for row in source.sparql_query(MODEL_ELEMENTS.replace('%MODEL%', uri))]
        self.__junctions = [BondgraphNode(*row)
                            for row in source.sparql_query(MODEL_JUNCTIONS.replace('%MODEL%', uri))]
        self.__bonds = [BondgraphBond(*row)
                            for row in source.sparql_query(MODEL_BONDS.replace('%MODEL%', uri))]
        self.__graph = nx.DiGraph()
        for node in self.__elements:
            self.__graph.add_node(node.uri)
        for node in self.__junctions:
            self.__graph.add_node(node.uri)
        for bond in self.__bonds:
            self.__graph.add_edge(bond.source, bond.target)

#===============================================================================

class BondgraphModelSource:
    def __init__(self, bondgraph_path: str):
        self.__namespace_map = NamespaceMap(NAMESPACES)
        self.__namespace_map.add_namespace('', f'{Path(bondgraph_path).resolve().as_uri()}#')
        self.__sparql_prefixes = self.__namespace_map.sparql_prefixes()
        self.__rdf = rdflib.Graph()
        self.__rdf.parse(bondgraph_path, format='turtle')
        self.__models = [BondgraphModel(self, row[0]) for row in self.sparql_query(BONDGRAPH_MODELS)]

    def sparql_query(self, query: str) -> list[list]:
    #================================================
        query_result = self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')
        if query_result is not None:
            return [[self.__namespace_map.simplify(term) for term in row]   # type: ignore
                                                                for row in query_result]
        return []

#===============================================================================
