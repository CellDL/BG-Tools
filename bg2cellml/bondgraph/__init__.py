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

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import NamespaceMap

from .framework import BondgraphFramework
from .namespaces import NAMESPACES
from .queries import BONDGRAPH_MODELS, MODEL_ELEMENTS, MODEL_JUNCTIONS, MODEL_BONDS

#===============================================================================

class Bondgraph:
    def __init__(self, bondgraph_path: Path, framework: BondgraphFramework):
        self.__rdf = rdflib.Graph()
        self.__rdf.parse(bondgraph_path, format='turtle')
        self.__namespace_map = NamespaceMap(NAMESPACES)
        self.__namespace_map.add_namespace('', f'{bondgraph_path.resolve().as_uri()}#')
        self.__sparql_prefixes = self.__namespace_map.sparql_prefixes()

        self.__graph = nx.DiGraph()

        self.__model_uris = []
        query_result = self.__query(BONDGRAPH_MODELS)
        if query_result is not None:
            for row in query_result:
                self.__model_uris.append(self.__namespace_map.simplify(row[0])) # type: ignore

        for model in self.__model_uris:
            print(model)
            self.__test_query(MODEL_ELEMENTS.replace('%MODEL%', model))
            self.__test_query(MODEL_JUNCTIONS.replace('%MODEL%', model))
            self.__test_query(MODEL_BONDS.replace('%MODEL%', model))

    def __query(self, query: str):
        return self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')

    def __test_query(self, query: str):
        result = self.__query(query)
        if result is not None:
            for row in result:
                print([self.__namespace_map.simplify(term) for term in row])    # type: ignore

#===============================================================================
