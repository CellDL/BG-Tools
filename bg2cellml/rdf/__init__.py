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
from typing import Optional, Self

#===============================================================================

import rdflib
from rdflib import BNode, Literal, URIRef

#===============================================================================

from ..utils import log, pretty_log

#===============================================================================

type ResultType = BNode | Literal | URIRef | None
type ResultRow = list[ResultType]

#===============================================================================

"""
Generate URIRefs for rdflib.
"""
class Namespace:
    def __init__(self, ns: str):
        self.__ns = ns

    def __str__(self):
        return self.__ns

    def __getattr__(self, attr: str='') -> URIRef:
        return URIRef(f'{self.__ns}{attr}')

#===============================================================================

class RDFGraph:
    def __init__(self, namespaces: Optional[dict[str, str]]=None):
        self.__graph = rdflib.Graph(bind_namespaces='none')
        if namespaces is not None:
            for prefix, namespace in namespaces.items():
                self.__graph.bind(prefix, namespace)

    def __contains__(self, triple: tuple) -> bool:
    #=============================================
        return triple in self.__graph

    def add(self, triple: tuple) -> Self:
    #====================================
        self.__graph.add(triple)
        return self

    def curie(self, uri: URIRef) -> str:
    #===================================
        return uri.n3(self.__graph.namespace_manager)

    def merge(self, graph: 'RDFGraph'):
    #==================================
        self.__graph += graph.__graph

    def query(self, query: str) -> list[ResultRow]:
    #==============================================
        try:
            return self.__graph.query(query)    # type: ignore
        except Exception as e:
            log.error(str(e))
            log.info(f'Query: {query}')
            return []

    def parse(self, source_uri: str|Path) -> bool:
    #=============================================
        try:
            self.__graph.parse(location=str(source_uri), format='turtle')
            return True
        except Exception as e:
            log.error(str(e))
            return False

    def remove(self, triple: tuple) -> Self:
    #=======================================
        self.__graph.remove(triple)
        return self

    def serialise(self) -> str:
    #==========================
        return self.__graph.serialize(format='turtle')

    def triples(self, triple: tuple):
    #================================
        return self.__graph.triples(triple)

#===============================================================================
