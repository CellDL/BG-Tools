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

from ..rdfstore import RdfStore
from ..rdfstore import BlankNode, Literal, NamedNode
from ..rdfstore import blankNode, literal, namedNode
from ..rdfstore import isBlankNode, isLiteral, isNamedNode

#===============================================================================

from ..utils import log

#===============================================================================

type ResultType = BlankNode | Literal | NamedNode | None
type ResultRow = list[ResultType]

#===============================================================================

class RDFGraph:
    def __init__(self, namespaces: Optional[dict[str, str]]=None):
        self.__graph = RdfStore(bind_namespaces='none', store='Oxigraph')
        if namespaces is not None:
            for prefix, namespace in namespaces.items():
                self.__graph.bind(prefix, namespace)

    @property
    def graph(self):
        return self.__graph

    def __contains__(self, triple: tuple) -> bool:
    #=============================================
        return triple in self.__graph

    def add(self, triple: tuple) -> Self:
    #====================================
        self.__graph.add(triple)
        return self

    def curie(self, uri: NamedNode) -> str:
    #======================================
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
            self.__graph.parse(location=str(source_uri), format='ox-turtle')
            return True
        except Exception as e:
            log.error(str(e))
            return False

    def remove(self, triple: tuple) -> Self:
    #=======================================
        self.__graph.remove(triple)
        return self

    def serialise(self, source_url: Optional[str]=None) -> str:
    #==========================================================
        if source_url is not None:
            self.__graph.bind('', f'{source_url}#')
        return self.__graph.serialize(format='turtle')

    def triples(self, triple: tuple):
    #================================
        return self.__graph.triples(triple)

#===============================================================================
