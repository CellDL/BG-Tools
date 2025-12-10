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

from collections import namedtuple
from pathlib import Path
from typing import Any, Optional, Self

#===============================================================================

import pyoxigraph as oxigraph

#===============================================================================

from ..utils import log

#===============================================================================

BlankNode = oxigraph.BlankNode
Literal = oxigraph.Literal
NamedNode = oxigraph.NamedNode

#===============================================================================

def blankNode(value: Optional[str]=None) -> BlankNode:
    return BlankNode(value)

def literal(value: str|int|float|bool, datatype: Optional[NamedNode]=None) -> Literal:
    return Literal(value, datatype=datatype)

def namedNode(uri: str) -> NamedNode:
    return NamedNode(uri)

#===============================================================================

def isBlankNode(node: Any) -> bool:
    return isinstance(node, BlankNode)

def isLiteral(node: Any) -> bool:
    return isinstance(node, Literal)

def isNamedNode(node: Any) -> bool:
    return isinstance(node, NamedNode)

#===============================================================================

def uri_fragment(uri: NamedNode) -> str:
    full_uri = uri.value
    return full_uri.rsplit('#')[-1]

#===============================================================================

type ResultType = BlankNode | Literal | NamedNode | None
type ResultRow = list[ResultType]

Triple = namedtuple('Triple', 'subject, predicate, object')

#===============================================================================

class RDFGraph:
    def __init__(self, namespaces: Optional[dict[str, str]]=None):
        self.__graph = oxigraph.Store()
        self.__namespaces = namespaces or {}
        self.__sparql_prefixes = '\n'.join([
            f'PREFIX {prefix}: <{ns_uri}>' for prefix, ns_uri in self.__namespaces.items()
        ])

    @property
    def graph(self):
        return self.__graph

    def __contains__(self, triple: Triple) -> bool:
    #==============================================
        try:
            self.__graph.quads_for_pattern(triple.subject, triple.predicate, triple.object).__next__()
            return True
        except StopIteration:
            return False

    def add(self, triple: Triple) -> Self:
    #=====================================
        self.__graph.add(oxigraph.Quad(triple.subject, triple.predicate, triple.object))
        return self

    def curie(self, uri: NamedNode) -> str:
    #======================================
        full_uri = uri.value
        for prefix, ns_uri in self.__namespaces.items():
            if full_uri.startswith(ns_uri):
                return f'{prefix}:{full_uri[len(ns_uri):]}'
        return full_uri

    def merge(self, graph: 'RDFGraph'):
    #==================================
        self.__graph.extend(graph.__graph.quads_for_pattern(None, None, None))

    def parse(self, source_uri: str|Path) -> bool:
    #=============================================
        try:
            print(f'Loading {source_uri}')
            self.__graph.load(path=source_uri, format=oxigraph.RdfFormat.TURTLE)
            return True
        except Exception as e:
            log.error(f'{e}: {source_uri}')
            return False

    def query(self, query: str) -> list[ResultRow]:
    #==============================================
        query = f'{self.__sparql_prefixes}\n{query}'
        try:
            return self.__graph.query(query)    # type: ignore
        except Exception as e:
            log.error(str(e))
            log.info(f'Query: {query}')
            return []

    def serialise(self, source_url: Optional[str]=None) -> str:
    #==========================================================
        if source_url is not None:
            self.__namespaces[''] =f'{source_url}#'
        bytes = self.__graph.dump(format=oxigraph.RdfFormat.TURTLE,
                                  prefixes=self.__namespaces)
        return bytes.decode('utf-8')    # pyright: ignore[reportOptionalMemberAccess]

#===============================================================================
