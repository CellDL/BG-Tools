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

from ..utils import Issue

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

type ResultType = BlankNode | Literal | NamedNode | None
type ResultRow = dict[str, ResultType]

Triple = namedtuple('Triple', 'subject, predicate, object')

#===============================================================================

class RdfGraph:
    def __init__(self, namespaces: Optional[dict[str, str]]=None):
        self.__graph = oxigraph.Store()
        self.__namespaces = namespaces or {}
        self.__sparql_prefixes = '\n'.join([
            f'PREFIX {prefix}: <{ns_uri}>' for prefix, ns_uri in self.__namespaces.items()
        ])

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

    def merge(self, graph: 'RdfGraph'):
    #==================================
        self.__graph.extend(graph.__graph.quads_for_pattern(None, None, None))

    def load(self, base_iri: str, source: str):
    #==========================================
        self.__graph.load(input=source, format=oxigraph.RdfFormat.TURTLE, base_iri=base_iri)

    def query(self, query: str) -> list[ResultRow]:
    #==============================================
        query = f'{self.__sparql_prefixes}\n{query}'
        try:
            return self.__graph.query(query)    # type: ignore
        except Exception as e:
            raise Issue(f'{e}: {query}')

#===============================================================================
