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
import sys
from typing import Any, Sequence, Self

#===============================================================================

import oximock

#===============================================================================

from ..utils import Issue

#===============================================================================

# Exports:

from oximock import blankNode, literal, namedNode

def isBlankNode(term: Any) -> bool:
    return term is not None and oximock.isBlankNode(term)

def isLiteral(term: Any) -> bool:
    return term is not None and oximock.isLiteral(term)

def isNamedNode(term: Any) -> bool:
    return term is not None and oximock.isNamedNode(term)

#===============================================================================

# Dummy classes just for type checking

class Term:
    value = ''

class BlankNode(Term):
    pass

class Literal(Term):
    pass

class NamedNode(Term):
    datatype = ''

#===============================================================================

type ResultType = BlankNode | Literal | NamedNode
type ResultRow = dict[str, ResultType]

Triple = namedtuple('Triple', 'subject, predicate, object')

#===============================================================================

class RdfGraph:
    def __init__(self, namespaces: dict[str, str]|None=None):
        self.__pyodide = 'pyodide' in sys.modules
        self.__store = oximock.RdfStore()
        self.__namespaces = namespaces or {}
        self.__sparql_prefixes = '\n'.join([
            f'PREFIX {prefix}: <{ns_uri}>' for prefix, ns_uri in self.__namespaces.items()
        ])

    def __contains__(self, triple: Triple) -> bool:
    #==============================================
        return self.__store.contains(triple.subject, triple.predicate, triple.object)

    def add(self, triple: Triple) -> Self:
    #=====================================
        self.__store.add(triple.subject, triple.predicate, triple.object)
        return self

    def merge(self, graph: 'RdfGraph'):
    #==================================
        for stmt in graph.__store.statements():
            self.__store.add(stmt.subject, stmt.predicate, stmt.object)

    def load(self, base_iri: str, source: str):
    #==========================================
        self.__store.load(base_iri, source)

    def query(self, query: str) -> Sequence[ResultRow]:
    #==================================================
        query = f'{self.__sparql_prefixes}\n{query}'
        try:
            rows = self.__store.query(query)
            if self.__pyodide:
                #  We are in the browser's Pyodide environment
                return [
                    { k: v.to_py() for k, v in row.items() }
                        for row in rows.to_py() # pyright: ignore[reportAttributeAccessIssue]
                ]
            else:
                # Otherwise convert QuerySolutions to a similar format
                keys = [var.value for var in rows.variables]
                return [ { k: row[k] for k in keys } for row in rows]
        except Exception as e:
            raise Issue(f'{e}: {query}')

#===============================================================================
#===============================================================================
