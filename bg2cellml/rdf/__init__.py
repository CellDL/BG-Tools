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

from typing import Optional, Self

#===============================================================================

from rdflib import BNode, Literal, URIRef

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

class NamespaceMap:
    def __init__(self, ns_map: Optional[dict[str, str]]=None):
        self.__prefix_dict: dict[str, str] = {}
        self.__reverse_map: dict[str, str] = {}
        if ns_map is not None:
            for prefix, namespace in ns_map.items():
                self.add_namespace(prefix, namespace)

    def add_namespace(self, prefix: str, namespace: str):
    #====================================================
        if (pfx := self.__reverse_map.get(namespace)) is not None:
            self.__prefix_dict.pop(pfx, None)
        if (ns := self.__prefix_dict.get(prefix)) is not None:
            self.__reverse_map.pop(ns, None)
        self.__prefix_dict[prefix] = namespace
        self.__reverse_map[namespace] = prefix

    def copy(self) -> 'NamespaceMap':
    #================================
        return NamespaceMap(self.__prefix_dict)

    def curie(self, uri) -> str:
    #===========================
        for prefix, namespace in self.__prefix_dict.items():
            if uri.startswith(namespace):
                return f'{prefix}:{uri[len(namespace):]}'
        return uri

    def delete_prefix(self, prefix: str):
    #====================================
        if (ns := self.__prefix_dict.pop(prefix, None)) is not None:
            self.__reverse_map.pop(ns, None)

    def merge_namespaces(self, other: Self) -> Self:
    #===============================================
        for prefix, namespace in other.__prefix_dict.items():
            self.add_namespace(prefix, namespace)
        return self

    def simplify(self, term):
    #========================
        if isinstance(term, URIRef):
            return self.curie(term)
        elif isinstance(term, BNode):
            return str(term)
        elif isinstance(term, Literal) and term.datatype is None:
            return str(term)
        return term

    def sparql_prefixes(self) -> str:
    #================================
        return '\n'.join([
            f'PREFIX {prefix}: <{namespace}>'
                for prefix, namespace in self.__prefix_dict.items()
        ])

    def uri(self, curie: str) -> URIRef:
    #===================================
        parts = curie.split(':', 1)
        if len(parts) == 2 and parts[0] in self.__prefix_dict:
            return URIRef(f'{self.__prefix_dict[parts[0]]}{parts[1]}')
        return URIRef(curie)

#===============================================================================
