from typing import Any, Iterator, Optional, Sequence

#===============================================================================

import pyoxigraph as oxigraph

#===============================================================================

BlankNode = oxigraph.BlankNode
Literal = oxigraph.Literal
NamedNode = oxigraph.NamedNode

#===============================================================================

def blankNode(value: Optional[str]=None) -> BlankNode:
    return oxigraph.BlankNode(value)

def literal(value: str|int|float|bool, datatype: Optional[NamedNode]=None) -> Literal:
    return oxigraph.Literal(value, datatype=datatype)

def namedNode(uri: str) -> NamedNode:
    return oxigraph.NamedNode(uri)

#===============================================================================

def isBlankNode(node: Any) -> bool:
    return isinstance(node, oxigraph.BlankNode)

def isLiteral(node: Any) -> bool:
    return isinstance(node, oxigraph.Literal)

def isNamedNode(node: Any) -> bool:
    return isinstance(node, oxigraph.NamedNode)

#===============================================================================

type ResultType = dict[str, BlankNode | Literal | NamedNode]

#===============================================================================

class RdfStore:
    def __init__(self):
        self.__graph = oxigraph.Store()

    def contains(self, s, p, o) -> bool:
        try:
            self.__graph.quads_for_pattern(s, p, o).__next__()
            return True
        except StopIteration:
            return False

    def add(self, s, p, o):
        self.__graph.add(oxigraph.Quad(s, p, o))

    def statements(self) -> Iterator[oxigraph.Quad]:
        return self.__graph.quads_for_pattern(None, None, None)

    def load(self, rdf: str, base_iri: Optional[str]=None):
        self.__graph.load(input=rdf, base_iri=base_iri, format=oxigraph.RdfFormat.TURTLE)

    def query(self, sparql: str) -> oxigraph.QuerySolutions:
        return self.__graph.query(sparql)    # type: ignore

#===============================================================================
#===============================================================================
