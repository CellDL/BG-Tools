#===============================================================================

from typing import Any, Optional

#===============================================================================

class BlankNode:
    pass

#===============================================================================

class NamedNode:
    def __str__(self) -> str:
        return ''

    @property
    def fragment(self) -> str:
        return ''

#===============================================================================

class Literal:
    def __str__(self) -> str:
        return self.value

    @property
    def datatype(self) -> Optional[NamedNode]:
        pass

    @property
    def value(self) -> str:
        return ''

#===============================================================================

def blankNode(uri: str) -> BlankNode:
    return BlankNode()

def literal(value: str, datatype: Optional[NamedNode]=None) -> Literal:
    return Literal()

def namedNode(uri: str) -> NamedNode:
    return NamedNode()

#===============================================================================

def isBlankNode(node: Any) -> bool:
    return isinstance(node, BlankNode)

def isLiteral(node: Any) -> bool:
    return isinstance(node, Literal)

def isNamedNode(node: Any) -> bool:
    return isinstance(node, NamedNode)

#===============================================================================

class RdfStore:
    pass

#===============================================================================
#===============================================================================
