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

from typing import Optional

#===============================================================================

from ..rdf import isNamedNode, NamedNode, uri_fragment

#===============================================================================

LOCAL_MODEL_BASE = 'https://bg-rdf.org/models/local/'

#===============================================================================

def pretty_uri(uri: Optional[str|NamedNode]) -> str:
#=====================================================
    if uri is not None:
        uri_text: str = uri.value if isNamedNode(uri) else uri  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        if uri_text.startswith(LOCAL_MODEL_BASE):
            pretty = uri_text[len(LOCAL_MODEL_BASE):]
        else:
            parts = uri_text.split('#', 1)
            if len(parts) > 1:
                pretty = '#' + parts[1]
            else:
                pretty = uri_text
    else:
        pretty = 'None'
    return pretty

#===============================================================================

class Labelled:
    def __init__(self, uri: NamedNode, symbol: Optional[str]=None, label: Optional[str]=None):
        self.__uri = uri.value
        if symbol is not None:
            self.__symbol = symbol
        else:
            self.__symbol = uri_fragment(self.__uri)
        self.__label = label

    def __str__(self) -> str:
        if self.__label is not None:
            return f'{str(self.__uri)} ({self.__label})'
        return str(self.__uri)

    @property
    def curie(self):
        return f':{uri_fragment(self.__uri)}'

    @property
    def label(self):
        return self.__label

    @property
    def symbol(self):
        return self.__symbol

    @property
    def uri(self) -> str:
        return self.__uri

#===============================================================================
#===============================================================================
