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

from typing import Any, Optional, TYPE_CHECKING

#===============================================================================

import lxml.etree as etree
import structlog
from structlog.dev import BRIGHT, GREEN, RESET_ALL

#===============================================================================

if TYPE_CHECKING:
    from ..rdf import URIRef

#===============================================================================

LOCAL_MODEL_BASE = 'https://bg-rdf.org/models/local/'

#===============================================================================
#===============================================================================

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(colors=True)
    ]
)

log = structlog.get_logger()

#===============================================================================
#===============================================================================

def pretty_log(s: Any) -> str:
#=============================
    return f'{RESET_ALL}{GREEN}{str(s)}{RESET_ALL}{BRIGHT}'

def pretty_uri(uri: Optional['str|URIRef']) -> str:
#==================================================
    if uri is not None:
        uri = str(uri)
        if uri.startswith(LOCAL_MODEL_BASE):
            pretty = uri[len(LOCAL_MODEL_BASE):]
        else:
            parts = uri.split('#', 1)
            if len(parts) > 1:
                pretty = '#' + parts[1]
            else:
                pretty = uri
    else:
        pretty = 'None'
    return pretty_log(pretty)

#===============================================================================
#===============================================================================

"""
Generate URIs for lxml.etree.
"""
class XMLNamespace:
    def __init__(self, ns: str):
        self.__ns = ns

    def __str__(self):
        return self.__ns

    def __call__(self, attr: str='') -> str:
        return f'{{{self.__ns}}}{attr}'

    def __getattr__(self, attr: str) -> str:
        return f'{{{self.__ns}}}{attr}'

#===============================================================================

def etree_from_string(xml: str) -> etree.Element:
    parser = etree.XMLParser(remove_blank_text=True)
    try:
        return etree.fromstring(xml, parser)
    except etree.XMLSyntaxError as error:
        raise ValueError(error)

#===============================================================================
