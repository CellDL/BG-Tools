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

from ..rdf import NamedNode
from ..rdf.namespace import Namespace

#===============================================================================

BGF = Namespace('https://bg-rdf.org/ontologies/bondgraph-framework#')
CDT = Namespace('https://w3id.org/cdt/')

#===============================================================================

NAMESPACES = {
    'bgf': 'https://bg-rdf.org/ontologies/bondgraph-framework#',
    'cdt': 'https://w3id.org/cdt/',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
}

def get_curie(uri: str|NamedNode) -> str:
#========================================
    full_uri = uri if isinstance(uri, str) else uri.value
    for prefix, ns_uri in NAMESPACES.items():
        if full_uri.startswith(ns_uri):
            return f'{prefix}:{full_uri[len(ns_uri):]}'
    return full_uri

#===============================================================================
