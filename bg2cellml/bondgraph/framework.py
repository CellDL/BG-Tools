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

import asyncio
from pathlib import Path
import sys
from typing import cast

#===============================================================================

from ..rdf import NamedNode, RdfGraph
from ..utils import Issue, make_issue

from .model import BondgraphModel
from .namespaces import NAMESPACES

from .framework_support import BondgraphElementTemplate, CompositeTemplate, ElementTemplate
from .framework_support import Domain, JunctionStructure

#===============================================================================

BGF_ONTOLOGY_URI = 'https://bg-rdf.org/ontologies/bondgraph-framework'
BGF_TEMPLATE_URI_PREFIX = 'https://bg-rdf.org/templates/'

_BGF_TEMPLATE_NAMES = [
    'chemical.ttl',
    'electrical.ttl',
    'hydraulic.ttl',
    'mechanical.ttl',
]

BGF_TEMPLATE_URIS: dict[str, str] = {
    (BGF_TEMPLATE_URI_PREFIX + name): name
        for name in _BGF_TEMPLATE_NAMES
}

#===============================================================================

# Determine where we can get the bundle BG-RDF framework

_browser = 'pyodide' in sys.modules
_packaged = 'site-packages' in __file__

#===============================================================================

if _browser:
    import pyodide.http         # pyright: ignore[reportMissingImports]

    async def _get_bgrdf(path: str) -> str:
        response = await pyodide.http.pyfetch(path)
        if response.ok:
            rdf = await response.text()
            return rdf
        raise Issue('Cannot fetch RDF from {path}: {response.status_text}')

    async def get_ontology(base: str='/') -> str:
        rdf = await _get_bgrdf(f'{base}bg-rdf/ontology.ttl')
        return rdf

    async def get_template(template: str, base: str='/') -> str:
        rdf = await _get_bgrdf(f'{base}bg-rdf/templates/{template}')
        return rdf

elif _packaged:
    import importlib.resources

    BGF_FRAMEWORK_PATH = Path('BG-RDF')

    async def get_ontology(base: str='/') -> str:
        return importlib.resources.read_text('bg2cellml', BGF_FRAMEWORK_PATH / 'schema/ontology.ttl')

    async def get_template(template: str, base: str='/') -> str:
        return importlib.resources.read_text('bg2cellml', BGF_FRAMEWORK_PATH / 'templates' / template)

else:
    BGF_FRAMEWORK_PATH = (Path(__file__).parent / '../../BG-RDF/').resolve()

    async def get_ontology(base: str='/') -> str:
        with open(BGF_FRAMEWORK_PATH / 'schema/ontology.ttl') as fp:
            return fp.read()

    async def get_template(template: str, base: str='/') -> str:
        with open(BGF_FRAMEWORK_PATH / 'templates' / template) as fp:
            return fp.read()

#===============================================================================
#===============================================================================

DOMAIN_QUERY = """
    SELECT DISTINCT ?domain ?label
                    ?flowName ?flowUnits
                    ?potentialName ?potentialUnits
                    ?quantityName ?quantityUnits
    WHERE {
        ?domain
            a bgf:PhysicalDomain ;
            bgf:hasFlow [
                bgf:varName ?flowName ;
                bgf:hasUnits ?flowUnits
            ] ;
            bgf:hasPotential [
                bgf:varName ?potentialName ;
                bgf:hasUnits ?potentialUnits
            ] ;
            bgf:hasQuantity [
                bgf:varName ?quantityName ;
                bgf:hasUnits ?quantityUnits
            ] .
        OPTIONAL { ?domain rdfs:label ?label }
    } ORDER BY ?domain"""

ELEMENT_TEMPLATE_DEFINITIONS = """
    SELECT DISTINCT ?uri ?element_class ?label ?domain ?relation
    WHERE {
        ?uri
            a bgf:ElementTemplate ;
            rdfs:subClassOf ?element_class ;
            rdfs:subClassOf* bgf:BondElement ;
            bgf:hasDomain ?domain .
        OPTIONAL { ?uri bgf:constitutiveRelation ?relation }
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

JUNCTION_STRUCTURES = """
    SELECT DISTINCT ?junction ?label
    WHERE {
        ?junction rdfs:subClassOf* bgf:JunctionStructure .
        OPTIONAL { ?junction rdfs:label ?label }
    } ORDER BY ?junction"""

COMPOSITE_TEMPLATE_DEFINITIONS = """
    SELECT DISTINCT ?uri ?template ?label
    WHERE {
        ?uri
            a bgf:CompositeTemplate ;
            rdfs:subClassOf* ?template .
        ?template
            a bgf:ElementTemplate .
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

#===============================================================================

'''
COMPOSITE_ELEMENTS = """
    SELECT DISTINCT ?model ?elementUri ?elementType
    WHERE {
        ?model
            a bgf:BondgraphModel ;
            bgf:hasBondElement ?elementUri .
        ?elementUri a ?elementType .
    } ORDER BY ?model ?elementUri"""
'''

#===============================================================================
#===============================================================================

class BondgraphFramework:
    _instance = None

    def __new__(cls, *args, **kwds):
    #===============================
        if cls._instance is None:
            cls._instance = super(BondgraphFramework, cls).__new__(cls)
        return cls._instance

    def __init__(self):
    #==================
        self.__ontology_graph = RdfGraph(NAMESPACES)
        self.__element_templates: dict[str, ElementTemplate] = {}
        self.__domains: dict[str, Domain] = {}
        self.__element_domains: dict[tuple[str, str], ElementTemplate] = {}
        self.__junctions: dict[str, JunctionStructure] = {}
        self.__composite_elements: dict[str, CompositeTemplate] = {}
        self.__issues: list[Issue] = []
        self.__framework_loaded = -1

    @property
    def framework_loaded(self):
        return self.__framework_loaded > 0

    async def load_framework(self, base: str='/'):
    #=============================================
        if self.__framework_loaded >= 0:
            while self.__framework_loaded == 0:
                await asyncio.sleep(0.01)
            return
        self.__framework_loaded = 0
        try:
            ontology = await get_ontology(base=base)
            self.__ontology_graph.load(BGF_ONTOLOGY_URI, ontology)
            for uri, template_name in BGF_TEMPLATE_URIS.items():
                template = await get_template(template_name, base=base)
                self.__add_template(uri, template)
            self.__framework_loaded = 1
        except Exception as e:
            self.__issues.append(make_issue(e))

    @property
    def has_issues(self) -> bool:
    #============================
        return len(self.__issues) > 0

    @property
    def issues(self) -> list[Issue]:
    #===============================
        return self.__issues

    def report_issue(self, reason: str):
    #===================================
        self.__issues.append(Issue(reason))

    def add_template(self, uri: str, template: str) -> bool:
    #=======================================================
        try:
            self.__add_template(uri, template)
            return True
        except Exception as e:
            self.__issues.append(make_issue(e))
        return False

    def __add_template(self, uri: str, template: str):
    #=================================================
        graph = RdfGraph(NAMESPACES)
        graph.merge(self.__ontology_graph)
        graph.load(uri, template)
        self.__domains.update({cast(NamedNode, row['domain']).value: Domain.from_rdf_graph(
                                    graph, row['domain'], row.get('label'),         # pyright: ignore[reportArgumentType]
                                    row['flowName'], row['flowUnits'],              # pyright: ignore[reportArgumentType]
                                    row['potentialName'], row['potentialUnits'],    # pyright: ignore[reportArgumentType]
                                    row['quantityName'], row['quantityUnits'])      # pyright: ignore[reportArgumentType]
        # ?domain ?label ?flowName ?flowUnits ?potentialName ?potentialUnits ?quantityName ?quantityUnits
                                for row in graph.query(DOMAIN_QUERY)})
        for row in graph.query(ELEMENT_TEMPLATE_DEFINITIONS):
            # ?uri ?element_class ?label ?domain ?relation
            if (domain := self.__domains.get(cast(NamedNode, row['domain']).value)) is None:
                self.report_issue(f"Unknown domain {row['domain']} for {row['uri']} element")
                continue
            self.__element_templates[cast(NamedNode, row['uri']).value] = ElementTemplate.from_rdf_graph(
                                            graph, row['uri'], row['element_class'], # pyright: ignore[reportArgumentType]
                                            row.get('label'), domain, row.get('relation'))   # pyright: ignore[reportArgumentType]
        self.__element_domains.update({
            (element.element_class, element.domain.uri.value): element
                for element in self.__element_templates.values() if element.domain is not None
        })
        self.__junctions.update({cast(NamedNode, row['junction']).value: JunctionStructure(row['junction'], row.get('label'))    # pyright: ignore[reportArgumentType]
            # ?junction ?label
            for row in graph.query(JUNCTION_STRUCTURES)})
        for row in graph.query(COMPOSITE_TEMPLATE_DEFINITIONS):
            # ?uri ?template ?label
            if (element := self.__element_templates.get(cast(NamedNode, row['template']).value)) is None:
                self.report_issue(f"Unknown BondElement {row['template']} for composite {row['uri']}")
                continue
            self.__composite_elements[row['uri'].value] = CompositeTemplate(row['uri'], element, row.get('label')) # pyright: ignore[reportArgumentType]

    def element_template(self, element_type: NamedNode, domain_uri: NamedNode|None) -> BondgraphElementTemplate|None:
    #================================================================================================================
        if domain_uri is None:
            # First see if element_type refers to a composite
            if (composite := self.__composite_elements.get(element_type.value)) is not None:
                return composite
            return self.__element_templates.get(element_type.value)
        else:
            return self.__element_domains.get((element_type.value, domain_uri.value))

    def junction(self, uri: str) -> JunctionStructure|None:
    #======================================================
        return self.__junctions.get(uri)

#===============================================================================

    def make_bondgraph_model(self, base_iri: str, rdf_source: str, debug=False) -> BondgraphModel:
    #=============================================================================================
        return BondgraphModel(self, base_iri, rdf_source, debug)

#===============================================================================
#===============================================================================

async def get_framework(base: str='/') -> BondgraphFramework:
    framework = BondgraphFramework()
    await framework.load_framework(base=base)
    return framework

#===============================================================================
#===============================================================================
