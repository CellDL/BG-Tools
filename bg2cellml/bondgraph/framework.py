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

from dataclasses import dataclass
from pathlib import Path
from typing import cast, Optional, Self

#===============================================================================

from rdflib.namespace import RDF, XSD

#===============================================================================

from ..rdf import BNode, Literal, RDFGraph, ResultType, URIRef
from ..units import Units, Value

from ..mathml import MathML
from .namespaces import BGF, CDT, NAMESPACES
from .utils import Labelled

#===============================================================================

BGF_FRAMEWORK_PATH = (Path( __file__).parent / '../../BG-RDF/').resolve()
BGF_ONTOLOGY = BGF_FRAMEWORK_PATH / 'schema/ontology.ttl'

BGF_TEMPLATE_PATH = BGF_FRAMEWORK_PATH / 'templates'
BGF_TEMPLATE_PREFIX = 'https://bg-rdf.org/templates/'

#===============================================================================

# Variable of integration

VOI_SYMBOL = 't'
VOI_UCUMUNIT = Literal('s', datatype=CDT.ucumunit)

#===============================================================================

DISSIPATOR         = BGF.Dissipator
FLOW_SOURCE        = BGF.FlowSource
FLOW_STORE         = BGF.FlowStore
POTENTIAL_SOURCE   = BGF.PotentialSource
QUANTITY_STORE     = BGF.QuantityStore

ONENODE_JUNCTION   = BGF.OneNode
TRANSFORM_JUNCTION = BGF.TransformNode  # Can act as a transformer of gyrator
ZERONODE_JUNCTION  = BGF.ZeroNode

#===============================================================================
#===============================================================================

TRANSFORM_RATIO_NAME     = 'RATIO'

TRANSFORM_FLOW_NAME      = 'v'
TRANSFORM_POTENTIAL_NAME = 'u'

TRANSFORM_PORT_IDS       = ['0', '1']

#===============================================================================

GYRATOR_EQUATIONS = MathML.from_string(f"""
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            </apply>
        </apply>
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            </apply>
        </apply>
    </math>
""")

#===============================================================================

TRANSFORMER_EQUATIONS = MathML.from_string(f"""
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>
            <eq/>
            <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <times/>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
                <ci>{TRANSFORM_POTENTIAL_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
            </apply>
        </apply>
        <apply>
            <eq/>
            <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[1]}</ci>
            <apply>
                <divide/>
                <ci>{TRANSFORM_FLOW_NAME}_{TRANSFORM_PORT_IDS[0]}</ci>
                <ci>{TRANSFORM_RATIO_NAME}</ci>
            </apply>
        </apply>
    </math>
""")

#===============================================================================
#===============================================================================

def optional_integer(value: ResultType, default: Optional[int]=None) -> Optional[int]:
#=====================================================================================
    if value is not None and isinstance(value, Literal) and value.datatype == XSD.integer:
        return int(value)
    return default

def clean_name(name: str) -> str:
#=================================
    return name.replace(':', '_').replace('-', '_').replace('.', '_')

#===============================================================================
#===============================================================================

class Variable:
    def __init__(self, element_uri: URIRef, name: str, units: Optional[Literal|Units]=None,
                                                       value: Optional[Literal]=None):
        self.__element_uri = element_uri
        self.__name = clean_name(name)
        self.__symbol: Optional[str] = None
        self.__units = Units.from_ucum(units) if isinstance(units, Literal) else units
        if value is not None:
            self.__value = Value.from_literal(value)
            if self.__units is None and self.__value.units is not None:
                self.__units = self.__value.units
        else:
            self.__value = None
        if self.__units is None:
            raise ValueError(f'Variable {name} for {element_uri} has no Units specified')
        if self.__value is not None:
            if self.__value.units is None:
                self.__value.set_units(self.__units)
            elif not self.__units.is_compatible_with(self.__value.units):
                raise ValueError(f'Value for variable {name} has incompatible units ({self.__value.units} != {self.__units})')

    def __str__(self):
        return f'{self.symbol} ({self.__value if self.__value is not None else self.__units})'

    @property
    def element_uri(self):
        return self.__element_uri

    @property
    def name(self):
        return self.__name

    @property
    def symbol(self) -> str:
        return self.__symbol if self.__symbol is not None else self.__name

    @property
    def value(self):
        return self.__value

    @property
    def units(self) -> Units:
        return self.__units         # type: ignore

    def copy(self, suffix: Optional[str]=None, domain: Optional['Domain']=None) -> 'Variable':
    #=========================================================================================
        if suffix is None:
            name = self.__name
        else:
            suffix = clean_name(suffix)
            if domain is None:
                name = f'{self.__name}_{suffix}'
            else:
                domain_symbols = domain.intrinsic_symbols
                suffix_parts = suffix.split('_')
                if self.name in domain_symbols and self.name == suffix_parts[0]:
                    name = suffix
                else:
                    name = f'{self.__name}_{suffix}'
        copy = Variable(self.__element_uri, name, units=self.__units)
        copy.__value = self.__value.copy() if self.__value is not None else None
        return copy

    def set_symbol(self, symbol: str):
    #=================================
        self.__symbol = symbol

    def set_value(self, value: Value):
    #=================================
        self.__value = value
        if self.__value.units is None:
            self.__value.set_units(self.__units)                        # type: ignore
        elif not self.__units.is_compatible_with(self.__value.units):   # type: ignore
            raise ValueError(
                f'Value for variable {self.__name} has incompatible units ({self.__units} != {self.__value.units})') # type: ignore

#===============================================================================

VOI_VARIABLE = Variable(URIRef(''), VOI_SYMBOL, units=VOI_UCUMUNIT)

#===============================================================================
#===============================================================================

DOMAIN_CONSTANTS = """
    SELECT DISTINCT ?name ?value
    WHERE {
        <%DOMAIN_URI%>
            a bgf:PhysicalDomain ;
            bgf:hasConstant [
                bgf:varName ?name ;
                bgf:hasValue ?value
            ] .
    }"""

#===============================================================================

class Domain(Labelled):
    def __init__(self, uri: URIRef, label: Optional[str],
                    flow_name: str, flow_units: Literal,
                    potential_name: str, potential_units: Literal,
                    quantity_name: str, quantity_units: Literal):
        super().__init__(uri, label)
        self.__flow = Variable(self.uri, flow_name, units=flow_units)
        self.__potential = Variable(self.uri, potential_name, units=potential_units)
        self.__quantity = Variable(self.uri, quantity_name, units=quantity_units)
        self.__intrinsic_symbols = [
            self.__flow.symbol,
            self.__potential.symbol,
            self.__quantity.symbol
        ]
        self.__constants: list[Variable] = []

    @classmethod
    def from_rdfgraph(cls, graph: RDFGraph,
                    uri: URIRef, label: Optional[str],
                    flow_name: str, flow_units: Literal,
                    potential_name: str, potential_units: Literal,
                    quantity_name: str, quantity_units: Literal) -> Self:
        self = cls(uri, label, flow_name, flow_units,
                                potential_name, potential_units,
                                quantity_name, quantity_units)
        self.__add_constants(graph)
        return self

    def __eq__(self, other):
        return self.uri == other.uri

    def __str__(self):
        return self.uri

    @property
    def constants(self):
        return self.__constants

    @property
    def flow(self):
        return self.__flow

    @property
    def intrinsic_symbols(self):
        return self.__intrinsic_symbols

    @property
    def name(self) -> str:
        return self.symbol

    @property
    def potential(self):
        return self.__potential

    @property
    def quantity(self):
        return self.__quantity

    def __add_constants(self, graph: RDFGraph):
    #==========================================
        self.__constants.extend([Variable(self.uri, str(row[0]), value=row[1])  # type: ignore
                                for row in graph.query(
                                        DOMAIN_CONSTANTS.replace('%DOMAIN_URI%', self.uri))])

#===============================================================================
#===============================================================================

@dataclass
class NamedPortVariable:
    name: str
    variable: Variable

    def __str__(self):
        return f'(name: {self.name}, variable: {self.variable})'

#===============================================================================

class PowerPort:
    def __init__(self, uri: URIRef, flow: NamedPortVariable, potential: NamedPortVariable):
        self.__uri = uri
        self.__flow = flow
        self.__potential = potential

    def __str__(self):
        return f'{self.__uri.fragment}, potential: {self.__potential}, flow: {self.__flow}'

    @property
    def flow(self) -> NamedPortVariable:
        return self.__flow

    @property
    def potential(self) -> NamedPortVariable:
        return self.__potential

    def copy(self, suffix: Optional[str]=None, domain: Optional[Domain]=None) -> 'PowerPort':
    #========================================================================================
        return PowerPort(self.__uri,
            NamedPortVariable(name=self.__flow.name, variable=self.__flow.variable.copy(suffix=suffix, domain=domain)),
            NamedPortVariable(name=self.__potential.name, variable=self.__potential.variable.copy(suffix=suffix, domain=domain))
        )

#===============================================================================
#===============================================================================

ELEMENT_PARAMETERS = """
    SELECT DISTINCT ?name ?units ?value
    WHERE {
        <%ELEMENT_URI%> bgf:hasParameter ?variable .
        ?variable bgf:varName ?name .
        OPTIONAL { ?variable bgf:hasUnits ?units }
        OPTIONAL { ?variable bgf:hasValue ?value }
    }"""

ELEMENT_VARIABLES = """
    SELECT DISTINCT ?name ?units ?value
    WHERE {
        <%ELEMENT_URI%> bgf:hasVariable ?variable .
        ?variable bgf:varName ?name .
        OPTIONAL { ?variable bgf:hasUnits ?units }
        OPTIONAL { ?variable bgf:hasValue ?value }
    }"""

#===============================================================================

ELEMENT_PORT_BONDS = """
    SELECT DISTINCT ?portId ?bondCount
    WHERE {
        {
            { <%ELEMENT_URI%> bgf:hasPort ?portId .
            }
        UNION {
            <%ELEMENT_URI%> bgf:hasPort ?port .
            ?port bgf:portId ?portId ;
            OPTIONAL { ?port bgf:bondCount ?bondCount }
            }
        }
    } ORDER BY ?portId"""

#===============================================================================

class ElementTemplate(Labelled):
    def __init__(self, uri: URIRef, element_class: URIRef,
                    label: Optional[str], domain: Domain, relation: str|Literal):
        super().__init__(uri, label)
        self.__element_class = element_class
        self.__domain = domain
        if element_class in [FLOW_SOURCE, POTENTIAL_SOURCE]:
            self.__relation = None
        else:
            mathml = None
            if isinstance(relation, Literal):
                if relation.datatype == BGF.mathml:
                    mathml = str(relation)
                else:
                    mathml = relation
            if mathml is None:
                raise ValueError(f'BondElement {uri} has no constitutive relation')
            try:
                self.__relation = MathML.from_string(mathml)
            except ValueError as error:
                raise ValueError(f'{self.uri}: {error}')
        self.__power_ports: dict[str, PowerPort] = {}
        self.__parameters: dict[str, Variable] = {}
        self.__variables: dict[str, Variable] = {}
        self.__intrinsic_variable: Optional[Variable] = None

    @classmethod
    def from_rdfgraph(cls, graph: RDFGraph, uri: URIRef, element_class: URIRef,
                        label: Optional[str], domain: Domain, relation: str|Literal) -> Self:
        self = cls(uri, element_class, label, domain, relation)
        self.__add_ports(graph)
        self.__add_variables(graph)
        self.__check_names()
        return self

    @property
    def constitutive_relation(self) -> Optional[MathML]:
        return self.__relation

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def element_class(self) -> URIRef:
        return self.__element_class

    @property
    def intrinsic_variable(self) -> Optional[Variable]:
        return self.__intrinsic_variable

    @property
    def parameters(self) -> dict[str, Variable]:
        return self.__parameters

    @property
    def power_ports(self) -> dict[str, PowerPort]:
        return self.__power_ports

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def __add_ports(self, graph: RDFGraph):
    #======================================
        port_bonds: dict[str, int|None] = {}
        for row in graph.query(
                        ELEMENT_PORT_BONDS.replace('%ELEMENT_URI%', self.uri)):
            if isinstance(row[0], Literal):
                port_bonds[str(row[0])] = optional_integer(row[1], 1)
        if len(port_bonds):
            flow_suffixed = (len(port_bonds) == 2) and (self.__element_class != DISSIPATOR)
            self.__power_ports = {}
            for id, bond_count in port_bonds.items():
                suffix = f'_{id}'
                flow_var = self.__port_name_variable(self.domain.flow, suffix if flow_suffixed else '')
                potential_var = self.__port_name_variable(self.domain.potential, suffix)
                self.__power_ports[id] = PowerPort(self.uri + suffix, flow_var, potential_var)
        else:
            self.__power_ports = {'': PowerPort(self.uri,
                                    self.__port_name_variable(self.domain.flow),
                                    self.__port_name_variable(self.domain.potential)
                                 )}

    def __port_name_variable(self, domain_variable: Variable, suffix: str='') -> NamedPortVariable:
    #==============================================================================================
        port_var_name = f'{domain_variable.name}{suffix}'
        return NamedPortVariable(name=port_var_name,
                                variable=Variable(self.uri, port_var_name, units=domain_variable.units))

    def __add_variables(self, graph: RDFGraph):
    #==========================================
        for row in graph.query(ELEMENT_PARAMETERS.replace('%ELEMENT_URI%', self.uri, True)):
            var_name = str(row[0])
            if var_name in self.__domain.intrinsic_symbols:
                raise ValueError(f'Cannot specify domain symbol {var_name} as a variable for {self.uri}')
            self.__parameters[var_name] = Variable(self.uri, str(row[0]), units=row[1], value=row[2])   # type: ignore
        for row in graph.query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri, True)):
            var_name = str(row[0])
            if var_name in self.__domain.intrinsic_symbols:
                raise ValueError(f'Cannot specify domain symbol {var_name} as a variable for {self.uri}')
            self.__variables[var_name] = Variable(self.uri, str(row[0]), units=row[1], value=row[2])   # type: ignore
        # A variable that is intrinsic to the element's class
        # Values of intrinsic variables are set by bgf:hasValue
        if self.__element_class == QUANTITY_STORE:
            self.__intrinsic_variable = self.__domain.quantity.copy()
        if self.__element_class == FLOW_STORE:
            self.__intrinsic_variable = self.__domain.flow.copy()
        elif self.__element_class == POTENTIAL_SOURCE:
            self.__intrinsic_variable = self.__domain.potential.copy()
        elif self.__element_class == FLOW_SOURCE:
            self.__intrinsic_variable = self.__domain.flow.copy()

    def __check_names(self):
    #=======================
        names = []
        def add_name(name: str, unique=True):
            if name not in names:
                names.append(name)
            elif unique:
                raise ValueError(f'Duplicate name `{name}` for {self.uri}')
        for name in self.__parameters.keys():
            add_name(name)
        for name in self.__variables.keys():
            add_name(name)
        eqn_names = self.__relation.variables if self.__relation is not None else []
        if len(names) > len(eqn_names):
            raise ValueError(f"{self.uri} has variables that are not in it's constitutive relation")
        for port in self.__power_ports.values():
            if port.flow is not None:
                add_name(port.flow.name, False)
            if port.potential is not None:
                add_name(port.potential.name, False)
        names.extend([c.name for c in self.__domain.constants])
        names.extend(self.__domain.intrinsic_symbols)
        names.append(VOI_VARIABLE.name)
        for name in eqn_names:
            if name not in names:
                raise ValueError(f'Constitutive relation of {self.uri} has undeclared name {name}')

#===============================================================================
#===============================================================================


class JunctionStructure(Labelled):
    def __init__(self, uri: URIRef, label: Optional[str]):
        super().__init__(uri, label)

#===============================================================================
#===============================================================================

class CompositeElement(Labelled):
    def __init__(self, uri: URIRef, template: ElementTemplate, junction: JunctionStructure, label: Optional[str]):
        super().__init__(uri, label)
        self.__template = template
        self.__junction = junction

    @property
    def junction(self):
        return self.__junction

    @property
    def template(self):
        return self.__template

#===============================================================================

class CompositeTemplate(Labelled):
    def __init__(self, uri: URIRef, template: ElementTemplate, label: Optional[str]):
        super().__init__(uri, label)
        self.__template = template

    @property
    def template(self):
        return self.__template

#===============================================================================
#===============================================================================

type BondgraphElementTemplate = ElementTemplate | CompositeTemplate

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

COMPOSITE_ELEMENT_DEFINITIONS = """
    SELECT DISTINCT ?uri ?template ?label
    WHERE {
        ?uri
            a bgf:CompositeElement ;
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

ELEMENT_IS_SOURCE = """
    SELECT DISTINCT ?bond
    WHERE {
        ?bond
            bgf:hasSource <%ELEMENT_URI%> ;
            bgf:hasTarget ?target1 .
    }"""

ELEMENT_IS_SOURCE_PORT = """
    SELECT DISTINCT ?bond ?bnode
    WHERE {
        ?bond
            bgf:hasSource ?bnode ;
            bgf:hasTarget ?target .
        ?bnode
            bgf:element <%ELEMENT_URI%> ;
            bgf:port "%PORT_ID%" .
    }"""

ELEMENT_IS_TARGET = """
    SELECT DISTINCT ?bond
    WHERE {
        ?bond
            bgf:hasSource ?source ;
            bgf:hasTarget <%ELEMENT_URI%> .
    } ORDER BY ?bond"""

ELEMENT_IS_TARGET_PORT = """
    SELECT DISTINCT ?bond ?bnode
    WHERE {
        ?bond
            bgf:hasSource ?source ;
            bgf:hasTarget ?bnode .
        ?bnode
            bgf:element <%ELEMENT_URI%> ;
            bgf:port "%PORT_ID%" .
    } ORDER BY ?bond"""

#===============================================================================

class _BondgraphFramework:
    _instance = None

    def __new__(cls, *args, **kwargs):
    #=================================
        if cls._instance is None:
            cls._instance = super(_BondgraphFramework, cls).__new__(cls)
        return cls._instance

    def __init__(self, bgf_ontology: str|Path, bgf_templates: Optional[list[Path|URIRef]]=None):
    #===========================================================================================
        self.__ontology = RDFGraph(NAMESPACES)
        self.__ontology.parse(bgf_ontology)
        self.__element_templates: dict[URIRef, ElementTemplate] = {}
        self.__domains: dict[URIRef, Domain] = {}
        self.__element_domains: dict[tuple[URIRef, URIRef], ElementTemplate] = {}
        self.__junctions: dict[URIRef, JunctionStructure] = {}
        self.__composite_elements: dict[URIRef, CompositeTemplate] = {}
        self.__loaded_templates = set()
        if bgf_templates is not None:
            for bgf_template in bgf_templates:
                self.add_template(bgf_template)

    def add_template(self, bgf_template: Path|URIRef):
    #=================================================
        if isinstance(bgf_template, Path):
            template_uri = bgf_template.resolve().as_uri()
        elif bgf_template.startswith(BGF_TEMPLATE_PREFIX):
            template_uri = (BGF_TEMPLATE_PATH / bgf_template[len(BGF_TEMPLATE_PREFIX):]).as_uri()
        else:
            template_uri = bgf_template
        if template_uri not in self.__loaded_templates:
            self.__loaded_templates.add(template_uri)
            graph = RDFGraph(NAMESPACES)
            graph.merge(self.__ontology)
            if not graph.parse(template_uri):
                return
            self.__domains.update({cast(URIRef, row[0]): Domain.from_rdfgraph(
                                        graph, row[0], row[1],                          # pyright: ignore[reportArgumentType]
                                        row[2], row[3], row[4], row[5], row[6], row[7]) # pyright: ignore[reportArgumentType]
                                    for row in graph.query(DOMAIN_QUERY)})
            for row in graph.query(ELEMENT_TEMPLATE_DEFINITIONS):
                if (domain := self.__domains.get(cast(URIRef, row[3]))) is None:
                    raise ValueError(f'Unknown domain {row[3]} for {row[0]} element')
                self.__element_templates[cast(URIRef, row[0])] = ElementTemplate.from_rdfgraph(
                                                graph, row[0], row[1], row[2],          # pyright: ignore[reportArgumentType]
                                                domain, row[4])                         # pyright: ignore[reportArgumentType]
            self.__element_domains.update({
                (element.element_class, element.domain.uri): element
                    for element in self.__element_templates.values() if element.domain is not None
            })
            self.__junctions.update({cast(URIRef, row[0]): JunctionStructure(row[0], row[1])    # pyright: ignore[reportArgumentType]
                for row in graph.query(JUNCTION_STRUCTURES)})
            for row in graph.query(COMPOSITE_ELEMENT_DEFINITIONS):
                if (element := self.__element_templates.get(cast(URIRef, row[1]))) is None:
                    raise ValueError(f'Unknown BondElement {row[1]} for composite {row[0]}')
                #if (junction := self.__junctions.get(row[2])) is None:          # type: ignore
                #    raise ValueError(f'Unknown JunctionStructure {row[2]} for composite {row[0]}')
                self.__composite_elements[row[0]] = CompositeTemplate(row[0], element, row[2]) # pyright: ignore[reportArgumentType]

    def element_template(self, element_type: URIRef, domain_uri: Optional[URIRef]) -> Optional[BondgraphElementTemplate]:
    #====================================================================================================================
        if domain_uri is None:
            # First see if element_type refers to a composite
            if (composite := self.__composite_elements.get(element_type)) is not None:
                return composite
            return self.__element_templates.get(element_type)
        else:
            return self.__element_domains.get((element_type, domain_uri))

    def junction(self, uri: URIRef) -> Optional[JunctionStructure]:
    #==============================================================
        return self.__junctions.get(uri)

    def junction_classes(self) -> list[str]:
    #=======================================
        return list(self.__junctions.keys())

#===============================================================================
#===============================================================================

BondgraphFramework = _BondgraphFramework(BGF_ONTOLOGY)

#===============================================================================
#===============================================================================
