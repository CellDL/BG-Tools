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

from typing import NamedTuple, Optional, Self

#===============================================================================

from rdflib.namespace import RDF, XSD

#===============================================================================

from ..rdf import BNode, Literal, RDFGraph, URIRef
from ..units import Units, Value

from ..mathml import MathML
from .namespaces import BGF, CDT, NAMESPACES
from .utils import Labelled

#===============================================================================

# Variable of integration

VOI_SYMBOL = 't'
VOI_UCUMUNIT = Literal('s', datatype=CDT.ucumunit)

#===============================================================================

DISSIPATOR_ELEMENT = BGF.Dissipator
POTENTIAL_STORE    = BGF.PotentialStore

ONENODE_JUNCTION   = BGF.OneNode
TRANSFORM_JUNCTION = BGF.TransformNode  # Can act as a transformer of gyrator
ZERONODE_JUNCTION  = BGF.ZeroNode

#===============================================================================

def optional_integer(value: Optional[Literal], default: Optional[int]=None) -> Optional[int]:
#============================================================================================
    if value is not None and value.datatype == XSD.integer:
        return int(value)
    return default

def clean_name(name: str) -> str:
#=================================
    return name.replace(':', '_').replace('-', '_').replace('.', '_')

#===============================================================================
#===============================================================================

class Variable:
    def __init__(self, element_uri: URIRef, name: str, units: Optional[Literal|Units], value: Optional[Literal]):
        self.__element_uri = element_uri
        self.__name = clean_name(name)
        self.__symbol = None
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
        return f'{self.symbol} ({self.__value if self.__value is not None else ''}{self.__units})'

    @property
    def element_uri(self):
        return self.__element_uri

    @property
    def name(self):
        return self.__name

    @property
    def symbol(self):
        return self.__symbol if self.__symbol is not None else self.__name

    @property
    def value(self):
        return self.__value

    @property
    def units(self) -> Units:
        return self.__units         # type: ignore

    def copy(self, suffix: Optional[str]=None, strip_name: bool=False) -> 'Variable':
    #================================================================================
        if strip_name and suffix is None:
            raise ValueError(f'Cannot strip name of variable {self.__name} if no suffix provided')
        elif suffix is None:
            name = self.__name
        elif strip_name:
            name = clean_name(suffix)
        else:
            name = f'{self.__name}_{clean_name(suffix)}'
        copy = Variable(self.__element_uri, name, self.__units, None)
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

VOI_VARIABLE = Variable(URIRef(''), VOI_SYMBOL, VOI_UCUMUNIT, None)

#===============================================================================
#===============================================================================

DOMAIN_CONSTANTS = """
    SELECT DISTINCT ?name ?value
    WHERE {
        <%DOMAIN_URI%>
            a bgf:ModellingDomain ;
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
        self.__flow = Variable(self.uri, flow_name, flow_units, None)
        self.__potential = Variable(self.uri, potential_name, potential_units, None)
        self.__quantity = Variable(self.uri, quantity_name, quantity_units, None)
        self.__intrinsic_symbols = [
            self.__flow.symbol,
            self.__potential.symbol,
            self.__quantity.symbol
        ]
        self.__constants: list[Variable] = []

    @classmethod
    def from_framework(cls, framework: '_BondgraphFramework',
                    uri: URIRef, label: Optional[str],
                    flow_name: str, flow_units: Literal,
                    potential_name: str, potential_units: Literal,
                    quantity_name: str, quantity_units: Literal) -> Self:
        self = cls(uri, label, flow_name, flow_units,
                                potential_name, potential_units,
                                quantity_name, quantity_units)
        self.__add_constants(framework)
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
        return self.__uri.fragment

    @property
    def potential(self):
        return self.__potential

    @property
    def quantity(self):
        return self.__quantity

    def __add_constants(self, framework: '_BondgraphFramework'):
    #===========================================================
        self.__constants.extend([Variable(self.uri, str(row[0]), None, row[1])  # type: ignore
                                for row in framework.knowledge.query(
                                        DOMAIN_CONSTANTS.replace('%DOMAIN_URI%', self.uri))])

#===============================================================================
#===============================================================================

ELEMENT_PORT_IDS = """
    SELECT DISTINCT ?port
    WHERE {
        <%ELEMENT_URI%> bgf:hasPort ?port .
    }
    ORDER BY ?port"""

#===============================================================================

class PortNameVariable(NamedTuple):
    name: str
    variable: Variable

    def __str__(self):
        return f'(name: {self.name}, variable: {self.variable})'

#===============================================================================

class PowerPort:
    def __init__(self, element: 'ElementTemplate', port_suffix: Optional[str]=None,
            flow_suffixed=True, potential_suffixed=True):
        self.__element = element
        self.__suffix = '' if port_suffix is None else f'_{port_suffix}'
        self.__flow = self.__name_variable(element.domain.flow, flow_suffixed)
        self.__potential = self.__name_variable(element.domain.potential, potential_suffixed)

    def __str__(self):
        return f'{self.__element.uri}{self.__suffix}, potential: {self.__potential}, flow: {self.__flow}'

    @property
    def element(self):
        return self.__element

    @property
    def flow(self) -> PortNameVariable:
        return self.__flow

    @property
    def potential(self) -> PortNameVariable:
        return self.__potential

    def copy(self, suffix: Optional[str]=None) -> 'PowerPort':
    #=========================================================
        copy = PowerPort(self.__element)
        copy.__suffix = self.__suffix
        copy.__flow = PortNameVariable(name=self.__flow.name,
                                         variable=self.__flow.variable.copy(suffix))
        copy.__potential = PortNameVariable(name=self.__potential.name,
                                              variable=self.__potential.variable.copy(suffix))
        return copy

    def __name_variable(self, domain_variable: Variable, add_suffix: bool) -> PortNameVariable:
    #==========================================================================================
        name = f'{domain_variable.name}{self.__suffix}' if add_suffix else domain_variable.name
        return PortNameVariable(name=name,
                                variable=Variable(self.__element.uri, name, domain_variable.units, None))

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = """
    SELECT DISTINCT ?name ?units ?value
    WHERE {
        <%ELEMENT_URI%> bgf:hasVariable ?variable .
        ?variable bgf:varName ?name .
        OPTIONAL { ?variable bgf:hasUnits ?units }
        OPTIONAL { ?variable bgf:hasValue ?value }
    }"""

#===============================================================================

class ElementTemplate(Labelled):
    def __init__(self, uri: URIRef, element_class: URIRef,
                    label: Optional[str], domain: Domain, relation: str|Literal):
        super().__init__(uri, label)
        self.__element_class = element_class
        self.__domain = domain
        mathml = None
        if isinstance(relation, Literal):
            if relation.datatype == BGF.mathml:
                mathml = str(relation)
        else:
            # Do we insist on datatyping? Default to MathML ??
            mathml = relation
        if mathml is None:
            raise ValueError(f'BondElement {uri} has no constitutive relation')
        try:
            self.__relation = MathML.from_string(mathml)
        except ValueError as error:
            raise ValueError(f'{self.uri}: {error}')
        self.__ports: dict[str, PowerPort] = {}
        self.__variables: dict[str, Variable] = {}
        self.__intrinsic_variable: Optional[Variable] = None

    @classmethod
    def from_framework(cls, framework: '_BondgraphFramework', uri: URIRef, element_class: URIRef,
                        label: Optional[str], domain_uri: URIRef, relation: str|Literal) -> Self:
        if (domain := framework.domain(domain_uri)) is None:
            raise ValueError(f'Unknown domain {domain_uri} for {uri} element')
        self = cls(uri, element_class, label, domain, relation)
        self.__add_ports(framework)
        self.__add_variables(framework)
        self.__check_names()
        return self

    @property
    def constitutive_relation(self) -> MathML:
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
    def ports(self) -> dict[str, PowerPort]:
        return self.__ports

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def __add_ports(self, framework: '_BondgraphFramework'):
    #=======================================================
        port_ids = [str(row[0]) for row in framework.knowledge.query(
                        ELEMENT_PORT_IDS.replace('%ELEMENT_URI%', self.uri))]
        if len(port_ids):
            same_flow = (len(port_ids) == 2) and (self.__element_class != DISSIPATOR_ELEMENT)
            self.__ports = {id: PowerPort(self, port_suffix=id, flow_suffixed=same_flow)
                                for id in port_ids}
        else:
            self.__ports = {'': PowerPort(self)}

    def __add_variables(self, framework: '_BondgraphFramework'):
    #===========================================================
        for row in framework.knowledge.query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', self.uri)):
            var_name = str(row[0])
            if var_name in self.__domain.intrinsic_symbols:
                raise ValueError(f'Cannot specify domain symbol {var_name} as a variable for {self.uri}')
            self.__variables[var_name] = Variable(self.uri, str(row[0]), row[1], row[2])   # type: ignore
        # A variable that is intrinsic to the element's class
        if self.__element_class == POTENTIAL_STORE:
            self.__intrinsic_variable = self.__domain.quantity.copy()

    def __check_names(self):
    #=======================
        names = []
        def add_name(name: str, unique=True):
            if name not in names:
                names.append(name)
            elif unique:
                raise ValueError(f'Duplicate name `{name}` for {self.uri}')
        for name in self.__variables.keys():
            add_name(name)
        for port in self.__ports.values():
            add_name(port.flow.name, False)
            add_name(port.potential.name, False)
        eqn_names = self.__relation.variables
        if len(names) > len(eqn_names):
            raise ValueError(f"{self.uri} has variables that are not in it's constitutive relation")
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
#===============================================================================

DOMAIN_QUERY = """
    SELECT DISTINCT ?domain ?label
                    ?flowName ?flowUnits
                    ?potentialName ?potentialUnits
                    ?quantityName ?quantityUnits
    WHERE {
        ?domain
            a bgf:ModellingDomain ;
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
            bgf:hasDomain ?domain ;
            bgf:constitutiveRelation ?relation .
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

JUNCTION_STRUCTURES = """
    SELECT DISTINCT ?junction ?label
    WHERE {
        ?junction rdfs:subClassOf* bgf:JunctionStructure .
        OPTIONAL { ?junction rdfs:label ?label }
    } ORDER BY ?junction"""

COMPOSITE_ELEMENT_DEFINITIONS = """
    SELECT DISTINCT ?uri ?template ?junction ?label
    WHERE {
        ?uri
            a bgf:CompositeElement ;
            bgf:elementTemplate ?template ;
            bgf:junctionStructure ?junction .
        OPTIONAL { ?uri rdfs:label ?label }
    } ORDER BY ?uri"""

#===============================================================================

COMPOSITE_ELEMENTS = """
    SELECT DISTINCT ?model ?elementUri ?elementType
    WHERE {
        ?model
            a bgf:BondgraphModel ;
            bgf:hasBondElement ?elementUri .
        ?elementUri a ?elementType .
    } ORDER BY ?model ?elementUri"""

MODEL_BONDS = """
    SELECT DISTINCT ?bond ?source ?target
    WHERE {
        ?bond
            bgf:hasSource ?source ;
            bgf:hasTarget ?target .
    } ORDER BY ?bond"""

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
        if cls._instance is None:
            cls._instance = super(_BondgraphFramework, cls).__new__(cls)
        return cls._instance

    def __init__(self, bg_knowledge: list[str]):
        self.__knowledge = RDFGraph(NAMESPACES)
        for knowledge in bg_knowledge:
            self.__knowledge.parse(knowledge)
        self.__domains = {row[0]: Domain.from_framework(
                                    self, row[0], row[1],                               # type: ignore
                                    row[2], row[3], row[4], row[5], row[6], row[7])     # type: ignore
                                for row in self.__knowledge.query(DOMAIN_QUERY)}
        self.__element_templates: dict[URIRef, ElementTemplate] = {                     # type: ignore
            row[0]: ElementTemplate.from_framework(self, row[0], row[1], row[2],        # type: ignore
                                                   row[3], row[4])                      # type: ignore
                for row in self.__knowledge.query(ELEMENT_TEMPLATE_DEFINITIONS)}
        self.__element_domains: dict[tuple[URIRef, URIRef], ElementTemplate] = {
            (element.element_class, element.domain.uri): element
                for element in self.__element_templates.values() if element.domain is not None
        }
        self.__junctions: dict[URIRef, JunctionStructure] = {           # type: ignore
            row[0]: JunctionStructure(row[0], row[1])                   # type: ignore
                for row in self.__knowledge.query(JUNCTION_STRUCTURES)}

        self.__composite_elements: dict[URIRef, CompositeElement] = {}
        for row in self.__knowledge.query(COMPOSITE_ELEMENT_DEFINITIONS):
            if (element := self.__element_templates.get(row[1])) is None:   # type: ignore
                raise ValueError(f'Unknown BondElement {row[1]} for composite {row[0]}')
            if (junction := self.__junctions.get(row[2])) is None:          # type: ignore
                raise ValueError(f'Unknown JunctionStructure {row[2]} for composite {row[0]}')
            self.__composite_elements[row[0]] = CompositeElement(row[0], element, junction, row[3]) # type: ignore

    @property
    def knowledge(self):
        return self.__knowledge

    def domain(self, uri: URIRef) -> Optional[Domain]:
    #=================================================
        return self.__domains.get(uri)

    def element_template(self, element_type: URIRef, domain_uri: Optional[URIRef]) -> Optional[ElementTemplate]:
    #========================================================================================================
        if domain_uri is None:
            return self.__element_templates.get(element_type)
        else:
            return self.__element_domains.get((element_type, domain_uri))

    def generate_bonds(self, model_uri: URIRef, model_graph: RDFGraph):
    #==================================================================
        for row in model_graph.query(MODEL_BONDS):
            if (row[1], None, None) in model_graph and (row[2], None, None) in model_graph:
                model_graph.add((model_uri, BGF.hasPowerBond, row[0]))

    def junction(self, uri: URIRef) -> Optional[JunctionStructure]:
    #==============================================================
        return self.__junctions.get(uri)

    def junction_classes(self) -> list[str]:
    #=======================================
        return list(self.__junctions.keys())

    def resolve_composites(self, model_uri: URIRef, model_graph: RDFGraph):
    #======================================================================
        for row in model_graph.query(COMPOSITE_ELEMENTS):
            if (composite := self.__composite_elements.get(row[2])) is not None:    # type: ignore
                element_uri: URIRef = row[1]                                        # type: ignore
                model_graph.add((element_uri, RDF.type, composite.template.uri))
                model_graph.remove((element_uri, RDF.type, row[2]))
                if composite.junction.uri == ONENODE_JUNCTION:
                    junction_symbol = composite.template.domain.flow.symbol
                elif composite.junction.uri == ZERONODE_JUNCTION:
                    junction_symbol = composite.template.domain.potential.symbol
                else:
                    junction_symbol = 'unknown'
                element_namespace = element_uri.defrag() + '#'
                if len(composite.template.ports):
                    if len(composite.template.ports) == 1:
                        # Add a single junction node for the element
                        junction_uri = element_namespace + f'{junction_symbol}_{element_uri.fragment}'
                        bond_uri = element_uri + '.bond'
                        model_graph.add((junction_uri, RDF.type, composite.junction.uri))
                        model_graph.add((model_uri, BGF.hasJunctionStructure, junction_uri))
                        # All bonds to the element now go to the new junction
                        for row in model_graph.query(ELEMENT_IS_SOURCE.replace('%ELEMENT_URI%', element_uri)):
                            model_graph.add((row[0], BGF.hasSource, junction_uri))
                            model_graph.remove((row[0], BGF.hasSource, element_uri))
                        for row in model_graph.query(ELEMENT_IS_TARGET.replace('%ELEMENT_URI%', element_uri)):
                            model_graph.add((row[0], BGF.hasTarget, junction_uri))
                            model_graph.remove((row[0], BGF.hasTarget, element_uri))
                        # Add a bond from the new junction to its element
                        model_graph.add((bond_uri, BGF.hasSource, junction_uri))
                        model_graph.add((bond_uri, BGF.hasTarget, element_uri))
                    else:
                        # The element has multiple ports so do the above for each port
                        for port in composite.template.ports.items():
                            port_id = port[0]
                            junction_uri = element_namespace + f'{junction_symbol}_{port_id}_{element_uri.fragment}'
                            bond_uri = element_uri + f'_{port_id}.bond'
                            model_graph.add((junction_uri, RDF.type, composite.junction.uri))
                            model_graph.add((model_uri, BGF.hasJunctionStructure, junction_uri))
                            # Bonds to a multi-ported element are to individual ports and keep direction
                            element_is_source = element_is_target = False
                            for row in model_graph.query(ELEMENT_IS_SOURCE_PORT.replace('%ELEMENT_URI%', element_uri)
                                                                               .replace('%PORT_ID%', port_id)):
                                model_graph.add((row[0], BGF.hasSource, junction_uri))
                                model_graph.remove((row[0], BGF.hasSource, row[1]))
                                model_graph.remove((row[1], BGF.element, element_uri))
                                model_graph.remove((row[1], BGF.port, Literal(port_id)))
                                element_is_source = True
                            for row in model_graph.query(ELEMENT_IS_TARGET_PORT.replace('%ELEMENT_URI%', element_uri)
                                                                               .replace('%PORT_ID%', port_id)):
                                model_graph.add((row[0], BGF.hasTarget, junction_uri))
                                model_graph.remove((row[0], BGF.hasTarget, row[1]))
                                model_graph.remove((row[1], BGF.element, element_uri))
                                model_graph.remove((row[1], BGF.port, Literal(port_id)))
                                element_is_target = True
                            if element_is_source and element_is_target:
                                raise ValueError(f'Port {port_id} of {element_uri} cannot have both forward and reverse connections')
                            elif element_is_source:
                                blank_node = BNode()
                                model_graph.add((bond_uri, BGF.hasSource, blank_node))
                                model_graph.add((blank_node, BGF.element, element_uri))
                                model_graph.add((blank_node, BGF.port, Literal(port_id)))
                                model_graph.add((bond_uri, BGF.hasTarget, junction_uri))
                            elif element_is_target:
                                model_graph.add((bond_uri, BGF.hasSource, junction_uri))
                                blank_node = BNode()
                                model_graph.add((bond_uri, BGF.hasTarget, blank_node))
                                model_graph.add((blank_node, BGF.element, element_uri))
                                model_graph.add((blank_node, BGF.port, Literal(port_id)))
                            else:
                                raise ValueError(f'Port {port_id} of {element_uri} has no connections to it')

#===============================================================================
#===============================================================================

BondgraphFramework = _BondgraphFramework([
    '../schema/ontology.ttl',
    '../schema/elements/chemical.ttl',
    '../schema/elements/electrical.ttl',
    '../schema/elements/hydraulic.ttl',
    '../schema/elements/mechanical.ttl',
])

#===============================================================================
#===============================================================================
