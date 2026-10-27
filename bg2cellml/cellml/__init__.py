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

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

#===============================================================================

import lxml.etree as etree
import pint
from pint.facets.plain import PlainUnit
import sigfig
import sympy

#===============================================================================

from ..bondgraph.framework_support import Variable, VOI_VARIABLE
from ..bondgraph.model_support import BondgraphElement, BondgraphJunction
from ..bondgraph.utils import clean_name, ModelElement
from ..mathml import Equation, MATHML_NS
from ..rdf import literal, namedNode, RdfGraph, Triple, uri_fragment
from ..rdf.namespace import Namespace, RDF, RDFS
from ..units import Unit
from ..units.parser import unit_name, unit_name_with_exponent, UnitComponent
from ..utils import XMLNamespace
from ..version import __version__

if TYPE_CHECKING:
    from ..bondgraph.model import BondgraphModel

#===============================================================================

CELLML_MODEL_URI = 'file:///CELLML_MODEL_URI'
CELLML_MODEL_NS = Namespace(f'{CELLML_MODEL_URI}#')

BQMODEL = Namespace('http://biomodels.net/model-qualifiers/')

METADATA_NAMESPACES = {
    'bgf': 'https://bg-rdf.org/ontologies/bondgraph-framework#',
    'bqmodel': str(BQMODEL),
    'rdfs': str(RDFS),
}

#===============================================================================

CELLML_NS = XMLNamespace('http://www.cellml.org/cellml/1.1#')
CMETA_NS = XMLNamespace('http://www.cellml.org/metadata/2.0#')

#===============================================================================

def cellml_element(tag: str, *args, **attributes) -> etree.Element:
#==================================================================
    if 'id' in attributes:
        attributes[CMETA_NS('id')] = attributes.pop('id')
    return etree.Element(CELLML_NS(tag), *args, **attributes)

def cellml_subelement(parent: etree.Element, tag: str, *args, **attributes) -> etree.Element:
#============================================================================================
    if 'id' in attributes:
        attributes[CMETA_NS('id')] = attributes.pop('id')
    return etree.SubElement(parent, CELLML_NS(tag), *args, **attributes)

def header_comment(source_uri: str) -> etree.Comment:
#====================================================
    utc = datetime.now(timezone.utc)
    return etree.Comment(f'''
  This CellML file was generated at {utc.isoformat()}

  by [BG-Tools](https://github.com/CellDL/BG-Tools), version {__version__}

  from {source_uri}
''')

#===============================================================================

CELLML_UNITS = [
    'ampere', 'farad', 'katal', 'lux', 'pascal', 'tesla',
    'becquerel', 'gram', 'kelvin', 'meter', 'radian', 'volt',
    'candela', 'gray', 'kilogram', 'metre', 'second', 'watt',
    'celsius', 'henry', 'liter', 'mole', 'siemens', 'weber',
    'coulomb', 'hertz', 'litre', 'newton', 'sievert',
    'dimensionless', 'joule', 'lumen', 'ohm', 'steradian',
]

#===============================================================================

PREFERRED_BASE_UNITS = {
    'kPa': (1, [
        ('joule', 0),
        ('litre', -1),
    ])
}

def base_units(unit: PlainUnit) -> tuple[int|float, list[tuple[str, int|float]]]:
#================================================================================
    name = unit_name(unit)
    if name in CELLML_UNITS:
        return (1, [(name, 1)])
    elif name in PREFERRED_BASE_UNITS:
        return PREFERRED_BASE_UNITS[name]       # pyright: ignore[reportReturnType]
    else:
        base_units = pint.Quantity(1, unit).to_base_units()
        return (base_units.magnitude, [(unit_name(u), e) for (u, e) in base_units.unit_items()])    # pyright: ignore[reportReturnType]

def unit_xml(name: str, exponent: int|float=1, prefix: int=0, multiplier:int|float=1) -> str:
#============================================================================================

    attributes: dict[str, str] = { 'units': name }
    if multiplier != 1:
        attributes['multiplier'] = str(multiplier)
    if prefix != 0:
        attributes['prefix'] = str(prefix)
    if exponent != 1:
        attributes['exponent'] = str(exponent)
    return f'    <unit {' '.join([f'{key}="{value}"' for key, value in attributes.items()])}/>'

#===============================================================================

DIMENSIONLESS_UNITS_NAME = 'dim'
CELLML_UNITS_ATTRIB = CELLML_NS('units')

DIMENSIONLESS_UNIT_DEFINITION = [
    f'<units name="{DIMENSIONLESS_UNITS_NAME}">',
    '    <unit units="dimensionless"/>',
    '  </units>'
]

#===============================================================================

def symbol_sort_key(symbol: str) -> str:
    return (symbol[2:] + symbol[0:2]) if symbol[0:2] in ['u_', 'v_'] else symbol

#===============================================================================

class CellMLComponent:
    def __init__(self, name: str, parent: etree.Element):
        self.__id = name
        self.__name = name
        self.__element = cellml_subelement(parent, 'component', name=name, id=self.__id)
        self.__bg_elements = defaultdict(list[str])

    @property
    def bg_elements(self):
    #=====================
        return self.__bg_elements

    @property
    def element(self):
    #=================
        return self.__element

    @property
    def id(self):
    #============
        return self.__id

    def add_dimensionless_attrib(self):
    #====================================
        for element in self.__element.findall(f'.//{MATHML_NS.cn}'):
            element.attrib[CELLML_UNITS_ATTRIB] = DIMENSIONLESS_UNITS_NAME

    def add_element(self, element: etree.Element):
    #=============================================
        self.__element.append(element)

    def add_variable(self, variable: 'CellMLVariable', bg_node_id: str|None=None):
    #=============================================================================
        self.__element.append(variable.get_element())
        if bg_node_id is not None:
            self.__bg_elements[bg_node_id].append(f'{self.__name}/{variable.symbol}')

#===============================================================================

class CellMLVariable:
    def __init__(self, component: CellMLComponent, variable: Variable):
        self.__id = f'{component.id}-{variable.symbol}'
        self.__symbol = variable.symbol
        self.__unit = variable.unit
        if variable.value is not None:
            if variable.value.unit is not None:
                self.__unit = variable.value.unit
            self.__initial_value = variable.value.value
        else:
            self.__initial_value = None

    @property
    def id(self):
    #============
        return self.__id

    @property
    def unit(self) -> Unit:
    #======================
        return self.__unit

    def get_element(self) -> etree.Element:
    #======================================
        element = cellml_element('variable', name=self.__symbol, units=self.__unit.name, id=self.__id)
        if self.__initial_value is not None:
            element.attrib['initial_value'] = f'{self.__initial_value}'
        return element

    @property
    def symbol(self):
    #================
        return self.__symbol

#===============================================================================

class CellMLModel:
    def __init__(self, model: 'BondgraphModel'):
        name = uri_fragment(model.uri).rsplit('.')[0]
        self.__name = f'BG_{clean_name(name)}'
        self.__cellml = cellml_element('model', name=self.__name,
                                        nsmap={
                                            None: str(CELLML_NS),
                                            'cellml': str(CELLML_NS),
                                            'cmeta': str(CMETA_NS)
                                        })
        self.__components: list[CellMLComponent] = []
        self.__components.append(CellMLComponent('main', self.__cellml))
        self.__first_component_element = self.__components[0].element
        self.__current_component = self.__components[0]

        self.__model = model
        self.__known_units: set[str] = set()
        self.__known_fixed: set[str] = set()
        self.__known_variables: dict[str, tuple[Variable, ModelElement]] = {}

        self.__metadata = RdfGraph()
        self.__metadata.add(Triple(namedNode(CELLML_MODEL_URI), BQMODEL.isDescribedBy, model.uri))

        self.__add_unit_xml(DIMENSIONLESS_UNIT_DEFINITION)  ## Only if <cn> in MathML??
        self.__add_fixed(VOI_VARIABLE)       # only if VOI in some element's CR??

        for element in model.elements:
            self.__add_element(element)
        for junction in model.junctions:
            self.__add_junction_variables(junction)
        self.__output_variable_definitions()
        self.__equations_to_mathml()
        for component in self.__components:
            component.add_dimensionless_attrib()

    @property
    def name(self):
    #==============
        return self.__name

    def __add_element(self, element: BondgraphElement):
    #==================================================
        for constant in element.domain.constants:
            self.__add_fixed(constant)
        for variable in element.variables.values():
            self.__add_variable(variable, element)

    def __add_fixed(self, variable: Variable):
    #=========================================
        if variable.symbol not in self.__known_fixed:
            cellml_variable = CellMLVariable(self.__current_component, variable)
            self.__add_units(cellml_variable.unit)
            self.__current_component.add_variable(cellml_variable)
            self.__known_fixed.add(variable.symbol)

    def __add_junction_variables(self, junction: BondgraphJunction):
    #===============================================================
        for variable in junction.variables.values():
            self.__add_variable(variable, junction)

    def __add_metadata(self, cellml_variable: CellMLVariable, variable: Variable, model_element: ModelElement):
    #==========================================================================================================
        if variable.type is not None:
            self.__metadata.add(Triple(CELLML_MODEL_NS(cellml_variable.id), BQMODEL.isDerivedFrom, namedNode(model_element.id)))
            self.__metadata.add(Triple(CELLML_MODEL_NS(cellml_variable.id), RDF.type, variable.type))
        if model_element.label:
            # This won't create duplicate statements
            self.__metadata.add(Triple(namedNode(model_element.id), RDFS.label, literal(model_element.label)))

    def __add_units(self, unit: Unit):
    #=================================
        elements = self.__elements_from_units(unit)
        if len(elements):
            for element in elements:
                self.__add_unit_xml(element)

    def __add_unit_xml(self, unit_xml: list[str]):
    #=============================================
        if len(unit_xml):
            units_element = etree.fromstring('\n'.join(unit_xml))
            self.__first_component_element.addprevious(units_element)

    def __add_variable(self, variable: Variable, element: ModelElement):
    #===================================================================
        if variable.symbol not in self.__known_variables:
            # variables/component
            self.__known_variables[variable.symbol] = (variable, element)

    def __output_variable_definitions(self):
    #=======================================
        for symbol in sorted(self.__known_variables.keys(), key=symbol_sort_key):
            variable, model_element = self.__known_variables[symbol]
            cellml_variable = CellMLVariable(self.__current_component, variable)
            self.__add_units(cellml_variable.unit)
            self.__current_component.add_variable(cellml_variable, model_element.id)
            self.__add_metadata(cellml_variable, variable, model_element)

    def __elements_from_units(self, unit: Unit) -> list[list[str]]:
    #==============================================================
        if unit.name in self.__known_units or unit.name in CELLML_UNITS:
            return []
        result = []

        def add_component_xml(unit_name: str, component: UnitComponent):
            elements: list[str] = [
                f'<units xmlns="{CELLML_NS}" name="{unit_name}">'
            ]
            base = base_units(component.unit)
            for (n, base_exponent) in enumerate(base[1]):
                (base_name, exponent) = base_exponent
                multiplier = sigfig.round(base[0], 5, warn=False) if n == 0 else 1
                elements.append(unit_xml(base_name,
                                         exponent=exponent,
                                         multiplier=multiplier))
            self.__known_units.add(unit_name)
            if len(elements) > 1:
                elements.append('  </units>')
                result.append(elements)

        elements: list[str] = [
            f'<units xmlns="{CELLML_NS}" name="{unit.name}">'
        ]
        xml_components: dict[str, UnitComponent] = {}
        for component in unit.unit_components:
            name = unit_name(component.unit)
            elements.append(unit_xml(unit_name(component.unit),
                                     exponent=component.exponent,
                                     prefix=component.prefix,
                                     multiplier=component.multiplier))
            if name not in self.__known_units and name not in CELLML_UNITS:
                xml_components[name] = component
        self.__known_units.add(unit.name)
        if len(elements) > 1:
            elements.append('  </units>')
            result.append(elements)
            for name, component in xml_components.items():
                add_component_xml(name, component)
        return result

    def __output_equations(self, equations: list[Equation], description: str):
    #=========================================================================
        if len(equations):
            self.__current_component.add_element(etree.Comment(f' {description}'))
            for equation in sorted(equations, key=lambda eq: str(eq.lhs)):
                self.__current_component.add_element(equation.mathml_equation())

    def __equations_to_mathml(self):
    #===============================
        equations = self.__model.equations
        element_odes: list[Equation] = []
        element_algebraics: list[Equation] = []
        junction_algebraics: list[Equation] = []
        for equation in equations:
            if equation.provenance == 'cr':
                if isinstance(equation.lhs, sympy.Symbol):
                    element_algebraics.append(equation)
                elif isinstance(equation.lhs, sympy.Derivative):
                    element_odes.append(equation)
            elif equation.provenance == 'be':
                element_algebraics.append(equation)
            else:
                junction_algebraics.append(equation)
        self.__output_equations(element_odes, 'Element ODEs')
        self.__output_equations(element_algebraics, 'Element algebraics')
        self.__output_equations(junction_algebraics, 'Junction algebraics')

    def annotation(self) -> list[dict[str, dict[str, list[str]]]]:
    #=============================================================
        bg_vars = defaultdict(list[str])
        for component in self.__components:
            for bg_element, variables in component.bg_elements.items():
                bg_vars[bg_element].extend(variables)
        return [
            { id: { 'variables': vars } }
                for id, vars in bg_vars.items()
        ]

    async def metadata(self, cellml_uri: str|None) -> str:
    #=====================================================
        namespaces = METADATA_NAMESPACES.copy()
        namespaces['model'] = str(CELLML_MODEL_NS)
        namespaces['diagram'] = f'{self.__model.uri.value}#'
        metadata = await self.__metadata.serialise(namespaces)
        if cellml_uri is not None:
            metadata = metadata.replace(CELLML_MODEL_URI, cellml_uri)
        return metadata

    def to_xml(self) -> str:
    #=======================
        self.__cellml.addprevious(header_comment(self.__model.uri.value))
        cellml_tree = etree.ElementTree(self.__cellml)
        return etree.tostring(cellml_tree,
            encoding='unicode', inclusive_ns_prefixes=['cellml'],
            pretty_print=True)

#===============================================================================
#===============================================================================
