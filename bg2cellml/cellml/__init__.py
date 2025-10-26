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

import lxml.etree as etree
import sympy

#===============================================================================

from ..bondgraph import BondgraphElement, BondgraphJunction, BondgraphModel
from ..bondgraph import VOI_VARIABLE
from ..bondgraph.framework import clean_name, Variable
from ..mathml import Equation, MATHML_NS
from ..units import Units
from ..utils import XMLNamespace

#===============================================================================

CELLML_NS = XMLNamespace('http://www.cellml.org/cellml/1.1#')

def cellml_element(tag: str, *args, **attributes) -> etree.Element:
#==================================================================
    return etree.Element(CELLML_NS(tag), *args, **attributes)

def cellml_subelement(parent: etree.Element, tag: str, *args, **attributes) -> etree.Element:
#============================================================================================
    return etree.SubElement(parent, CELLML_NS(tag), *args, **attributes)

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

DIMENSIONLESS_UNITS_NAME = 'dim'
CELLML_UNITS_ATTRIB = CELLML_NS('units')

DIMENSIONLESS_UNIT_DEFINITION = [
    f'<units name="{DIMENSIONLESS_UNITS_NAME}">',
    '<unit units="dimensionless"/>',
    '</units>'
]

#===============================================================================

def symbol_sort_key(symbol: str) -> str:
    return (symbol[2:] + symbol[0:2]) if symbol[0:2] in ['u_', 'v_'] else symbol

#===============================================================================

class CellMLVariable:
    def __init__(self, variable: Variable):
        self.__symbol = variable.symbol
        self.__units = variable.units.name
        if variable.value is not None:
            self.__initial_value = variable.value.value
        else:
            self.__initial_value = None

    def get_element(self) -> etree.Element:
    #======================================
        element = cellml_element('variable', name=self.__symbol, units=self.__units)
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
        self.__name = f'BG_{clean_name(model.uri.fragment)}'
        self.__cellml = cellml_element('model', name=self.__name,
                                        nsmap={None: str(CELLML_NS), 'cellml': str(CELLML_NS)})
        self.__main = cellml_subelement(self.__cellml, 'component', name='main')
        self.__model = model
        self.__known_units: set[str] = set()
        self.__known_fixed: set[str] = set()
        self.__known_variables: dict[str, Variable] = {}

        self.__add_unit_xml(DIMENSIONLESS_UNIT_DEFINITION)  ## Only if <cn> in MathML??
        self.__add_fixed(VOI_VARIABLE)       # only if VOI in some element's CR??

        for element in model.elements:
            self.__add_element(element)
        for junction in model.junctions:
            self.__add_junction_variables(junction)
        self.__output_variable_definitions()
        self.__equations_to_mathml()
        self.__add_dimensionless_attrib()

    @property
    def name(self):
    #==============
        return self.__name

    def __add_element(self, element: BondgraphElement):
    #==================================================
        for constant in element.domain.constants:
            self.__add_fixed(constant)
        for variable in element.variables.values():
            self.__add_variable(variable)

    def __add_dimensionless_attrib(self):
    #====================================
        for element in self.__main.findall(f'.//{MATHML_NS.cn}'):
            element.attrib[CELLML_UNITS_ATTRIB] = DIMENSIONLESS_UNITS_NAME

    def __add_fixed(self, variable: Variable):
    #===========================================
        if variable.symbol not in self.__known_fixed:
            self.__add_units(variable.units)
            cellml_variable = CellMLVariable(variable)
            self.__main.append(cellml_variable.get_element())
            self.__known_fixed.add(variable.symbol)

    def __add_junction_variables(self, junction: BondgraphJunction):
    #===============================================================
        for variable in junction.variables.values():
            self.__add_variable(variable)

    def __add_units(self, units: Units):
    #===================================
        elements = self.__elements_from_units(units)
        if len(elements):
            for element in elements:
                self.__add_unit_xml(element)

    def __add_unit_xml(self, unit_xml: list[str]):
    #=============================================
        if len(unit_xml):
            units_element = etree.fromstring(''.join(unit_xml))
            self.__main.addprevious(units_element)

    def __add_variable(self, variable: Variable):
    #============================================
        if variable.symbol not in self.__known_variables:
            self.__known_variables[variable.symbol] = variable

    def __output_variable_definitions(self):
    #=======================================
        for symbol in sorted(self.__known_variables.keys(), key=symbol_sort_key):
            variable = self.__known_variables[symbol]
            self.__add_units(variable.units)
            cellml_variable = CellMLVariable(variable)
            self.__main.append(cellml_variable.get_element())

    def __elements_from_units(self, units: Units) -> list[list[str]]:
    #================================================================
        result = []
        def elements_from_units(units: Units) -> list[str]:
            if units.name in self.__known_units or units.name in CELLML_UNITS:
                return []
            elements = []
            elements.append(f'<units xmlns="{CELLML_NS}" name="{units.name}">')
            for item in units.base_items():
                if item[0] not in self.__known_units:
                    item_elements = elements_from_units(Units(item[0]))
                    result.append(item_elements)
                name = Units.normalise_name(item[0])
                if item[1] == 0: elements.append(f'<unit units="{name}"/>')
                else: elements.append(f'<unit units="{name}" exponent="{item[1]}"/>')
            elements.append('</units>')
            self.__known_units.add(units.name)
            return elements
        result.append(elements_from_units(units))
        return result

    def __output_equations(self, equations: list[Equation], description: str):
    #=========================================================================
        if len(equations):
            self.__main.append(etree.Comment(f' {description}'))
            for equation in sorted(equations, key=lambda eq: str(eq.lhs)):
                self.__main.append(equation.mathml_equation())

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

    def to_xml(self) -> str:
    #=======================
        cellml_tree = etree.ElementTree(self.__cellml)
        return etree.tostring(cellml_tree,
            encoding='unicode', inclusive_ns_prefixes=['cellml'],
            pretty_print=True)

#===============================================================================
#===============================================================================
