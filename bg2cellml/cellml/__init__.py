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

import lxml.etree as etree

#===============================================================================

from ..bondgraph import BondgraphElement, BondgraphJunction, BondgraphModel
from ..bondgraph import VOI_VARIABLE
from ..bondgraph.framework import clean_name, Variable
from ..mathml import MathML
from ..rdf import XMLNamespace
from ..units import Units

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
        self.__name = clean_name(model.uri)
        self.__cellml = cellml_element('model', name=self.__name, nsmap={None: str(CELLML_NS)})
        self.__main = cellml_subelement(self.__cellml, 'component', name='main')
        self.__known_units: set[str] = set()
        self.__known_symbols: set[str] = set()
        self.__add_variable(VOI_VARIABLE)       # only if VOI in some element's CR??
        for element in model.elements:
            self.__add_element(element)
        # Add comment  -- variables for junctions...
        for junction in model.junctions:
            self.__add_junction_variables(junction)
        # Add comment  -- CRs for junction.uri...
        for junction in model.junctions:
            self.__add_constitutive_relation(junction.constitutive_relation)

    @property
    def name(self):
    #==============
        return self.__name

    def __add_element(self, element: BondgraphElement):
    #==================================================
        for constant in element.domain.constants:
            self.__add_variable(constant)
        for variable in element.variables.values():
            self.__add_variable(variable)
        for port in element.ports.values():
            self.__add_variable(port.flow.variable)
            self.__add_variable(port.potential.variable)
        self.__add_constitutive_relation(element.constitutive_relation)

    def __add_junction_variables(self, junction: BondgraphJunction):
    #===============================================================
        for variable in junction.variables:
            self.__add_variable(variable)

    def __add_constitutive_relation(self, constitutive_relation: Optional[MathML]):
    #==============================================================================
        if constitutive_relation is not None:
            self.__main.append(constitutive_relation.mathml)

    def __add_units(self, units: Units):
    #===================================
        elements = self.__elements_from_units(units)
        if len(elements):
            units_element = etree.fromstring(''.join(elements))
            self.__main.addprevious(units_element)

    def __add_variable(self, variable: Variable):
    #============================================
        if variable.symbol not in self.__known_symbols:
            self.__add_units(variable.units)
            cellml_variable = CellMLVariable(variable)
            self.__main.append(cellml_variable.get_element())
            self.__known_symbols.add(variable.symbol)

    def __elements_from_units(self, units: Units) -> list[str]:
    #==========================================================
        if units.name in self.__known_units or units.name in CELLML_UNITS:
            return []
        elements = []
        elements.append(f'<units xmlns="{CELLML_NS}" name="{units.name}">')
        for item in units.base_items():
            if item[0] not in self.__known_units:
                item_elements = self.__elements_from_units(Units(item[0]))
                elements.extend(item_elements)
            name = Units.normalise_name(item[0])
            if item[1] == 0: elements.append(f'<unit units="{name}"/>')
            else: elements.append(f'<unit units="{name}" exponent="{item[1]}"/>')
        elements.append('</units>')
        self.__known_units.add(units.name)
        return elements

    def to_xml(self) -> bytes:
    #=========================
        cellml_tree = etree.ElementTree(self.__cellml)
        return etree.tostring(cellml_tree,
            encoding='unicode', inclusive_ns_prefixes=['cellml'],
            pretty_print=True)

#===============================================================================
