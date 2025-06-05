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

from typing import Optional, TYPE_CHECKING

#===============================================================================

import lxml.etree as etree
import sympy
from sympy.printing.mathml import MathMLContentPrinter

#===============================================================================

from ..definitions import BONDGRAPH_EQUATIONS
from ..rdf import XMLNamespace
from ..units import Units

if TYPE_CHECKING:
    from ..bondgraph import BondgraphModel, BondgraphNode

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
    def __init__(self, name: str, units: Units):
        self.__name = name
        self.__units = units.name
        self.__initial_value = None

    def set_initial_value(self, value: float):
    #=========================================
        self.__initial_value = value

    def get_element(self) -> etree.Element:
    #======================================
        element = cellml_element('variable', name=self.__name, units=self.__units)
        if self.__initial_value is not None:
            element.attrib['initial_value'] = f'{self.__initial_value}'
        return element

    @property
    def name(self):
    #==============
        return self.__name

#===============================================================================

class CellMLModel:
    def __init__(self, name: str, time_var:str='t', time_units: Units=Units('s')):
        self.__name = name
        self.__time_var = time_var
        self.__time_units = time_units
        self.__have_time_var: bool = False
        self.__cellml = cellml_element('model', name=name.replace(' ', '_').replace('-', '_'), nsmap={None: str(CELLML_NS)})
        self.__main = cellml_subelement(self.__cellml, 'component', name='main')
        self.__known_units: list[str] = []

    @property
    def name(self):
    #==============
        return self.__name

    def __add_equations(self, node: 'BondgraphNode'):
    #================================================
        equations = BONDGRAPH_EQUATIONS.get(node.type, [])
        if len(equations):
            for equation in equations:
                if 'TIME' in equation:
                    self.__add_time_var()
                    TIME = sympy.Symbol(self.__time_var)
                    break
            local_names = locals()
            n = 0
            node_delta = []
            for name in node.delta.split():
                if name not in ['+', '-']:
                    local_names[f'N_{n}'] = sympy.Symbol(name)
                    node_delta.append(f'N_{n}')
                    n += 1
                else:
                    node_delta.append(name)
            node_delta = ' '.join(node_delta)
            for quantity, name, value in node.quantity_values:
                local_names[quantity.variable] = sympy.Symbol(name)
            NODE = sympy.Symbol(node.name)
            mathml_printer = MathMLContentPrinter({'disable_split_super_sub': True})
            mathml = ['<math xmlns="http://www.w3.org/1998/Math/MathML">']
            mathml.extend([mathml_printer.doprint(eval(equation.format(NODE_DELTA=node_delta)))
                                                        for equation in equations])
            mathml.append('</math>')
            self.__main.append(etree.fromstring(''.join(mathml)))

    def add_node(self, node: 'BondgraphNode'):
    #=========================================
        self.__add_variable(node.name, node.units, node.value)
        for quantity, name, value in node.quantity_values:
            self.__add_variable(name, quantity.units, value)
        # Assign equation variables now that quantities have names
        self.__add_equations(node)

    def __add_time_var(self):
    #========================
        if not self.__have_time_var:
            self.__add_units(self.__time_units)
            self.__add_variable(self.__time_var, self.__time_units)
            self.__have_time_var = True

    def __add_units(self, units: Units):
    #===================================
        elements = self.__elements_from_units(units)
        if len(elements):
            units_element = etree.fromstring(''.join(elements))
            self.__main.addprevious(units_element)

    def __add_variable(self, name: str, units: Units, init: Optional[float]=None):
    #=============================================================================
        self.__add_units(units)
        variable = CellMLVariable(name, units)
        if init is not None:
            variable.set_initial_value(init)
        self.__main.append(variable.get_element())

    def __elements_from_units(self, units: Units) -> list[str]:
    #==========================================================
        if str(units) in self.__known_units or str(units) in CELLML_UNITS:
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
        self.__known_units.append(str(units))
        return elements

    def to_xml(self) -> bytes:
    #=========================
        cellml_tree = etree.ElementTree(self.__cellml)
        return etree.tostring(cellml_tree,
            encoding='utf-8', inclusive_ns_prefixes=['cellml'],
            pretty_print=True, xml_declaration=True)

#===============================================================================

def generate_cellml(bondgraph: 'BondgraphModel') -> bytes:
#=========================================================
    cellml = CellMLModel(bondgraph.name)
    for node in bondgraph.nodes:
        cellml.add_node(node)
    return cellml.to_xml()

#===============================================================================
