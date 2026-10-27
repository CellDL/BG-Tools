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

from typing import Self

#===============================================================================

import pint
from pint.facets.plain import PlainUnit
from ucumvert import PintUcumRegistry

#===============================================================================

from ..bondgraph.namespaces import CDT
from ..rdf import isLiteral, Literal, NamedNode

from .parser import create_unit_expression, UnitComponent, unit_name

#===============================================================================

ureg = pint.UnitRegistry()
ucum = PintUcumRegistry()

#===============================================================================

class Unit:
    def __init__(self, unit_string: str):
        self.__expression = create_unit_expression(unit_string)
        self.__name = unit_name(unit_string)
        self.__unit_quantity = ureg(unit_string)

    @classmethod
    def from_ucum(cls, ucum_unit: Literal|str) -> Self:
        if isLiteral(ucum_unit):
            if (ucum_unit.datatype is not None                      # pyright: ignore[reportAttributeAccessIssue]
            and ucum_unit.datatype.value != CDT.ucumunit.value):    # pyright: ignore[reportAttributeAccessIssue]
                raise TypeError(f'Unit value has unexpected datatype: {ucum_unit.datatype}')    # pyright: ignore[reportAttributeAccessIssue]
            return cls(str(ucum(ucum_unit.value).units))            # pyright: ignore[reportAttributeAccessIssue]
        return cls(str(ucum(ucum_unit).units))                      # pyright: ignore[reportArgumentType]

    @property
    def unit_components(self) -> list[UnitComponent]:
        return self.__expression.components

    @property
    def name(self):
        return self.__name

    def convert(self, value: float, other: Unit) -> float:
        return pint.Quantity(value, self.__unit_quantity.u).m_as(other.__unit_quantity)

    def is_compatible_with(self, other: Unit):
        return self.__unit_quantity.is_compatible_with(other.__unit_quantity)

#===============================================================================

class Value:
    def __init__(self, value: float, unit: Unit|None):
        self.__unit = unit
        self.__value = value

    @classmethod
    def from_literal(cls, literal_value: Literal) -> Self:
        if literal_value.datatype.value == CDT.ucum.value:  # pyright: ignore[reportAttributeAccessIssue]
            parts = str(literal_value.value).split()
            value = float(parts[0])
            unit = Unit.from_ucum(parts[1])
        elif literal_value.datatype is None:                # pyright: ignore[reportAttributeAccessIssue]
            value = float(literal_value.value)
            unit = None
        else:
            raise TypeError(f'Literal value has unexpected datatype: {literal_value.datatype}') # pyright: ignore[reportAttributeAccessIssue]
        return cls(value, unit)

    def __str__(self):
        return f'{self.__value} {self.__unit}'

    @property
    def unit(self):
        return self.__unit

    @property
    def value(self) -> float:
        return self.__value

    def copy(self) -> 'Value':
        return Value(self.__value, self.__unit)

    def set_unit(self, unit: Unit):
        if self.__unit is None:
            self.__unit = unit
        elif not self.__unit.is_compatible_with(unit):
            raise ValueError(f'Can not change Unit of a Value ({self.__unit} != {unit})')
        elif unit != self.__unit:
            self.__value = self.__unit.convert(self.__value, unit)
            self.__unit = unit

#===============================================================================

class Quantity:
    def __init__(self, uri: NamedNode, unit: Literal, label: Literal|None=None, variable: Literal|None=None):
        self.__uri = uri
        self.__unit = Unit.from_ucum(unit)
        self.__label = str(label) if label is not None else str(uri)
        self.__variable = str(variable) if variable is not None else self.__label

    @property
    def label(self) -> str|None:
        return self.__label

    @property
    def unit(self):
        return self.__unit

    @property
    def uri(self):
        return self.__uri

    @property
    def variable(self):
        return self.__variable

#===============================================================================
#===============================================================================
