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

from typing import Optional, Self

#===============================================================================

import pint
from rdflib import Literal, URIRef
from ucumvert import PintUcumRegistry

#===============================================================================

from ..bondgraph.namespaces import CDT

#===============================================================================

unit_registry = PintUcumRegistry()

PREFERRED_BASE_ITEMS = {
    'kilopascal': [
        ('joule', 0),
        ('litre', -1),
    ]
}

SUBSTITUTIONS = {
    'kilopascal': 'kPa',
    'liter': 'litre',
    'meter': 'metre'
}

#===============================================================================

class Units:
    def __init__(self, units: str|pint.Unit):
        if isinstance(units, str):
            units = unit_registry[units]
        self.__units = units
        self.__name = Units.normalise_name(str(self.__units))

    @classmethod
    def from_ucum(cls, ucum_units: Literal|str) -> Self:
        if (not isinstance(ucum_units, str)
         and ucum_units.datatype != CDT.ucumunit
         and ucum_units.datatype is not None):
            raise TypeError(f'Units value has unexpected datatype: {ucum_units.datatype}')
        return cls(unit_registry.from_ucum(str(ucum_units)))

    @staticmethod
    def normalise_name(name: str) -> str:
    #====================================
        name = (name.replace(' * ', '_')
                    .replace(' / ', '_per_')
                    .replace(' ** 2', '_squared'))
        for fullname, replacement in SUBSTITUTIONS.items():
            name = name.replace(fullname, replacement)
        return name

    def __eq__(self, other):
        return self.__units == other.__units

    def __str__(self):
        return str(self.__units)

    @property
    def name(self):
        return self.__name

    def base_items(self):
    #====================
        unit_quantity = pint.Quantity(1, self.__units)
        return PREFERRED_BASE_ITEMS.get(str(self),
                                        unit_quantity.unit_items())
        ## if not PREFERRED_BASE_ITEMS start by going through
        ## self.__pint_units.unit_items() and only go to_base_units() if not a known
        ## CELLML_UNIT...

    def is_compatible_with(self, other: Self):
    #=========================================
        return self.__units.is_compatible_with(other.__units)

#===============================================================================

class Value:
    def __init__(self, value: Literal):
        if value.datatype == CDT.ucum:
            parts = str(value).split()
            self.__value = float(parts[0])
            self.__units = Units.from_ucum(parts[1])
        elif value.datatype is None:
            self.__value = float(value)
            self.__units = None
        else:
            raise TypeError(f'Literal value has unexpected datatype: {value.datatype}')

    def __str__(self):
        return f'{self.__value} {self.__units}'

    @property
    def units(self):
        return self.__units

    @property
    def value(self) -> float:
        return self.__value

    def set_units(self, units: Units):
    #=================================
        if self.__units is None:
            self.__units = units
        elif units != self.__units:
            raise ValueError(f'Can not reassign Units of a Value ({self.__units} != {units})')

    def set_value(self, value: float):
    #================================
        self.__value = value

#===============================================================================

class Quantity:
    def __init__(self, uri: URIRef, units: Literal, label: Optional[Literal]=None, variable: Optional[Literal]=None):
        self.__uri = uri
        self.__units = Units.from_ucum(units)
        self.__label = str(label) if label is not None else str(uri)
        self.__variable = str(variable) if variable is not None else self.__label

    @property
    def label(self) -> Optional[str]:
        return self.__label

    @property
    def units(self):
        return self.__units

    @property
    def uri(self):
        return self.__uri

    @property
    def variable(self):
        return self.__variable

#===============================================================================
