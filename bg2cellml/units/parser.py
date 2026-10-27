#===============================================================================
#===============================================================================

import ast
import logging
import math
import operator
import os

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)

#===============================================================================

import pint
from pint.facets.plain import PlainUnit
from ucumvert import PintUcumRegistry

#===============================================================================

ureg = pint.UnitRegistry()
ucum = PintUcumRegistry()

#===============================================================================

CELLML_UNITS = [
    'ampere', 'farad', 'katal', 'lux', 'pascal', 'tesla',
    'becquerel', 'gram', 'kelvin', 'meter', 'radian', 'volt',
    'candela', 'gray', 'kilogram', 'metre', 'second', 'watt',
    'celsius', 'henry', 'liter', 'mole', 'siemens', 'weber',
    'coulomb', 'hertz', 'litre', 'newton', 'sievert',
    'dimensionless', 'joule', 'lumen', 'ohm', 'steradian',
]

SUBSTITUTIONS = {
    'kilopascal': 'kPa',
    'liter': 'litre',
    'meter': 'metre'
}

#===============================================================================

PREFIX_TO_EXPONENT: dict[str, int] = {}
PREFIXES: set[str] = set()

for _prefix in ureg._prefixes.values():
    if _prefix.name != '' and _prefix.name not in PREFIXES:
        PREFIXES.add(_prefix.name)
        exponent = round(math.log10(_prefix.value), 5)     # pyright: ignore[reportArgumentType]
        if exponent == int(exponent):
            PREFIX_TO_EXPONENT[_prefix.name] = int(exponent)

def unit_prefix(unit: str) -> tuple[str, int]:
#=============================================
    for name, exponent in PREFIX_TO_EXPONENT.items():
        if unit.startswith(name):
            base_name = unit[len(name):]
            # CellML has both `gram` and `kilogram` as base units
            if base_name == 'gram' and exponent > 0:
                base_name = 'kilogram'
                exponent -= 3
            return (base_name, exponent)
    return (unit, 0)


def unit_name(name: str) -> str:
#===============================
    name = (name.replace(' * ', '_')
                .replace(' / ', '_per_')
                .replace(' ** 2', '_squared')
                .replace(' ** 3', '_cubed')
                .replace(' ** ', '_power_')
                .replace('1_per_', 'per_'))
    for fullname, replacement in SUBSTITUTIONS.items():
        name = name.replace(fullname, replacement)
    return name

#===============================================================================

class PrefixedUnit:
    def __init__(self, unit: PlainUnit, prefix: int, exponent: int|float=1):
        self.__unit = unit
        self.__prefix = prefix
        self.__exponent = exponent

    @classmethod
    def from_string(cls, unit: str):
        u_p = unit_prefix(unit)
        return cls(ureg(u_p[0]).u, u_p[1])

    @classmethod
    def from_ucum(cls, unit: str):
        u_p = unit_prefix(unit)
        return cls(ureg(u_p[0]).u, u_p[1])

    @property
    def exponent(self) -> int|float:
        return self.__exponent

    @property
    def prefix(self) -> int:
        return self.__prefix

    @property
    def unit(self) -> PlainUnit:
        return self.__unit

    def __str__(self):
        if self.__exponent == 0:
            return '1'
        unit = str(self.__unit) if self.__prefix == 0 else f'10**{self.__prefix} {self.__unit}'
        if self.__exponent == 1:
            return unit
        else:
            return f'({unit})**{self.__exponent}'

    def __truediv__(self, other: PrefixedUnit):
        if self.__unit == other.__unit:
            result = PrefixedUnit(self.__unit, 0, 0)
        else:
            result = PrefixedUnit(self.__unit.__pow__(self.__exponent)/other.__unit.__pow__(other.__exponent), # pyright: ignore[reportArgumentType]
                                  self.__prefix - other.__prefix)
        logging.debug(f'{self} / {other} -> {result}')
        return result

    def __mul__(self, other):
        if self.__unit == other.__unit:
            result = PrefixedUnit(self.__unit, 2*self.__prefix, 2*self.__exponent)
        else:
            result = PrefixedUnit(self.__unit.__pow__(self.__exponent)*other.__unit.__pow__(other.__exponent),
                                  self.__prefix + other.__prefix)
        logging.debug(f'{self} * {other} -> {result}')
        return result

    def __pow__(self, exponent):
        result = PrefixedUnit(self.__unit, exponent*self.__prefix, exponent*self.__exponent)
        logging.debug(f'{self} ^ {exponent} -> {result}')
        return result

    def reciprocal(self):
        result = PrefixedUnit(self.__unit, self.__prefix, -self.__exponent)
        logging.debug(f'1 / {self} -> {result}')
        return result

#===============================================================================

class UnitEvaluator(ast.NodeVisitor):
    _bin_ops = {
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)
        op_type = type(node.op)
        if op_type == ast.Div and left_val == 1:
            return right_val.reciprocal()
        elif op_type in self._bin_ops:
            return self._bin_ops[op_type](left_val, right_val)
        raise TypeError(f"Unsupported binary operator: {op_type.__name__}")

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        raise TypeError(f"Unsupported unary operator: {op_type.__name__}")

    def visit_Constant(self, node):
        # Handles numbers (e.g., 5, 3.14)
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Name(self, node):
        # Treat variable names in the expression as units (e.g., meters, kg, seconds)
        try:
            return PrefixedUnit.from_string(node.id)
        except pint.UndefinedUnitError:
            raise NameError(f"Unknown unit or variable: '{node.id}'")

    def generic_visit(self, node):
        # Raise an error for any syntax node that isn't explicitly allowed (e.g., Calls, Attributes)
        raise ValueError(f"Malicious or unsupported expression structure: {type(node).__name__}")

#===============================================================================
#===============================================================================
