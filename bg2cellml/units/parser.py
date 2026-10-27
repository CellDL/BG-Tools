#===============================================================================
#===============================================================================

import ast
from dataclasses import dataclass
import logging
import math
import operator
import os

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)

#===============================================================================

import pint
from pint.facets.plain import PlainUnit

#===============================================================================

ureg = pint.UnitRegistry()

#===============================================================================

SUBSTITUTIONS = {
    'kilopascal': 'kPa',
    'liter': 'litre',
    'meter': 'metre'
}

def unit_name(name: str|PlainUnit) -> str:
#=========================================
    name = (str(name).replace(' * ', '_')
                     .replace(' / ', '_per_')
                     .replace(' ** 2', '_squared')
                     .replace(' ** 3', '_cubed')
                     .replace(' ** ', '_power_')
                     .replace('1_per_', 'per_'))
    for fullname, replacement in SUBSTITUTIONS.items():
        name = name.replace(fullname, replacement)
    return name

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
            elif base_name == 'pascal' and exponent > 0:
                base_name = 'kilopascal'
                exponent -= 3
            return (base_name, exponent)
    return (unit, 0)

def unit_name_with_exponent(unit: str|PlainUnit, exponent: int|float) -> str:
    name = unit_name(unit)
    if exponent == 1:
        return name
    elif exponent == -1:
        return f'per_{name}'
    elif exponent == 2:
        return f'{name}_squared'
    elif exponent == 3:
        return f'{name}_cubed'
    else:
        return f'{name}_power_{exponent}'

#===============================================================================

class UnitComponent:
    def __init__(self, unit: PlainUnit, prefix: int, exponent: int|float=1, multiplier: int|float=1):
        logging.debug(f'PU: {prefix} {unit} {exponent}')
        self.__unit = unit
        self.__prefix = prefix
        self.__exponent = exponent
        self.__multiplier = multiplier

    @classmethod
    def from_string(cls, unit: str):
        u_p = unit_prefix(unit)
        return cls(ureg(u_p[0]).u, u_p[1])

    @property
    def id(self) -> str:
        return f'{self.__unit}_{self.__prefix}_{self.__multiplier}'

    def same(self, other):
        return self.id == other.id

    def __str__(self):
        if self.__exponent == 0:
            return f'{self.__multiplier}'
        multiplier = f'{self.__multiplier}*' if self.__multiplier != 1 else ''
        unit = unit_name(self.__unit) if self.__prefix == 0 else f'10**{self.__prefix}.{unit_name(self.__unit)}'
        if self.__exponent == 1:
            return f'{multiplier}{unit}'
        else:
            return f'{multiplier}({unit})**{self.__exponent}'

    def __truediv__(self, other: UnitComponent):
        assert self.same(other)
        return UnitComponent(self.__unit, self.__prefix, self.__exponent - other.__exponent, self.__multiplier)

    def __mul__(self, other):
        assert self.same(other)
        return UnitComponent(self.__unit, self.__prefix, self.__exponent + other.__exponent, self.__multiplier)

    def __pow__(self, exponent):
        return UnitComponent(self.__unit, self.__prefix, exponent*self.__exponent, self.__multiplier)

    @property
    def exponent(self) -> int|float:
        return self.__exponent

    @property
    def name(self) -> str:
        return unit_name_with_exponent(self.__unit, self.__exponent)

    @property
    def multiplier(self):
        return self.__multiplier

    @property
    def prefix(self) -> int:
        return self.__prefix

    @property
    def unit(self) -> PlainUnit:
        return self.__unit

#===============================================================================

class UnitExpression:
    def __init__(self, *components: UnitComponent):
        self.__components: dict[str, UnitComponent] = {
            unit.id: unit for unit in components
        }

    def __truediv__(self, other):
        components = self.__components.copy()
        for component in other.__components.values():
            if component.id in components:
                components[component.id] /= component
            else:
                components[component.id] = component.__pow__(-1)
        return UnitExpression(*components.values())

    def __mul__(self, other):
        components = self.__components.copy()
        for component in other.__components.values():
            if component.id in components:
                components[component.id] *= component
            else:
                components[component.id] = component
        return UnitExpression(*components.values())

    def __pow__(self, exponent):
        return UnitExpression(*[component.__pow__(exponent)
                                for component in self.__components.values()])

    @property
    def components(self) -> list[UnitComponent]:
        return list(self.__components.values())

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
            return right_val.__pow__(-1)
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

    def visit_Name(self, node) -> UnitExpression:
        # Treat variable names in the expression as units (e.g., meters, kg, seconds)
        try:
            return UnitExpression(UnitComponent.from_string(node.id))
        except pint.UndefinedUnitError:
            raise NameError(f"Unknown unit or variable: '{node.id}'")

    def generic_visit(self, node):
        # Raise an error for any syntax node that isn't explicitly allowed (e.g., Calls, Attributes)
        raise ValueError(f"Malicious or unsupported expression structure: {type(node).__name__}")

#===============================================================================

def create_unit_expression(unit_string: str) -> UnitExpression:
    tree = ast.parse(unit_string, mode='eval')
    return UnitEvaluator().visit(tree)

#===============================================================================
#===============================================================================
