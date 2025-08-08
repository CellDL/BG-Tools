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
from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

#===============================================================================

import lxml.etree as etree
import numpy as np
import scipy.sparse
import sympy
from sympy.printing.mathml import MathMLContentPrinter

#===============================================================================

from ..utils import etree_from_string, XMLNamespace

#===============================================================================

MATHML_NS = XMLNamespace('http://www.w3.org/1998/Math/MathML')

#===============================================================================
#===============================================================================

def sympy_from_mathml(node: etree.Element) -> sympy.Expr:
#========================================================
    if node.tag == MATHML_NS.bvar:
        return sympy_from_mathml(node.getchildren()[0])
    elif node.tag == MATHML_NS.ci:
        return sympy.Symbol(node.text)
    elif node.tag == MATHML_NS.cn:
        return sympy.Float(float(node.text))
    elif node.tag != MATHML_NS.apply:
        raise ValueError(f'Expected MathML `apply`, got `{node.tag}`')
    children = node.getchildren()
    operator = children[0].tag
    if operator == MATHML_NS.times:
        terms = [sympy_from_mathml(child) for child in children[1:]
                    if not isinstance(child, etree._Comment)]
        return sympy.Mul(*terms)
    elif operator == MATHML_NS.divide:
        return sympy.Mul(sympy_from_mathml(children[1]),
                         sympy.Pow(sympy_from_mathml(children[2]), sympy.Integer(-1)))
    elif operator == MATHML_NS.plus:
        terms = [sympy_from_mathml(child) for child in children[1:]
                    if not isinstance(child, etree._Comment)]
        return sympy.Add(*terms)
    elif operator == MATHML_NS.minus:
        if len(children) == 2:
            return sympy.Mul(sympy.Integer(-1), sympy_from_mathml(children[1]))
        else:
            return sympy.Add(sympy_from_mathml(children[1]),
                             sympy.Mul(sympy.Integer(-1), sympy_from_mathml(children[2])))
    elif operator == MATHML_NS.ln:
        return sympy.log(sympy_from_mathml(children[1]))
    elif operator == MATHML_NS.exp:
        return sympy.exp(sympy_from_mathml(children[1]))
    elif operator == MATHML_NS.power:
        return sympy.Pow(sympy_from_mathml(children[1]), sympy_from_mathml(children[2]))
    elif operator == MATHML_NS.diff:
        return sympy.Derivative(sympy_from_mathml(children[2]), sympy_from_mathml(children[1]), evaluate=False)
    else:
        raise ValueError(f'Unsupported MathML operator: {operator}')

#===============================================================================

sympy_mathml_printer = MathMLContentPrinter({'disable_split_super_sub': True})

def sympy_to_mathml(sympy_object) -> str:
#========================================
    mathml = ['<math xmlns="http://www.w3.org/1998/Math/MathML">']
    mathml.append(sympy_mathml_printer.doprint(sympy_object))
    mathml.append('</math>')
    return ''.join(mathml)

#===============================================================================
#===============================================================================

@dataclass
class Equation:
    lhs: sympy.Symbol | sympy.Derivative
    rhs: sympy.Basic

    def __str__(self):
        return f'{self.lhs} = {self.rhs}'

    def sympy_equation(self):
    #========================
        return sympy.Eq(self.lhs, self.rhs)

    def mathml_equation(self) -> etree.Element:
    #==========================================
        return etree.fromstring(
            sympy_to_mathml(
                self.sympy_equation()))

#===============================================================================

"""
Maps a ``key`` to an ``int`` to use as an index.
"""
class KeyIndex(dict[Hashable, int]):
    def __init__(self):
        self.__keys: list[Hashable] = []

    def __getitem__(self, key: Hashable):
    #===============================
        if key in self:
           return super().__getitem__(key)
        else:
           index = len(self.__keys)
           self.__keys.append(key)
           super().__setitem__(key, index)
           return index

    def key(self, index: int) -> Hashable:
    #=====================================
        return self.__keys[index]

#===============================================================================

class LinearEquations:
    def __init__(self):
        self.__symbol_index = KeyIndex()
        self.__data_cols: list[int] = []
        self.__data_rows: list[int] = []
        self.__data_values: list[int] = []
        self.__equation_number: int = 0
        self.__matrix = scipy.sparse.csr_array((0, 0), dtype=np.int8)
        self.__equalities: list[Equation] = []
        self.__equations: list[sympy.Expr] = []

    @property
    def equations(self) -> list[Equation]:
    #=====================================
        return self.__equalities

    @property
    def sympy_equations(self) -> list[sympy.Expr]:
    #=============================================
        return self.__equations

    @property
    def matrix(self):
    #================
        return self.__matrix

    @property
    def symbol_index(self):
    #======================
        return self.__symbol_index

    def add_equation(self, inputs: list[str], outputs: list[str]):
    #=============================================================
        if set(inputs) != set(outputs):
            for input in inputs:
                col = self.__symbol_index[input]
                self.__data_cols.append(col)
                self.__data_rows.append(self.__equation_number)
                self.__data_values.append(1)
            for output in outputs:
                col = self.__symbol_index[output]
                self.__data_cols.append(col)
                self.__data_rows.append(self.__equation_number)
                self.__data_values.append(-1)
            self.__equation_number += 1

    def make_matrix(self):
    #=====================
        if len(self.__data_values):
            self.__matrix = scipy.sparse.coo_array((np.array(self.__data_values),
                                                    (np.array(self.__data_rows), np.array(self.__data_cols)))).tocsr()
        #print(self.__symbol_index)
        #print(self.__matrix.toarray())
        self.__build_equations()

    def __build_equations(self):
    #===========================
        for row in self.__matrix:
            lhs = None
            lhs_factor = 1
            rhs_terms = []
            coefficients = row.toarray()    # type: ignore
            for symbol, index in self.__symbol_index.items():
                if abs(coefficients[index]) == 1:
                    if lhs is None:
                        lhs = sympy.Symbol(symbol)
                        lhs_factor = coefficients[index]
                    else:
                        rhs_terms.append(coefficients[index]*sympy.Symbol(symbol))
            if lhs is None:
                raise ValueError(f'Cannot determine LHS of equation: {[str(term) for term in rhs_terms]}')
            elif rhs_terms:
                self.__equalities.append(Equation(lhs, sympy.Mul(-lhs_factor, sympy.Add(*rhs_terms))))
                self.__equations.append(sympy.Add(lhs, *rhs_terms))

#===============================================================================

class EquationSet:
    def __init__(self, equations: list[Equation]):
        self.__equations = equations
        self.__var_equations = defaultdict(set)
        self.__free_symbols = set()
        for n, equation in enumerate(self.__equations):
            free_symbols = equation.rhs.free_symbols
            free_symbols.add(equation.lhs)
            self.__free_symbols |= free_symbols
            for symbol in free_symbols:
                self.__var_equations[symbol].add(n)
        self.reset()

    def contains(self, symbol) -> bool:
    #==================================
        return symbol in self.__free_symbols

    def reset(self):
    #===============
        self.__used_equations = []

    def solve(self, symbol: sympy.Basic) -> Optional[sympy.Expr]:
    #============================================================
        indices = self.__var_equations[symbol]
        for index in indices:
            if index not in self.__used_equations:
                equation = self.__equations[index]
                self.__used_equations.append(index)
                if symbol == equation.lhs:
                    soln = equation.rhs
                elif symbol == -equation.lhs:
                    soln = -equation.rhs
                else:
                    try:
                        soln = sympy.solve(equation.lhs - equation.rhs, symbol, dict=True)
                        soln = soln[0][symbol] if len(soln) == 1 else None
                    except NotImplementedError:
                        soln = None
                if soln is not None:
                    return soln

#===============================================================================

class ODEResolver:
    def __init__(self, algebraics: list[Equation], junctions: list[Equation]):
        self.__algebraic_set = EquationSet(algebraics)
        self.__junction_set = EquationSet(junctions)

    def resolve(self, ode: Equation) -> Equation:
    #============================================
        self.__junction_set.reset()
        self.__algebraic_set.reset()
        return Equation(ode.lhs, self.__resolve_rhs(ode.rhs))

    def __resolve_rhs(self, rhs: sympy.Basic) -> sympy.Basic:
    #========================================================
        free_symbols = list(rhs.free_symbols)
        symbol_index = 0
        seen_symbols = set()
        while symbol_index < len(free_symbols):
            symbol = free_symbols[symbol_index]
            symbol_index += 1
            if symbol not in seen_symbols:
                seen_symbols.add(symbol)
                # First try to solve for symbol in algebriacs
                if self.__algebraic_set.contains(symbol):
                    soln = self.__algebraic_set.solve(symbol)
                    if soln is not None:
                        rhs = rhs.subs(symbol, soln)
                        # Don't resolve symbols in solution
                        continue
                if self.__junction_set.contains(symbol):
                    soln = self.__junction_set.solve(symbol)
                    if soln is not None:
                        rhs = rhs.subs(symbol, soln)
                        # Resolve symbols in solution
                        free_symbols.extend(soln.free_symbols)
        return rhs

#===============================================================================
#===============================================================================

class MathML:
    def __init__(self, mathml: etree.Element):
        self.__mathml = mathml
        self.__variables = defaultdict(list)
        for element in self.__mathml.findall(f'.//{MATHML_NS.ci}'):
            self.__variables[element.text].append(element)
        self.__build_equations()

    @classmethod
    def from_string(cls, formulae: str) -> 'MathML':
        return cls(etree_from_string(formulae))

    @property
    def equalities(self) -> list[Equation]:
        return self.__equalities

    @property
    def variables(self) -> list[str]:
        return list(self.__variables.keys())

    def __str__(self):
    #=================
        return etree.tostring(self.__mathml, encoding='unicode', pretty_print=True)

    def copy(self) -> 'MathML':
    #==========================
        return MathML(deepcopy(self.__mathml))

    def substitute(self, name: str, symbol: str, missing_ok=False):
    #==============================================================
        if name not in self.__variables:
            if missing_ok:
                return
            raise ValueError(f'Variable {name} not in formulae, cannot substitute it')
        elif name == symbol:
            return
        elif symbol in self.__variables:
            raise ValueError(f'Symbol {symbol} is already in formulae, cannot substitute to it')
        for element in self.__variables[name]:
            element.text = symbol
        self.__variables[symbol] = self.__variables[name]
        del self.__variables[name]
        self.__build_equations()

    def __build_equations(self):
    #===========================
        self.__equalities: list[Equation] = []
        for equation in self.__mathml.findall(f'.//{MATHML_NS.apply}/{MATHML_NS.eq}'):
            lhs = sympy_from_mathml(equation.getnext())
            rhs = sympy_from_mathml(equation.getnext().getnext())
            if isinstance(rhs, sympy.Derivative):
                if isinstance(lhs, sympy.Derivative):
                    raise ValueError(f'Unsupported form of ODE equation: {equation.xml}')
                self.__equalities.append(Equation(rhs, lhs))
            elif isinstance(lhs, sympy.Symbol) or isinstance(lhs, sympy.Derivative):
                self.__equalities.append(Equation(lhs, rhs))
            else:
                raise ValueError(f'Unsupported form of equation: {equation.xml}')

#===============================================================================
