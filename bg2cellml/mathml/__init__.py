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
from copy import deepcopy

#===============================================================================

import lxml.etree as etree

#===============================================================================

def equal(term_0: str, term_1: str) -> str:
#==========================================
    if term_0 == term_1:
        return ''
    elif term_0 == '<cn>0.0</cn>':
        return f'''<apply>
    <eq/>
    {term_1}
    {term_0}
</apply>
'''
    else:
        return f'''<apply>
    <eq/>
    {term_0}
    {term_1}
</apply>
'''

def var_symbol(symbol: str) -> str:
#==================================
    return f'<ci>{symbol}</ci>'

def sum_variables(symbols: list[str]) -> str:
#============================================
    if len(symbols) == 0:
        return '<cn>0.0</cn>'
    elif len(symbols) == 1:
        return var_symbol(symbols[0])
    else:
        return f'''<apply>
    <plus/>
    {'\n    '.join([var_symbol(symbol) for symbol in symbols])}
</apply>
'''

#===============================================================================

class MathML:
    def __init__(self, mathml: etree.Element):
        self.__mathml = mathml
        self.__symbols = defaultdict(list)
        for element in self.__mathml.findall('.//{http://www.w3.org/1998/Math/MathML}ci'):
            self.__symbols[element.text].append(element)

    @classmethod
    def from_string(cls, formulae: str) -> 'MathML':
        parser = etree.XMLParser(remove_blank_text=True)
        try:
            return cls(etree.fromstring(formulae, parser))
        except etree.XMLSyntaxError as error:
            raise ValueError(error)

    def __str__(self):
        return etree.tostring(self.__mathml, encoding='unicode', pretty_print=True)

    @property
    def mathml(self) -> etree.Element:
        return self.__mathml

    @property
    def symbols(self) -> list[str]:
        return list(self.__symbols.keys())

    def copy(self) -> 'MathML':
    #==========================
        return MathML(deepcopy(self.__mathml))

    def substitute(self, symbol: str, replacement: str):
    #===================================================
        if symbol not in self.__symbols:
            raise ValueError(f'Symbol {symbol} not in formulae, cannot substitute it')
        elif replacement in self.__symbols:
            raise ValueError(f'Symbol {replacement} is already in formulae, cannot substitute to it')
        for element in self.__symbols[symbol]:
            element.text = replacement
        self.__symbols[replacement] = self.__symbols[symbol]
        del self.__symbols[symbol]

#===============================================================================
