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

from .helper import compare_bondgraph_models, compare_simulation

#===============================================================================

def test_simple_electrical_circuit():
#====================================
    compare_simulation('../examples/example_A1.ttl',
                       '../pmr/Simple_electrical_circuit/FAIRDO BG example 3.1')

def test_simple_reaction():
#==========================
    compare_simulation('../examples/example_B1.ttl',
                       '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

def test_simplified_reaction():
#==============================
    compare_simulation('../examples/example_B1_simplified.ttl',
                       '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

def test_bondgraph_simplification():
#===================================
    compare_bondgraph_models('../examples/example_B1.ttl',
                             '../examples/example_B1_simplified.ttl',
                             '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

#===============================================================================
#===============================================================================
