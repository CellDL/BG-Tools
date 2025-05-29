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

BONDGRAPH_ELEMENTS = [
    'bgf:ChemicalConcentration',
    'bgf:ChemicalReaction',

    'bgf:ElectricalCapacitor',
    'bgf:ElectricalResistor',
]

#===============================================================================

BONDGRAPH_JUNCTIONS = [
    'bgf:OneNode',
    'bgf:TransformNode',
    'bgf:ZeroNode',
]

#===============================================================================
#===============================================================================

BONDGRAPH_MODELS = f"""
SELECT DISTINCT ?model
WHERE {{
    ?model a bg:BondGraph .
}} ORDER BY ?model"""

#===============================================================================

MODEL_ELEMENTS = f"""
SELECT DISTINCT ?element ?type
WHERE {{
    %MODEL% bg:hasBondElement ?element .
    ?element a ?type .
    OPTIONAL {{ ?element bgf:parameterValue ?param }}
    OPTIONAL {{ ?element bgf:stateValue ?state }}
    FILTER (?type IN ({', '.join(BONDGRAPH_ELEMENTS)}))
}} ORDER BY ?element"""

#===============================================================================

MODEL_JUNCTIONS = f"""
SELECT DISTINCT ?junction ?type ?domain
WHERE {{
    %MODEL% bg:hasJunctionStructure ?junction .
    ?junction a ?type .
    OPTIONAL {{ ?junction bgf:hasDomain ?domain }}
    FILTER (?type IN ({', '.join(BONDGRAPH_JUNCTIONS)}))
}} ORDER BY ?junction"""

#===============================================================================

MODEL_BONDS = f"""
SELECT DISTINCT ?bond ?source ?target
WHERE {{
    %MODEL% bg:hasPowerBond ?bond .
    ?bond bgf:hasSource ?source .
    ?bond bgf:hasTarget ?target .
}} ORDER BY ?bond ?source ?target"""

#===============================================================================
