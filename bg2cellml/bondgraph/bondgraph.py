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

from pathlib import Path
from typing import Optional

from pprint import pprint

#===============================================================================

import rdflib
import networkx as nx

#===============================================================================

from ..rdf import Labelled, NamespaceMap
from ..units import Value

from .framework import BondgraphFramework as FRAMEWORK
from .namespaces import NAMESPACES

#===============================================================================
#===============================================================================

ELEMENT_VARIABLES = f"""
    SELECT DISTINCT ?symbol ?value
    WHERE {{
        %ELEMENT_URI% bgf:hasVariable [
            bgf:hasSymbol ?symbol ;
            bgf:hasValue ?value
        ] .
    }}"""

#===============================================================================
#===============================================================================

MODEL_ELEMENTS = f"""
    SELECT DISTINCT ?uri ?type ?label
    WHERE {{
        %MODEL% bg:hasBondElement ?uri .
        ?uri a ?type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        FILTER (?type IN ({', '.join(FRAMEWORK.element_classes())}))
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphElement(Labelled):
    def __init__(self, uri: str, template: str, label: Optional[str], values: dict[str, Value]):
        super().__init__(uri, label)
        # print(uri, [str(v) for v in params.values()], [str(v) for v in states.values()])
        self.__template = FRAMEWORK.element(template)
        if self.__template is None:
            raise ValueError(f'Unknown BondElement {template} for node {uri}')
        self.__relation = self.__template.constitutive_relation
        self.__variables = self.__template.variables

#===============================================================================
#===============================================================================

MODEL_BONDS = f"""
    SELECT DISTINCT ?uri ?source ?target ?label
    WHERE {{
        %MODEL% bg:hasPowerBond ?uri .
        ?uri bgf:hasSource ?source .
        ?uri bgf:hasTarget ?target .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }} ORDER BY ?uri ?source ?target"""



#===============================================================================

class BondgraphBond(Labelled):
    def __init__(self, model: 'BondgraphModel', uri: str, source: str, target: str, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__source = source
        self.__target = target

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

#===============================================================================
#===============================================================================

MODEL_JUNCTIONS = f"""
    SELECT DISTINCT ?uri ?type ?label ?flow ?potential
    WHERE {{
        %MODEL% bg:hasJunctionStructure ?uri .
        ?uri a ?type .
        OPTIONAL {{ ?uri rdfs:label ?label }}
        OPTIONAL {{
            ?uri bgf:hasFlow [
                bgf:hasValue ?flow
            ]
        }}
        OPTIONAL {{
            ?uri bgf:hasPotential [
                bgf:hasValue ?potential
            ]
        }}
        FILTER (?type IN ({', '.join(FRAMEWORK.junction_classes())}))
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphJunction(Labelled):
    def __init__(self, uri, type: str, label: Optional[str], flow: Optional[rdflib.Literal], potential: Optional[rdflib.Literal]):
        super().__init__(uri, label)
        self.__junction = FRAMEWORK.junction(type)
        self.__flow = Value(flow) if flow is not None else None
        self.__potential = Value(potential) if potential is not None else None

        print(uri, self.__flow, self.__potential)

#===============================================================================
#===============================================================================

"""
:u_0
    a bgf:ZeroNode ;    ## 0-nodes can have a potential value; 1-nodes can have a flow value
    bgf:hasVariable [  ## has to be compatible with assigned domain....
        bgf:hasValue "11 J/coulomb"^^cdt:ucum
    ] .

The terminals of a JS network are the BEs it connects to and these
determine possible potential (u) and flow (v) symbols for JS nodes.

For each JS subgraph/network (reactions will divide JS network):
    Build flow and potential matrices to determine their equations.
        This will include transform nodes (Tf and Gy).


Each BE gets specific symbols for its parameter, state, and powerport
variables (and constants, when the same symbol has different values).
    ==> constants' registry (node, symbol, value)


:R_C_R_circuit
[[':C_1', 'bgf:ElectricalCapacitor'],
 [':R_0', 'bgf:ElectricalResistor'],
 [':R_1', 'bgf:ElectricalResistor']]
[[':u_0', 'bgf:ZeroNode', 'bgf:Electrical'],
 [':u_1', 'bgf:ZeroNode', 'bgf:Electrical'],
 [':v_0', 'bgf:OneNode', 'bgf:Electrical']]


[[':u_0.v_0', ':u_0', ':v_0'],
 [':u_1.C_1', ':u_1', ':C_1'],
 [':u_1.R_1', ':u_1', ':R_1'],
 [':v_0.R_0', ':v_0', ':R_0'],
 [':v_0.u_1', ':v_0', ':u_1']]
"""

#===============================================================================
#===============================================================================

BONDGRAPH_MODELS = f"""
    SELECT DISTINCT ?uri ?label
    WHERE {{
        ?uri a bg:BondGraph .
        OPTIONAL {{ ?uri rdfs:label ?label }}
    }} ORDER BY ?uri"""

#===============================================================================

class BondgraphModel(Labelled):
    def __init__(self, source: 'BondgraphModelSource', uri: str, label: Optional[str]=None):
        super().__init__(uri, label)
        self.__elements = []
        for row in source.sparql_query(MODEL_ELEMENTS.replace('%MODEL%', uri)):
            element_uri = row[0]
            values = {row[1]: Value(row[2]) for row in
                        source.sparql_query(ELEMENT_VARIABLES.replace('%ELEMENT_URI%', element_uri))}
            self.__elements.append(BondgraphElement(element_uri, row[1], row[2], values))
        self.__junctions = [BondgraphJunction(*row)
                                for row in source.sparql_query(MODEL_JUNCTIONS.replace('%MODEL%', uri))]
        self.__bonds = [BondgraphBond(self, *row)
                            for row in source.sparql_query(MODEL_BONDS.replace('%MODEL%', uri))]

        # Construct network graph of PowerBonds
        self.__make_bond_network()

        # Check domain consistency and identify gyrators
        self.__check_domains()

    def __assign_symbols(self):
    #==========================
        pass

    def __check_domains(self):
    #=========================
        pass

    def __make_bond_network(self):
    #=============================
        self.__graph = nx.DiGraph()
        for element in self.__elements:
            self.__graph.add_node(element.uri)
            # Needs power ports of elements as nodes....

        for junction in self.__junctions:
            self.__graph.add_node(junction.uri)

        for bond in self.__bonds:
            if (bond_source := bond.source) not in self.__graph:
                raise ValueError(f'No element or junction for source {bond_source} of bond {bond.uri}')
            if (bond_target := bond.target) not in self.__graph:
                raise ValueError(f'No element or junction for target {bond_target} of bond {bond.uri}')
            self.__graph.add_edge(bond_source, bond_target)
        if not nx.is_weakly_connected(self.__graph):
            raise ValueError('Resulting network graph is disconnected')

#===============================================================================
#===============================================================================

class BondgraphModelSource:
    def __init__(self, bondgraph_path: str):
        self.__namespace_map = NamespaceMap(NAMESPACES)
        self.__namespace_map.add_namespace('', f'{Path(bondgraph_path).resolve().as_uri()}#')
        self.__sparql_prefixes = self.__namespace_map.sparql_prefixes()
        self.__rdf = rdflib.Graph()
        self.__rdf.parse(bondgraph_path, format='turtle')
        self.__models = [BondgraphModel(self, *row) for row in self.sparql_query(BONDGRAPH_MODELS)]

    def sparql_query(self, query: str) -> list[list]:
    #================================================
        query_result = self.__rdf.query(f'{self.__sparql_prefixes}\n{query}')
        if query_result is not None:
            return [[self.__namespace_map.simplify(term) for term in row]   # type: ignore
                                                                for row in query_result]
        return []

#===============================================================================
#===============================================================================
